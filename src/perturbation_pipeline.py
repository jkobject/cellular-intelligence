"""Perturbation modeling pipeline for Exercise 2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import anndata as ad
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def load_experiment_data(
    data_dir: str | Path,
    rounds: range | list[int] = range(1, 11),
) -> ad.AnnData:
    """Load all rounds into one consistent AnnData object.

    The output contains:
    - `X`: expression state for each `(cell, round)` row
    - `obs['round']`: timepoint index
    - `obsm['perturbation']`: perturbation vector applied at that round
    """
    base_dir = Path(data_dir)
    round_data: list[ad.AnnData] = []

    for round_id in rounds:
        round_adata = ad.read_h5ad(base_dir / f"expression_round_{round_id}.h5ad")
        round_adata.obs["round"] = round_id
        round_adata.obsm["perturbation"] = np.load(
            base_dir / f"perturbation_round_{round_id}.npy"
        ).astype(np.float32, copy=False)
        round_data.append(round_adata)

    return ad.concat(round_data, axis=0, join="outer", index_unique=None)


def _extract_cell_ids(obs_names: list[str] | np.ndarray) -> np.ndarray:
    """Extract per-cell IDs from names like `Round3_17` -> `17`."""
    suffixes = np.array([str(name).rsplit("_", maxsplit=1)[-1] for name in obs_names])
    if np.all(np.char.isnumeric(suffixes)):
        return suffixes.astype(np.int64)
    return suffixes


def build_transition_dataset(
    adata: ad.AnnData,
    source_timepoints: list[int] | None = None,
    mode: str = "forward_transition",
) -> dict[str, np.ndarray]:
    """Build consecutive-round samples for one supervised task mode.

    Supported modes:
    - `forward_transition`: `[X_t, P_t] -> X_{t+1}`
    - `inverse_perturbation`: `(X_{t+1} - X_t) -> P_t`
    - `inverse_perturbation_two_states`: `[X_t, X_{t+1}] -> P_t`
    """
    if "round" not in adata.obs:
        raise ValueError("`adata.obs['round']` is required.")
    if "perturbation" not in adata.obsm:
        raise ValueError("`adata.obsm['perturbation']` is required.")

    rounds = adata.obs["round"].to_numpy(dtype=np.int64)
    expression = np.asarray(adata.X, dtype=np.float32)
    perturbation = np.asarray(adata.obsm["perturbation"], dtype=np.float32)
    cell_ids = _extract_cell_ids(adata.obs_names.to_numpy())

    unique_rounds = np.sort(np.unique(rounds))
    if source_timepoints is None:
        source_timepoints = [int(t) for t in unique_rounds[:-1]]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    src_tp_parts: list[np.ndarray] = []
    tgt_tp_parts: list[np.ndarray] = []

    for t in source_timepoints:
        current_idx = np.where(rounds == t)[0]
        next_idx = np.where(rounds == (t + 1))[0]
        if len(current_idx) == 0 or len(next_idx) == 0:
            continue

        next_lookup = {cell_ids[idx]: idx for idx in next_idx}
        matched_current_idx: list[int] = []
        matched_next_idx: list[int] = []

        for idx in current_idx:
            match_idx = next_lookup.get(cell_ids[idx])
            if match_idx is not None:
                matched_current_idx.append(idx)
                matched_next_idx.append(match_idx)

        if not matched_current_idx:
            continue

        matched_current = np.array(matched_current_idx, dtype=np.int64)
        matched_next = np.array(matched_next_idx, dtype=np.int64)

        x_t = expression[matched_current]
        x_tp1 = expression[matched_next]
        p_t = perturbation[matched_current]

        if mode == "forward_transition":
            x_mode = np.concatenate([x_t, p_t], axis=1)
            y_mode = x_tp1
        elif mode == "inverse_perturbation":
            x_mode = x_tp1 - x_t
            y_mode = p_t
        elif mode == "inverse_perturbation_two_states":
            x_mode = np.concatenate([x_t, x_tp1], axis=1)
            y_mode = p_t
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. "
                "Use 'forward_transition', 'inverse_perturbation', "
                "or 'inverse_perturbation_two_states'."
            )

        x_parts.append(x_mode)
        y_parts.append(y_mode)
        src_tp_parts.append(np.full(x_t.shape[0], t, dtype=np.int64))
        tgt_tp_parts.append(np.full(x_t.shape[0], t + 1, dtype=np.int64))

    if not x_parts:
        raise ValueError(
            "No transition samples could be built from the provided adata."
        )

    return {
        "X": np.concatenate(x_parts, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "source_timepoint": np.concatenate(src_tp_parts, axis=0),
        "target_timepoint": np.concatenate(tgt_tp_parts, axis=0),
    }


def split_dataset(
    adata: ad.AnnData,
    test_timepoints: list[int] | tuple[int, ...] = (9, 10),
    val_fraction: float = 0.2,
    random_state: int = 42,
    dataset_mode: str = "forward_transition",
) -> dict[str, Any]:
    """Split consecutive-round data by timepoint for test and random split for val.

    Test split is defined by `target_timepoint in test_timepoints`.
    Validation split is random over samples from non-test timepoints.
    """
    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError("`val_fraction` must be in [0, 1).")

    dataset = build_transition_dataset(adata, mode=dataset_mode)
    x = dataset["X"]
    y = dataset["y"]
    target_timepoint = dataset["target_timepoint"]

    is_test = np.isin(target_timepoint, np.array(test_timepoints, dtype=np.int64))
    test_indices = np.where(is_test)[0]
    train_pool_indices = np.where(~is_test)[0]

    if train_pool_indices.size == 0:
        raise ValueError("No train samples left after test_timepoint filtering.")

    rng = np.random.default_rng(random_state)
    shuffled_train_pool = rng.permutation(train_pool_indices)
    n_val = int(np.floor(shuffled_train_pool.size * val_fraction))

    val_indices = shuffled_train_pool[:n_val]
    train_indices = shuffled_train_pool[n_val:]

    return {
        "train": (x[train_indices], y[train_indices]),
        "val": (x[val_indices], y[val_indices]),
        "test": (x[test_indices], y[test_indices]),
        "indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        },
        "timepoints": {
            "train": target_timepoint[train_indices],
            "val": target_timepoint[val_indices],
            "test": target_timepoint[test_indices],
        },
    }


def build_mlp_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: list[int] | int = 128,
    random_state: int = 42,
    max_iter: int = 300,
) -> Pipeline:
    """Build MLP for `(input_dim, hidden_dim, output_dim)` mapping."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("`input_dim` and `output_dim` must be positive.")
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim]
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_dim),
                    activation="relu",
                    solver="adam",
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_lregression_model() -> Pipeline:
    """Build linear regression baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_xgboost_model(
    random_state: int = 42,
    n_estimators: int = 120,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> Any:
    """Build XGBoost regressor for multi-output prediction."""
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install it with `uv pip install xgboost`."
        ) from exc

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        multi_strategy="multi_output_tree",
        random_state=random_state,
        n_jobs=-1,
    )


def build_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    random_state: int = 42,
    mlp_hidden_dim: list[int] | int = 128,
    mlp_max_iter: int = 300,
) -> Any:
    """Build a model by name: `mlp`, `linear_regression`, or `xgboost`."""
    normalized_name = model_name.lower()

    if normalized_name == "mlp":
        return build_mlp_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mlp_hidden_dim,
            max_iter=mlp_max_iter,
            random_state=random_state,
        )
    if normalized_name in {"log_regression", "linear_regression"}:
        return build_lregression_model()
    if normalized_name == "xgboost":
        return build_xgboost_model(random_state=random_state)

    raise ValueError(
        f"Unsupported model_name: {model_name}. Use 'mlp', 'linear_regression', or 'xgboost'."
    )


def train(model: Any, train_data: tuple[np.ndarray, np.ndarray]) -> Any:
    """Fit model with `(X_train, y_train)` data."""
    x_train, y_train = train_data
    model.fit(x_train, y_train)
    return model


def merge_train_val_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge train and validation splits into one training pool."""
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    return (
        np.concatenate([x_train, x_val], axis=0),
        np.concatenate([y_train, y_val], axis=0),
    )


def run_kfold_cv(
    model: Any,
    train_data: tuple[np.ndarray, np.ndarray],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate one model with k-fold cross-validation on train data."""
    if n_splits < 2:
        raise ValueError("`n_splits` must be at least 2.")

    x_train, y_train = train_data
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []

    for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_train), start=1):
        fold_model = clone(model)
        fold_model.fit(x_train[fit_idx], y_train[fit_idx])
        fold_prediction = fold_model.predict(x_train[val_idx])
        metrics = evaluate_predictions(
            prediction=fold_prediction,
            ground_truth=y_train[val_idx],
        )
        metrics["fold"] = float(fold_id)
        fold_metrics.append(metrics)

    metric_names = ["mse", "rmse", "mae", "r2"]
    mean_metrics = {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }
    std_metrics = {
        name: float(np.std([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def select_model_with_kfold(
    train_data: tuple[np.ndarray, np.ndarray],
    candidate_models: Sequence[str] = ("mlp", "log_regression", "xgboost"),
    n_splits: int = 5,
    random_state: int = 42,
    mlp_hidden_dim: list[int] | int = 128,
    mlp_max_iter: int = 300,
) -> dict[str, Any]:
    """Run k-fold CV for candidate models and select best by mean RMSE."""
    x_train, y_train = train_data
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("Expected 2D arrays for `x_train` and `y_train`.")
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Could not infer valid input/output dimensions.")

    model_results: dict[str, dict[str, Any]] = {}

    for model_name in candidate_models:
        model = build_model(
            model_name=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            random_state=random_state,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_max_iter=mlp_max_iter,
        )
        model_results[model_name] = run_kfold_cv(
            model=model,
            train_data=train_data,
            n_splits=n_splits,
            random_state=random_state,
        )

    best_model_name = min(
        model_results,
        key=lambda name: model_results[name]["mean_metrics"]["rmse"],
    )

    return {
        "best_model_name": best_model_name,
        "cv_results": model_results,
    }


def train_with_model_selection(
    train_data: tuple[np.ndarray, np.ndarray],
    candidate_models: Sequence[str] = ("mlp", "log_regression", "xgboost"),
    n_splits: int = 5,
    random_state: int = 42,
    mlp_hidden_dim: list[int] | int = 128,
    mlp_max_iter: int = 300,
) -> dict[str, Any]:
    """Select the best model via k-fold CV and fit it on all train data."""
    selection = select_model_with_kfold(
        train_data=train_data,
        candidate_models=candidate_models,
        n_splits=n_splits,
        random_state=random_state,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_max_iter=mlp_max_iter,
    )

    x_train, y_train = train_data
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("Expected 2D arrays for `x_train` and `y_train`.")
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    best_model = build_model(
        model_name=selection["best_model_name"],
        input_dim=input_dim,
        output_dim=output_dim,
        random_state=random_state,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_max_iter=mlp_max_iter,
    )
    best_model.fit(x_train, y_train)

    return {
        "model": best_model,
        "best_model_name": selection["best_model_name"],
        "cv_results": selection["cv_results"],
    }


def evaluate_predictions(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute MSE, RMSE, MAE and R2."""
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape.")

    mse = mean_squared_error(ground_truth, prediction)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(ground_truth, prediction)),
        "r2": float(r2_score(ground_truth, prediction)),
    }


def evaluate_predictions_by_timepoint(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    timepoints: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate predictions overall and per target timepoint."""
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape.")
    if prediction.shape[0] != timepoints.shape[0]:
        raise ValueError("`timepoints` length must match number of samples.")

    result: dict[str, dict[str, float]] = {
        "overall": evaluate_predictions(prediction, ground_truth)
    }
    for timepoint in np.sort(np.unique(timepoints)):
        idx = timepoints == timepoint
        result[f"timepoint_{int(timepoint)}"] = evaluate_predictions(
            prediction[idx], ground_truth[idx]
        )
    return result


def test(
    model: Any,
    test_data: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray | dict[str, float]]:
    """Predict on test data and return predictions + metrics."""
    x_test, y_test = test_data
    prediction = model.predict(x_test)
    return {
        "prediction": prediction,
        "metrics": evaluate_predictions(prediction=prediction, ground_truth=y_test),
    }


def compute_umap_embeddings(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    expression_key: str = "X_umap_expression",
    perturbation_key: str = "X_umap_perturbation",
) -> ad.AnnData:
    """Compute UMAP embeddings for expression and perturbation spaces."""
    import umap

    if "perturbation" not in adata.obsm:
        raise ValueError("`adata.obsm['perturbation']` is required.")

    expression = np.asarray(adata.X, dtype=np.float32)
    perturbation = np.asarray(adata.obsm["perturbation"], dtype=np.float32)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init="random",
    )
    adata.obsm[expression_key] = reducer.fit_transform(expression)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init="random",
    )
    adata.obsm[perturbation_key] = reducer.fit_transform(perturbation)

    return adata


def plot_umap_by_timepoint(
    adata: ad.AnnData,
    expression_key: str = "X_umap_expression",
    perturbation_key: str = "X_umap_perturbation",
    point_size: int = 6,
    alpha: float = 0.7,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot expression and perturbation UMAPs colored by `obs['round']`."""
    import matplotlib.pyplot as plt

    if "round" not in adata.obs:
        raise ValueError("`adata.obs['round']` is required.")
    if expression_key not in adata.obsm:
        raise ValueError(f"`adata.obsm['{expression_key}']` is missing.")
    if perturbation_key not in adata.obsm:
        raise ValueError(f"`adata.obsm['{perturbation_key}']` is missing.")

    rounds = adata.obs["round"].to_numpy(dtype=np.int64)
    unique_rounds = np.sort(np.unique(rounds))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    scatter = axes[0].scatter(
        adata.obsm[expression_key][:, 0],
        adata.obsm[expression_key][:, 1],
        c=rounds,
        cmap="tab10",
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )
    axes[0].set_title("Expression UMAP")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")

    axes[1].scatter(
        adata.obsm[perturbation_key][:, 0],
        adata.obsm[perturbation_key][:, 1],
        c=rounds,
        cmap="tab10",
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )
    axes[1].set_title("Perturbation UMAP")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    colorbar = fig.colorbar(scatter, ax=axes, ticks=unique_rounds)
    colorbar.set_label("timepoint")

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200, bbox_inches="tight")

    return fig
