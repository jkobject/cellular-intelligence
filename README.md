# Cellular Intelligence Exercise 2

## Problem
We want to model how perturbations change cell state over time.

Data is provided for 10 rounds:
- `expression_round_<t>.h5ad`: expression state at timepoint `t`
- `perturbation_round_<t>.npy`: perturbation applied at timepoint `t`

Goal:
- Learn a transition model that predicts the next expression state from the current state and perturbation:
- `[X_t, P_t] -> X_{t+1}`

## How We Solved It
We built a simple, scalable baseline pipeline in [`src/perturbation_pipeline.py`](/Users/jkobject/Documents/code/cellular-intelligence/src/perturbation_pipeline.py):

1. Load all rounds into one consistent `AnnData`.
2. Build transition samples aligned by cell ID across consecutive timepoints.
3. Split data with time awareness:
- test split is by timepoint (default target rounds 9 and 10)
- train/val split is random inside remaining (non-test) timepoints
4. Train robustly using k-fold cross-validation model selection over:
- `mlp`
- `log_regression` (linear regression baseline for continuous targets)
- `xgboost`
5. Fit the best model on the full training pool and evaluate on held-out test timepoints.
6. Generate UMAP visualizations for perturbation and expression spaces, colored by timepoint.

## Main Functions
Core functions are in [`src/perturbation_pipeline.py`](/Users/jkobject/Documents/code/cellular-intelligence/src/perturbation_pipeline.py):

- `load_experiment_data(data_dir, rounds=range(1, 11)) -> AnnData`
- `build_transition_dataset(adata) -> dict`
- `split_dataset(adata, test_timepoints=(9, 10), val_fraction=0.2, random_state=42) -> dict`
- `merge_train_val_splits(splits) -> (X, y)`
- `build_model(model_name, input_dim, output_dim, ...) -> model`
- `train(model, train_data) -> model`
- `run_kfold_cv(model, train_data, n_splits=5, ...) -> dict`
- `select_model_with_kfold(train_data, candidate_models=(...), ...) -> dict`
- `train_with_model_selection(train_data, candidate_models=(...), ...) -> dict`
- `test(model, test_data) -> {"prediction", "metrics"}`
- `evaluate_predictions(prediction, ground_truth) -> metrics`
- `evaluate_predictions_by_timepoint(prediction, ground_truth, timepoints) -> dict`
- `compute_umap_embeddings(adata, ...) -> AnnData`
- `plot_umap_by_timepoint(adata, ...) -> Figure`

## Notebook
End-to-end workflow:
- [`exercise2_perturbation_modeling.ipynb`](/Users/jkobject/Documents/code/cellular-intelligence/exercise2_perturbation_modeling.ipynb)

It covers:
- data loading
- time-aware split
- k-fold model selection
- final test evaluation
- per-timepoint metrics
- UMAP plots

## How To Run
```bash
uv venv
uv pip install --python .venv/bin/python numpy anndata scikit-learn jupyter ipykernel pandas matplotlib umap-learn xgboost
```

Then run:
```bash
.venv/bin/python -m jupyter notebook
```

Open:
- `exercise2_perturbation_modeling.ipynb`

## What We Learned
- Time-aware splitting matters: random global split can leak temporal information.
- A simple transition formulation `[X_t, P_t] -> X_{t+1}` is clear and works well as a first baseline.
- Cross-validation improves robustness compared to one train/val split.
- Strong linear baselines can be very competitive on this toy embedded dataset.
- UMAP helps sanity-check whether timepoints and perturbation structure are separable.

## Next Improvements
- Hyperparameter search inside each model family (not only model-family selection).
- Group-aware CV by cell ID or source timepoint to reduce leakage risk further.
- Sequence models (RNN/Transformer/state-space) for multi-step rollout, not only one-step transition.
- Better uncertainty estimation (ensembles or quantile models).
