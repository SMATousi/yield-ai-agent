# Milestone 2 — Plan

## Estimated Duration

~4 weeks (one focused developer)

---

## Tasks

### 1. Dataset Integration
- Load and merge Planting Date Trial (1,800 obs) and Varieties Test (6,700 obs)
- Add `dataset_source` indicator column to distinguish sub-datasets
- Confirm the 50-predictor schema aligns between both sub-datasets; resolve any column mismatches

### 2. Preprocessing
- Build sklearn pipeline: imputation → encoding → standardization
- Fit transformers on train environments only (no leakage from val/test environments)
- Validate that environment IDs are preserved through the pipeline for CV splitting

### 3. Environment-Based Split
- Assign environments to splits by random shuffle of environment IDs (not observations):
  - Train: 70% of environments
  - Val: 15% of environments (NAS objective)
  - Test: 15% of environments (held out until final eval)
- Ensure both sub-datasets' environments are represented proportionally in each split

### 4. Baseline Model
- Implement a feed-forward MLP in PyTorch
- Wire up training loop with MSE loss and Adam optimizer
- Verify end-to-end training on the train split; check val RMSE before NAS

### 5. Optuna NAS Setup
- Define search space (see table below)
- Configure TPE sampler and MedianPruner
- Objective: minimize validation RMSE on the held-out environment val split
- Early stopping per trial: no improvement over 10 epochs

### 6. NAS Execution
- Run 75–150 trials (reduced from the generic plan due to ~8,500 obs; over-searching risks overfitting the val split)
- Log each trial: full hyperparameter set, train/val loss curve, best val RMSE
- Integrate MLflow or W&B callbacks; persist `optuna_study.pkl`

### 7. Final Model Training
- Select best trial's architecture and hyperparameters
- Retrain on train + val environments combined (85% of environments)
- Evaluate once on held-out test environments — no further tuning after this point

### 8. Serialization
- Save `model.pt`, `preprocessor.pkl`, `optuna_study.pkl`, `eval_report.json`
- Write reproducible `train.py` (CLI args for data path, output dir, random seed)

---

## Neural Architecture Search Space

Given the small dataset (~8,500 obs), the search space emphasizes regularization:

| Hyperparameter | Range / Choices |
|---|---|
| `n_layers` | 1 – 4 (capped lower than generic plan) |
| `hidden_size` | 32, 64, 128, 256 |
| `activation` | ReLU, GELU, SiLU |
| `dropout_rate` | 0.1 – 0.5 (step 0.05; non-zero floor) |
| `use_batch_norm` | True / False |
| `learning_rate` | 1e-4 – 1e-2 (log scale) |
| `weight_decay` | 1e-5 – 1e-2 (log scale; wider range for stronger regularization) |
| `batch_size` | 32, 64, 128 |
| `optimizer` | Adam, AdamW |

All layers share the same `hidden_size` and `activation`. Skip connections are post-MVP.

---

## Environment-Based Data Split

| Split | Fraction of Environments | Purpose |
|---|---|---|
| Train | 70% | Model training within each NAS trial |
| Val | 15% | NAS objective (val RMSE on unseen environments) |
| Test | 15% | Held out — evaluated once after NAS |

Random split is over environment IDs, not observations, to prevent data leakage across fields.

---

## Post-Training: Extrapolation

Once the model is finalized:
- Assemble extrapolation feature table covering all target (location, crop, management) combinations
- Run batch inference (no-grad, model in eval mode)
- Write `(location_id, crop_type, input_variables…, predicted_yield)` rows into the yield database
- The agent layer reads only from this database — the model is not called at inference time
