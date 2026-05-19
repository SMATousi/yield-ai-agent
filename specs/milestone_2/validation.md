# Milestone 2 — Validation

## Done When

Best model is serialized to disk with a reproducible training script and an evaluation report showing cross-environment generalization metrics.

---

## Validation Strategy

**Environment-based holdout** — entire environments are withheld from training. This tests whether the model generalizes to unseen fields, which is the real deployment scenario. Random-observation splits are not used as they inflate metrics by leaking environment-specific signals into the val/test sets.

| Split | Environments | Observations (approx) |
|---|---|---|
| Train | 70% of 608 (≈426) | ~5,950 |
| Val | 15% of 608 (≈91) | ~1,275 |
| Test | 15% of 608 (≈91) | ~1,275 |

---

## Evaluation Metrics

Reported on the held-out test environments:

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error (primary) |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| MAPE | Mean Absolute Percentage Error |

Report metrics:
- Overall (all test observations)
- Stratified by sub-dataset (Planting Date Trial vs. Varieties Test)
- Stratified by environment if sample size allows

---

## NAS Experiment Tracking

Each Optuna trial must log:
- Full hyperparameter set
- Train and val loss curves
- Best val RMSE (on held-out val environments)

Use MLflow or W&B study callbacks; store `optuna_study.pkl` for reproducibility. Val RMSE must be computed on the environment-holdout split, not a random sample of observations.

---

## Acceptance Checklist

- [ ] Both sub-datasets merged; `dataset_source` column present
- [ ] Preprocessor fitted on train environments only (no leakage)
- [ ] Environment-based split verified: no environment appears in more than one split
- [ ] NAS completed with at least 75 trials logged
- [ ] Best trial selected by lowest val RMSE on held-out environments
- [ ] Final model retrained on train + val environments combined
- [ ] Test environments evaluated exactly once; results written to `eval_report.json`
- [ ] Metrics reported overall, by sub-dataset, and by environment where feasible
- [ ] All artifacts present: `model.pt`, `preprocessor.pkl`, `optuna_study.pkl`, `eval_report.json`, `train.py`
- [ ] `train.py` runs end-to-end from raw data to saved artifacts with a single command
