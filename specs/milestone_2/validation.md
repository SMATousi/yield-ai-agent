# Milestone 2 — Validation

## Done When

Best model is serialized to disk with a reproducible training script and an evaluation report showing cross-environment generalization metrics.

---

## Validation Strategy

**Environment-based holdout** — entire environments are withheld from training. This tests whether the model generalizes to unseen fields, which is the real deployment scenario. Random-observation splits are not used as they inflate metrics by leaking environment-specific signals into the val/test sets.

### VT Dataset (Pre-training)

| Split | Environments (approx) | Observations (approx) |
|---|---|---|
| Train | 70% of 11,723 (~8,206) | ~57,941 |
| Val | 15% of 11,723 (~1,759) | ~12,416 |
| Test | 15% of 11,723 (~1,759) | ~12,416 |

### Planting Date Dataset (Fine-tuning)

| Split | Environments (approx) | Observations (approx) |
|---|---|---|
| Train | 70% of 22 (~15) | ~1,257 |
| Val | 15% of 22 (~3) | ~270 |
| Test | 15% of 22 (~4) | ~269 |

The Planting Date dataset has only 22 env_ids; test metrics will be reported per-environment.

---

## Evaluation Metrics

Reported on the held-out test environments for both VT and Planting Date:

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error in original yield units (primary) |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| MAPE | Mean Absolute Percentage Error |

Report metrics:
- Overall (all test observations)
- Stratified by environment where sample size allows

---

## NAS Experiment Tracking

Each Optuna trial must log:
- Full hyperparameter set
- Train and val loss curves
- Best val RMSE (on held-out val environments, in original yield units)

Use MLflow or W&B study callbacks; store `optuna_study.pkl` for reproducibility. Val RMSE must be computed on the environment-holdout split, not a random sample of observations.

---

## Acceptance Checklist

- [ ] VT and Planting Date datasets loaded and preprocessed independently
- [ ] Preprocessors fitted on train environments only (no leakage from val/test)
- [ ] Environment-based split verified: no environment appears in more than one split
- [ ] MLP NAS completed with ≥75 trials on VT; FT-Transformer NAS with ≥50 trials on VT
- [ ] Best FT-Transformer architecture selected by lowest val RMSE on held-out VT environments
- [ ] Pre-trained model (`model_vt.pt`) trained on VT train + val environments; evaluated once on VT test
- [ ] Fine-tuned model (`model_planting.pt`) produced via 2-stage fine-tuning on Planting Date
- [ ] Test environments for both datasets evaluated exactly once; results in `eval_report.json`
- [ ] Metrics reported overall and per-environment for Planting Date
- [ ] All artifacts present: `model_vt.pt`, `model_planting.pt`, `preprocessor.pkl`, `optuna_study.pkl`, `eval_report.json`
- [ ] `train.py` runs end-to-end from raw CSVs to saved artifacts with a single command
