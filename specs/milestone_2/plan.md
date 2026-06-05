# Milestone 2 — Plan

## Estimated Duration

~6 weeks (one focused developer)

---

## Tasks

### 1. VT Dataset Preparation
- Load VT dataset (140,885 obs, 18,285 env_ids)
- Drop rows missing **both** `population` and `rs`; retain rows missing only one (cleaned: ~82,773 obs, ~11,723 env_ids)
- Retrieve weather variables and document remaining missing value counts
- Confirm 58-predictor column schema

### 2. Planting Date Dataset Preparation
- Load Planting Date dataset (1,796 obs, 22 env_ids)
- Document 72 observations with missing weather predictors (Mt. Vernon 2024–2025)
- Confirm column schema matches VT

### 3. Preprocessing Pipeline
- Impute missing continuous values (median per phenological-stage group, fitted on train envs only)
- Encode `rm` → `rm_id` (integer); fit mapping on train environments only; place `rm_id` as the last feature column so categorical features occupy predictable tail positions
- Standardize all continuous features and `yield` (zero mean, unit variance, fitted on train envs only)
- Persist fitted scalers alongside model checkpoint

### 4. Environment-Based Split
- VT: assign environments (not observations) to splits — Train 70%, Val 15%, Test 15%
- Planting Date: same 70/15/15 split by environment (only 22 env_ids; all three splits will be small)
- Verify no environment leaks across splits

### 5. Baseline MLP
- Implement `YieldMLP` in PyTorch; wire up training loop (MSE, Adam/AdamW)
- Verify end-to-end on VT train split; confirm val RMSE before NAS

### 6. FT-Transformer Implementation
- Implement `YieldFTTransformer` with `_FeatureTokenizer` + Transformer Encoder + head
- Verify identical forward-pass interface to `YieldMLP` (flat float32 in → scalar out)
- Test `freeze_backbone` / `unfreeze_backbone` for transfer learning workflow

### 7. NAS — VT Dataset
- Run MLP NAS (75 trials) and FT-Transformer NAS (50 trials) on VT data
- Objective: minimize val RMSE on held-out VT environments
- Log all trials; persist `optuna_study.pkl`

### 8. Pre-training on VT
- Select best FT-Transformer architecture from NAS
- Retrain on VT train + val environments combined
- Evaluate once on VT test environments; write to `eval_report_vt.json`
- Save `model_vt.pt`

### 9. Fine-tuning on Planting Date
- Load `model_vt.pt`; replace head with a freshly initialized linear layer
- Stage 1: freeze backbone, train head only (higher LR, ~20–30 epochs)
- Stage 2: unfreeze backbone, end-to-end fine-tuning (LR ≤ 1e-4, early stopping)
- Evaluate on Planting Date test environments; save `model_planting.pt`

### 10. Serialization
- Save all artifacts: `model_vt.pt`, `model_planting.pt`, `preprocessor.pkl`, `optuna_study.pkl`, `eval_report.json`
- `train.py` runs end-to-end from raw CSVs to saved artifacts with a single command

---

## NAS Search Spaces

### YieldMLP

Given the large VT dataset, regularization is still a primary lever but a wider search is feasible:

| Hyperparameter | Range / Choices |
|---|---|
| `n_layers` | 1 – 4 |
| `hidden_size` | 32, 64, 128, 256 |
| `activation` | ReLU, GELU, SiLU |
| `dropout_rate` | 0.1 – 0.5 (step 0.05) |
| `use_batch_norm` | True / False |
| `learning_rate` | 1e-4 – 1e-2 (log scale) |
| `weight_decay` | 1e-5 – 1e-2 (log scale) |
| `batch_size` | 32, 64, 128 |
| `optimizer` | Adam, AdamW |

### YieldFTTransformer

| Hyperparameter | Range / Choices |
|---|---|
| `d_token` | 64, 128, 256 |
| `n_heads` | 4, 8 (all choices divisible into every `d_token`) |
| `n_layers` | 1, 2, 3 |
| `d_ffn_factor` | 1.333, 2.0, 4.0 |
| `dropout` | 0.0 – 0.3 (step 0.05) |
| `learning_rate` | 1e-4 – 1e-3 (log scale; lower ceiling than MLP) |
| `weight_decay` | 1e-5 – 1e-2 (log scale) |
| `batch_size` | 64, 128, 256 |
| `optimizer` | Adam, AdamW |

---

## Environment-Based Data Split

| Dataset | Split | Fraction | Environments (approx) | Observations (approx) |
|---|---|---|---|---|
| VT | Train | 70% | ~8,206 | ~57,941 |
| VT | Val | 15% | ~1,759 | ~12,416 |
| VT | Test | 15% | ~1,759 | ~12,416 |
| Planting Date | Train | 70% | ~15 | ~1,257 |
| Planting Date | Val | 15% | ~3 | ~270 |
| Planting Date | Test | 15% | ~4 | ~269 |

Split is over environment IDs, not observations.

---

## Post-Training: Extrapolation

Once models are finalized:
- Assemble extrapolation feature table covering all target (location, crop, management) combinations
- Run batch inference (no-grad, model in eval mode)
- Write `(location_id, crop_type, input_variables…, predicted_yield)` rows into the yield database
- The agent layer reads only from this database — the model is not called at inference time
