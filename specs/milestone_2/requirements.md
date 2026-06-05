# Milestone 2 — Requirements

## Goal

A trained, evaluated yield prediction model selected via neural architecture search, using a transfer-learning regime: pre-train on the large VT dataset, then fine-tune on the smaller Planting Date dataset.

---

## Problem Statement

Supervised regression: predict crop yield from a 58-predictor feature vector describing planting management, crop variety, and environment (soil + weather) at a given location and year. Cross-environment generalization is the primary deployment concern.

---

## Datasets

Both datasets share the same 60-column schema: 58 predictors + `yield` + `env_id`. Each `env_id` encodes a unique (latitude, longitude, year) combination.

### VT Dataset — Pre-training Corpus

| Attribute | Value |
|---|---|
| Total observations | 140,885 |
| Unique env_ids | 18,285 |
| After dropping rows missing **both** population and row spacing | 82,773 obs / 11,723 env_ids |

Missing values (full dataset):
- `population`: 31,158 observations
- `rs` (row spacing): 46,911 observations
- Weather variables: TBD (data retrieval pending)

Rows missing **both** population and row spacing are excluded before training. Rows missing only one are retained.

### Planting Date Dataset — Fine-tuning Target

| Attribute | Value |
|---|---|
| Observations | 1,796 |
| Unique env_ids | 22 |
| Missing values | 72 observations (weather predictors; Mt. Vernon 2024–2025) |

Missing weather values are imputed (median per phenological-stage group, fitted on train environments only).

---

## Input Features (58 Predictors)

### Management

| Column | Type | Description |
|---|---|---|
| `pday` | Continuous | Planting date as day of year |
| `rm` | Categorical | Relative maturity group |
| `rs` | Continuous | Row spacing (inches) |
| `population` | Continuous | Planting density (plants/acre, thousands) |

### Weather (28 features)

Seven weather variables summarized over four phenological stages: **V**, **R1**, **R3**, **R5**.

Variables: `rain`, `vpd`, `srad`, `ptt`, `et0_hs`, `w_def`, `tmean`

Column naming: `<variable>_<stage>` — e.g., `rain_V`, `srad_R5`, `tmean_R3`.

### Soil (32 features)

Eight soil properties at four depth breakpoints: **5 cm**, **15 cm**, **30 cm**, **60 cm** (DSSAT nomenclature).

Properties: `sloc`, `slcl`, `slsi`, `ssks`, `ssat`, `slll`, `slhw`, `awc`

Column naming: `<property>_<depth>` — e.g., `slcl_5`, `awc_30`, `sloc_60`.

---

## Target Variable

| Variable | Type | Unit |
|---|---|---|
| `yield` | Continuous | bu/acre (standardized at preprocessing) |

---

## Model Architectures

Both architectures share the same input/output contract (flat float32 tensor in, scalar yield out) and are evaluated via NAS.

### YieldMLP
Feed-forward MLP with variable depth and width.

### YieldFTTransformer
FT-Transformer (Gorishniy et al., 2021) adapted for tabular yield prediction:

1. **Feature Tokenizer** — projects each feature into a shared `d_token`-dimensional space:
   - Numerical: `token_i = x_i · W_i + b_i` (per-feature weight vector)
   - Categorical (`rm`): standard embedding lookup
2. **[CLS] token** — learnable token prepended to the feature sequence
3. **Transformer Encoder** — L layers of pre-norm multi-head self-attention + GELU FFN
4. **Head** — linear projection on the [CLS] output → scalar yield

The Feature Tokenizer + Transformer Encoder form the **backbone**; the head is the only layer replaced during fine-tuning.

**Fixed design decisions (both models):**
- Input: flat float32 feature tensor; categorical features occupy the last `len(cat_cardinalities)` positions
- Output: single scalar, linear activation
- Loss: MSE
- Optimizer: Adam or AdamW (NAS choice)

---

## Transfer Learning Strategy

| Phase | Dataset | Action |
|---|---|---|
| Pre-training | VT (82,773 obs) | NAS over FT-Transformer hyperparameters; train best model to convergence |
| Fine-tuning (stage 1) | Planting Date (1,796 obs) | Freeze backbone; train head only |
| Fine-tuning (stage 2) | Planting Date (1,796 obs) | Unfreeze backbone; end-to-end fine-tuning with low LR |

---

## Preprocessing Pipeline

1. For VT: drop rows missing both `population` and `rs`; retain rows missing only one
2. Impute remaining missing continuous values (median per env stage group, fitted on train envs)
3. Encode `rm` as integer ID (`rm_id`); fit mapping on train environments only
4. Standardize all continuous features (zero mean, unit variance, fitted on train envs)
5. Standardize `yield` (zero mean, unit variance) — invert at evaluation time
6. Persist scaler alongside model checkpoint

---

## Outputs / Artifacts

| Artifact | Description |
|---|---|
| `model_vt.pt` | FT-Transformer backbone pre-trained on VT |
| `model_planting.pt` | Fine-tuned model for Planting Date dataset |
| `preprocessor.pkl` | Fitted sklearn scalers / encoders |
| `optuna_study.pkl` | Full Optuna study for reproducibility |
| `eval_report.json` | Final held-out environment metrics |
| `train.py` | Reproducible training script |
