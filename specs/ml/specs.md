# ML Model Specs

## Problem Statement

Supervised regression: predict crop yield (kg/ha) from a tabular feature vector describing soil conditions, climate, applied inputs, and crop variety at a given location.

---

## Input Features

### Soil
| Feature | Type | Notes |
|---|---|---|
| `soil_type` | Categorical | e.g., clay, loam, sandy loam, silt |
| `soil_ph` | Continuous | 3.5–9.0 |
| `organic_matter_pct` | Continuous | % by weight |
| `sand_pct` | Continuous | % |
| `silt_pct` | Continuous | % |
| `clay_pct` | Continuous | % |
| `cec` | Continuous | Cation exchange capacity (meq/100g) |

### Nutrients & Fertilizer
| Feature | Type | Notes |
|---|---|---|
| `n_rate_kg_ha` | Continuous | Nitrogen application rate |
| `p_rate_kg_ha` | Continuous | Phosphorus application rate |
| `k_rate_kg_ha` | Continuous | Potassium application rate |
| `soil_n_baseline` | Continuous | Pre-season soil N (ppm) |
| `soil_p_baseline` | Continuous | Pre-season soil P (ppm) |
| `soil_k_baseline` | Continuous | Pre-season soil K (ppm) |

### Climate
| Feature | Type | Notes |
|---|---|---|
| `annual_rainfall_mm` | Continuous | mm |
| `growing_season_rainfall_mm` | Continuous | mm during crop growing period |
| `mean_temp_c` | Continuous | Mean growing season temperature |
| `min_temp_c` | Continuous | Minimum growing season temperature |
| `max_temp_c` | Continuous | Maximum growing season temperature |
| `solar_radiation_mj_m2` | Continuous | Growing season total |
| `frost_days` | Continuous | Count of frost days in season |
| `humidity_pct` | Continuous | Mean relative humidity |

### Crop & Management
| Feature | Type | Notes |
|---|---|---|
| `crop_variety` | Categorical | Variety / cultivar identifier |
| `planting_density` | Continuous | Plants per hectare |
| `planting_doy` | Continuous | Day of year of planting |

### Location
| Feature | Type | Notes |
|---|---|---|
| `latitude` | Continuous | Decimal degrees |
| `longitude` | Continuous | Decimal degrees |
| `elevation_m` | Continuous | Meters above sea level |

---

## Target Variable

| Variable | Type | Unit |
|---|---|---|
| `yield` | Continuous | kg / ha |

---

## Model Architecture

**Base:** Feed-forward neural network (MLP) with variable depth and width, selected by NAS.

**Fixed design decisions:**
- Input: feature vector after preprocessing (standardization for continuous, embedding or one-hot for categorical)
- Output: single scalar (yield), linear activation
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam or AdamW (chosen by NAS)

---

## Neural Architecture Search (Optuna)

### Search Space

| Hyperparameter | Range / Choices |
|---|---|
| `n_layers` | 1 – 6 |
| `hidden_size` | 32, 64, 128, 256, 512 |
| `activation` | ReLU, GELU, SiLU, Tanh |
| `dropout_rate` | 0.0 – 0.5 (step 0.05) |
| `use_batch_norm` | True / False |
| `learning_rate` | 1e-4 – 1e-2 (log scale) |
| `weight_decay` | 1e-6 – 1e-2 (log scale) |
| `batch_size` | 32, 64, 128, 256 |
| `optimizer` | Adam, AdamW |

All layers share the same `hidden_size` and `activation` (uniform architecture). Mixed-size / skip-connection variants are post-MVP.

### NAS Protocol

- **Sampler:** TPE (Tree-structured Parzen Estimator) — Optuna default
- **Pruner:** MedianPruner (prune trials that underperform median at intermediate epochs)
- **Trials:** 150–300 (budget TBD based on compute)
- **Objective:** minimize validation RMSE
- **Early stopping per trial:** no improvement over 10 epochs
- **Data split:** 70% train / 15% val (used during NAS) / 15% test (held out until final evaluation)

### Experiment Tracking

Each Optuna trial logs:
- Full hyperparameter set
- Train and val loss curve
- Best val RMSE achieved

Use MLflow or W&B study callbacks; store study artifact for reproducibility.

---

## Final Model Training

After NAS:
1. Take the best trial's architecture and hyperparameters
2. Retrain on train + val combined (85% of data)
3. Evaluate once on held-out test set — no further tuning after this point

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error (primary) |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| MAPE | Mean Absolute Percentage Error |

Report metrics overall and stratified by crop type.

---

## Preprocessing Pipeline

1. Remove rows with missing target
2. Impute remaining missing values (median for continuous, mode for categorical)
3. One-hot encode low-cardinality categoricals; learned embeddings for high-cardinality (e.g., `crop_variety` if > 20 classes)
4. Standardize all continuous features (zero mean, unit variance, fitted on train set only)
5. Persist fitted transformers alongside the model checkpoint

---

## Outputs / Artifacts

| Artifact | Description |
|---|---|
| `model.pt` | Trained PyTorch model weights |
| `preprocessor.pkl` | Fitted sklearn pipeline |
| `optuna_study.pkl` | Full Optuna study for reproducibility |
| `eval_report.json` | Final test-set metrics |
| `train.py` | Reproducible training script |

---

## Post-Training: Extrapolation

Once the model is finalized:
- Construct the extrapolation feature table: all (location, crop, input_var) combinations across the target area at the desired grid resolution
- Run batch inference (no-grad, model in eval mode)
- Write `(location_id, crop_type, input_variables…, predicted_yield)` rows into the yield database
- The agent layer reads only from this database — the model is not called at inference time
