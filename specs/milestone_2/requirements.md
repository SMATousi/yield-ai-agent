# Milestone 2 — Requirements

## Goal

A trained, evaluated yield prediction model selected via neural architecture search, built on the actual trial dataset.

---

## Problem Statement

Supervised regression: predict crop yield (kg/ha) from a 50-predictor feature vector describing planting conditions, crop variety, and environment (soil + weather) at a given location/environment.

---

## Dataset

Two sub-datasets are combined into a single training corpus:

| Sub-dataset | Observations | Environments |
|---|---|---|
| Planting Date Trial | 1,800 | 22 |
| Varieties Test | 6,700 | 586 |
| **Combined** | **~8,500** | **608** |

Each observation belongs to one environment. Environments are distinct field sites and/or years. Cross-environment generalization is the primary deployment concern.

---

## Input Features (~50 Predictors)

Features fall into four groups as defined in the source data:

### Planting Date
Continuous and/or cyclical features encoding when the crop was planted (e.g., day of year, week of year).

### Maturity Group
Categorical feature indicating the crop's maturity classification (e.g., soybean maturity groups MG 0–10). Encoded as an embedding or one-hot depending on cardinality.

### Pop / RS (Population & Row Spacing)
Continuous management inputs:
- Planting density (plants per hectare or seeds/acre)
- Row spacing (cm or inches)

### Soil & Weather
The bulk of the 50 predictors — continuous features covering:
- Soil properties (texture, pH, organic matter, nutrient baselines, CEC)
- In-season weather (rainfall, temperature, solar radiation, humidity, frost days)
- Growing season summaries aggregated per environment

> Exact column names are derived from the source data files in Milestone 1. This spec uses the group structure from the dataset overview.

---

## Target Variable

| Variable | Type | Unit |
|---|---|---|
| `yield` | Continuous | kg / ha (or bu/acre — normalize at preprocessing) |

---

## Model Architecture

**Base:** Feed-forward MLP with variable depth and width, selected by NAS (Optuna).

**Fixed design decisions:**
- Input: preprocessed feature vector (see below)
- Output: single scalar (yield), linear activation
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam or AdamW (chosen by NAS)
- Regularization is critical given ~8,500 observations — dropout and weight decay are primary levers

---

## Preprocessing Pipeline

1. Merge Planting Date Trial and Varieties Test; add a `dataset_source` indicator feature
2. Remove rows with missing target
3. Impute remaining missing values (median for continuous, mode for categorical)
4. One-hot encode low-cardinality categoricals; learned embeddings for Maturity Group if > 10 classes
5. Standardize all continuous features (zero mean, unit variance, fitted on train environments only)
6. Persist fitted transformers alongside the model checkpoint

---

## Outputs / Artifacts

| Artifact | Description |
|---|---|
| `model.pt` | Trained PyTorch model weights |
| `preprocessor.pkl` | Fitted sklearn pipeline |
| `optuna_study.pkl` | Full Optuna study for reproducibility |
| `eval_report.json` | Final held-out environment metrics |
| `train.py` | Reproducible training script |
