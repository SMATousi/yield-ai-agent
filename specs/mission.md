# Mission

## Purpose

Yield AI Agent helps farmers and agronomists maximize crop yield by providing data-driven cultivation recommendations. Given a target crop and location, the system identifies the combination of soil preparation, fertilization, and climate-aligned practices most likely to achieve the highest yield — removing guesswork from agronomic decision-making.

## Core Concept

```
Tabular training data
        │
        ▼
  ML Yield Model  ──────────────────────────────────┐
        │                                           │
        ▼                                           │
 Extrapolation over                          Inputs: soil, climate,
 full target area                            fertilizer, crop, location
        │
        ▼
  Yield Database
  (area-wide predictions)
        │
        ▼
   AI Agent UI
        │
        ▼
  User query: crop type + location
        │
        ▼
  Sort database by max predicted yield
        │
        ▼
  Recommend optimal input variables
```

## Who It Is For

- Farmers choosing inputs before a planting season
- Agronomists advising on field management at scale
- Agricultural researchers exploring yield potential across regions

## What Success Looks Like

A user can describe their crop and location in natural language and receive a ranked, actionable set of input recommendations (soil amendments, fertilizer rates, variety selection, etc.) grounded in model predictions across the full cultivation area — not just point estimates.

## Out of Scope (v1)

- Real-time sensor integration or IoT data ingestion
- Economic optimization (cost of inputs vs. revenue)
- Pest, disease, or irrigation scheduling recommendations
- Model retraining from user feedback
