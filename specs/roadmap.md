# Roadmap

Milestones are sequential. Estimated durations assume one focused developer.

---

## Milestone 1 — Data Foundation (~2 weeks)

**Goal:** Have clean, model-ready tabular data and a clear feature schema.

- Identify and acquire training datasets (public agronomic databases, field trial records)
- Define the canonical feature schema (see `ml/specs.md` for input/output spec)
- Build data cleaning, validation, and preprocessing pipeline
- Document data provenance and known gaps

**Done when:** A single `.csv` / dataset can be loaded and split into train/val/test with no manual intervention.

---

## Milestone 2 — ML Model with NAS (~4 weeks)

**Goal:** A trained, evaluated yield prediction model selected via neural architecture search.

- Implement baseline feed-forward NN in PyTorch
- Define Optuna search space (architecture + hyperparameters)
- Run NAS trials; log experiments (MLflow or W&B)
- Select best architecture; retrain on full train+val set
- Evaluate on held-out test set; document metrics

**Done when:** Best model is serialized to disk with reproducible training script and evaluation report.

---

## Milestone 3 — Extrapolation Pipeline (~2 weeks)

**Goal:** Run the trained model across the entire target geographic area to populate a yield database.

- Define the extrapolation grid (geographic resolution, crop × location combinations)
- Assemble the "Extrapolation data" feature table covering all grid cells
- Run inference; store `(location, crop, input_vars, predicted_yield)` records
- Validate coverage and spot-check predictions for plausibility

**Done when:** A queryable database exists with predictions for all target area × crop combinations.

---

## Milestone 4 — Agent MVP (~3 weeks)

**Goal:** A working AI agent that takes user queries and returns yield-maximizing recommendations.

- Set up LangChain or LlamaIndex agent with database query tool
- Implement query parsing: extract crop type and location from natural language
- Implement ranking: sort database records by predicted yield for the matched crop × location
- Format and return top-N input variable recommendations
- Build a minimal CLI or API interface for testing

**Done when:** An end-to-end query ("I want to grow wheat near Shiraz") returns a ranked recommendation list.

---

## Milestone 5 — Integration, Evaluation & Polish (~2 weeks)

**Goal:** Hardened, documented system ready for early users.

- End-to-end integration tests
- Latency profiling and query optimization on the database
- Improve agent response quality (prompt tuning, edge cases)
- Write user-facing documentation and setup guide
- Internal demo / stakeholder review

**Done when:** System handles a diverse test query set reliably and a non-developer can run it from the README.

---

## Future (Post-MVP)

- Web or mobile front-end
- Economic layer: cost of inputs vs. projected yield revenue
- Irrigation and scheduling recommendations
- Model retraining pipeline as new field data accumulates
- Multi-language support
