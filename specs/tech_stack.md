# Tech Stack

## Language

- **Python 3.11+** — primary language across all components

---

## ML Layer

| Concern | Tool |
|---|---|
| Neural network framework | PyTorch |
| Neural architecture search | Optuna |
| Experiment tracking | MLflow or Weights & Biases (TBD) |
| Data manipulation | Pandas, NumPy |
| Preprocessing / encoders | scikit-learn |
| Model serialization | `torch.save` + ONNX export (for portability) |

See `ml/specs.md` for the full ML architecture and NAS search space.

---

## Extrapolation & Database Layer

| Concern | Tool |
|---|---|
| Batch inference | PyTorch (same model, no-grad mode) |
| Database | PostgreSQL (production) / SQLite (development) — TBD |
| ORM / query layer | SQLAlchemy |

The extrapolation database stores: `(location_id, crop_type, input_variables…, predicted_yield)`.

---

## Agent Layer

| Concern | Tool |
|---|---|
| Agent framework | LangChain or LlamaIndex (TBD based on retrieval needs) |
| LLM backbone | Anthropic Claude (via API) or open-source alternative |
| Database tool | Custom SQL tool exposed to the agent |
| API serving | FastAPI |

---

## Infrastructure & Tooling

| Concern | Tool |
|---|---|
| Dependency management | `uv` or `pip` + `pyproject.toml` |
| Testing | `pytest` |
| Linting / formatting | `ruff` |
| Type checking | `mypy` |
| Version control | Git / GitHub |
| Containerization | Docker (for deployment) |

---

## Key Decisions Still Open

- **Database engine:** PostgreSQL vs. SQLite vs. DuckDB — decide after understanding query patterns and deployment target
- **Agent framework:** LangChain vs. LlamaIndex — decide based on whether structured DB retrieval or semantic retrieval dominates
- **Experiment tracker:** MLflow (self-hosted) vs. W&B (cloud) — depends on team preference and privacy requirements
