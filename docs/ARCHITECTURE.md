# Architecture

## System Overview

This is a **batch ML pipeline + read-only dashboard**, not a client-server web app with
a database. There is no request/response API and no persistent database server —
`main.py` runs the entire pipeline once and writes its outputs as files into `models/`;
the Streamlit dashboard then just reads those files. This is a deliberate, appropriate
scope match for an academic ML project: the interesting engineering is in the pipeline
and the models, not in building a multi-user web backend (the companion SRS document
describes what that larger system would look like, as a future extension).

## ASCII Architecture Diagram

```
                              ┌───────────────────────────┐
                              │   Raw accident dataset     │
                              │  data/raw/accidents.csv    │
                              │ (synthetic generator OR a  │
                              │  real MoRTH/Kaggle/STATS19 │
                              │  CSV with the same columns)│
                              └─────────────┬──────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │   INGESTION & VALIDATION   │
                              │   src/data/load_data.py    │
                              │  - schema check             │
                              │  - severity-label check     │
                              │  - GPS bounds check          │
                              │  - profiling (missing %, etc)│
                              └─────────────┬──────────────┘
                                            │ raises DataValidationError on failure
                                            ▼
                              ┌───────────────────────────┐
                              │        PREPROCESSING       │
                              │  src/data/preprocess.py    │
                              │  - drop duplicates          │
                              │  - impute missing values    │
                              │  - coerce dtypes             │
                              └─────────────┬──────────────┘
                                            │
                                            ▼   data/processed/accidents_clean.csv
                              ┌───────────────────────────┐
                              │     FEATURE ENGINEERING    │
                              │ src/features/              │
                              │   feature_engineering.py   │
                              │  - hour_bucket, is_weekend, │
                              │    is_rush_hour, is_night   │
                              │  - weather_severity_index   │
                              │  - shared ColumnTransformer │
                              │    (one-hot + scale)        │
                              └─────────────┬──────────────┘
                                            │
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                        ▼
      ┌─────────────────────┐ ┌─────────────────────┐  ┌──────────────────────┐
      │  PILLAR 1: SEVERITY  │ │  PILLAR 2: RISK      │  │ PILLAR 3: HOTSPOT    │
      │  src/models/         │ │  src/models/          │  │ src/hotspot/         │
      │   severity_model.py  │ │   risk_model.py       │  │  hotspot_detection.py│
      │                      │ │                       │  │                      │
      │ Supervised, per-crash│ │ Aggregated, per        │  │ Unsupervised, spatial │
      │ classification:      │ │ road/time/weather      │  │ only (lat/lon):       │
      │ LR / RF / XGBoost /  │ │ segment: Laplace-      │  │ K-Means + DBSCAN +    │
      │ LightGBM, SMOTE      │ │ smoothed severity-      │  │ KDE, then ranked by   │
      │ inside pipeline,     │ │ weighted score (0-100)  │  │ severity-weighted     │
      │ picked by Fatal      │ │ + GradientBoosting      │  │ score                 │
      │ recall               │ │ Regressor for unseen    │  │                      │
      │                      │ │ combos                  │  │                      │
      └──────────┬───────────┘ └──────────┬─────────────┘  └───────────┬───────────┘
                 │                        │                            │
                 ▼                        ▼                            ▼
      severity_model.joblib      risk_model.joblib +          hotspots_ranked.csv
      shap_background.joblib     risk_lookup.csv
                 │                        │                            │
                 └────────────┬───────────┴─────────────┬──────────────┘
                              ▼                          │
                  ┌───────────────────────┐              │
                  │   TRUST / EXPLAIN-     │              │
                  │   ABILITY LAYER        │              │
                  │  src/explainability/   │              │
                  │    explain.py          │              │
                  │  SHAP (global + local) │              │
                  │  + LIME (local)         │              │
                  └───────────┬────────────┘              │
                              │                            │
                              ▼                            ▼
                  ┌─────────────────────────────────────────────────┐
                  │            STREAMLIT DASHBOARD                    │
                  │             dashboard/app.py                      │
                  │  Tab 1: Severity Prediction (+ SHAP/LIME "why")   │
                  │  Tab 2: Risk Scoring (segment lookup)             │
                  │  Tab 3: Hotspot Map (Folium)                      │
                  │  Tab 4: Model Performance (benchmark table)       │
                  └─────────────────────────────────────────────────┘
                              ▲
                              │  reads models/*.joblib, *.csv, metrics.json
                              │  (loaded once via @st.cache_resource,
                              │   never retrains)
                  ┌─────────────────────────────────────────────────┐
                  │       models/  (persisted pipeline artifacts)     │
                  │  severity_model.joblib · risk_model.joblib ·      │
                  │  risk_lookup.csv · hotspots_ranked.csv ·          │
                  │  shap_background.joblib · metrics.json            │
                  └─────────────────────────────────────────────────┘

     Orchestrated end-to-end by main.py, which runs every box above in order
     and writes every file in the bottom box in a single `python main.py` run.
```

## How the Pieces Interact

### "Frontend": the Streamlit dashboard
`dashboard/app.py` is the only user-facing surface. It is intentionally **read-only
with respect to models** — it never trains anything, it only loads what `main.py`
already produced (via `@st.cache_resource`, so artifacts load once per server process,
not once per page interaction). This keeps the dashboard startup to a couple of seconds
and keeps "training a model" and "showing a model" as two clearly separate concerns,
which also makes each one independently testable.

### "Backend": there isn't a network backend
Because this is a single-machine batch pipeline (see `docs/PROJECT_QA.md` — "why no
Flask/FastAPI backend?" is a very likely viva question, answered there), the "backend"
is just Python modules under `src/` called directly, both by `main.py` (batch) and by
`dashboard/app.py` (interactive lookups against already-trained models). There is no
REST API in this implementation; the SRS's REST/JWT/RBAC layer is scoped as a future
productionisation step, not part of this deliverable.

### "Database": flat files, deliberately
There is no PostgreSQL/SQLite here. Every artifact is a file:
- `data/raw/accidents.csv` / `data/processed/accidents_clean.csv` — the dataset itself
- `models/*.joblib` — trained scikit-learn/imblearn pipelines (via `joblib.dump`)
- `models/*.csv` — the risk lookup table and ranked hotspot table (plain CSVs, so they
  can be opened directly in Excel/Sheets by a non-technical stakeholder if needed)
- `models/metrics.json` — every benchmark number, for the dashboard's Model Performance
  tab and for `docs/CODE_EXPLANATION.md`/report writing

This is the right amount of infrastructure for a project whose job is to demonstrate a
modelling pipeline, not to serve concurrent users — see the SRS for what a real
multi-user deployment would add on top (accounts, RBAC, audit logs, a real database).

### ML models
Three genuinely different model *types* answer three genuinely different questions —
this is the core design decision of the whole project (also covered on slide 8 of the
presentation deck, "Design Rationale"):

| | Severity (Pillar 1) | Risk (Pillar 2) | Hotspot (Pillar 3) |
|---|---|---|---|
| Learning type | Supervised, multi-class classification | Supervised regression over aggregated data | Unsupervised, spatial clustering |
| Grain | One row = one already-occurred accident | One row = one road/time/weather **segment** | One row = one GPS coordinate |
| Answers | "How bad is this crash?" | "How dangerous is this place/time, before anything happens?" | "Where do crashes keep concentrating?" |
| Uses severity label? | Yes (it's the target) | Yes (aggregated into the risk score) | No (clustering is spatial-only; severity is used only afterward, to *rank* the clusters found) |

Forcing these into one model would blur all three questions into one objective function
that answers none of them well — this is documented reasoning, not an afterthought (see
slide 8 in the original deck).

### Explainability layer
`src/explainability/explain.py` sits between Pillar 1 and the dashboard only (severity
is the pillar with an individual per-instance prediction worth explaining; risk and
hotspot outputs are already aggregate/spatial and interpreted directly). It:
- builds a `shap.TreeExplainer` for the tree-based classifiers (RF/XGBoost/LightGBM) or
  a generic `shap.Explainer` around `predict_proba` for Logistic Regression,
- computes **global** feature importance (mean |SHAP value| across a background sample)
  for the dashboard's benchmark story, and
- computes a **local** explanation (SHAP and, independently, LIME) for one dashboard
  prediction at a time.

### External services
None. Everything runs locally/offline once the Python dependencies are installed — no
API keys, no paid services, no network calls at runtime. This is a deliberate scope
choice for a reproducible academic submission (see `.env.example` — every variable in
it is a local path or a tuning knob, never a secret).

## Data Flow at a Glance

See `docs/DATA_FLOW.md` for the full, step-by-step trace with actual column names and
transformations. In short: **raw CSV → validated → cleaned → feature-engineered →
split three ways into the three pillars → each pillar's artifacts saved to `models/` →
dashboard reads those artifacts on demand.**
