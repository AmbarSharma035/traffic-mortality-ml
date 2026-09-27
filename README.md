# Reducing Traffic Mortality Using Machine Learning

**One data pipeline, three coordinated outcomes — Severity Prediction, Risk
Prediction, and Hotspot Detection — built to move road safety from hindsight to
foresight.**

B.Tech CSE (Data Science) major project — Inderprastha Engineering College.
Ambar Sharma · Jivesh Dhama · Aditya Tomar · Siddharth Soni.

> Full documentation set lives in [`docs/`](docs/) — this README is the quick-start.
> See `docs/PROJECT_QA.md` for interview/viva prep, `docs/CODE_EXPLANATION.md` for a
> walkthrough of the trickier code, and `docs/ARCHITECTURE.md` for the full system
> design.

## Overview

Road-safety practice today is reactive: accidents get investigated after the fact, and
national statistics are published once a year — too slow and too coarse to guide
day-to-day intervention. This project turns historical accident data into three
proactive, usable outputs instead:

1. **Severity Prediction** — given a crash's conditions (road, weather, light, speed,
   time), predict whether it's Minor / Serious / Fatal.
2. **Risk Prediction** — score a road/time/weather combination 0–100 for how dangerous
   it tends to be, *before* an accident happens.
3. **Hotspot Detection** — find and rank the physical locations where accidents keep
   concentrating.

All three are exposed through a single Streamlit dashboard, with SHAP/LIME
explanations for every severity prediction so a non-technical stakeholder can see
*why*, not just *what*.

**No real accident dataset was accessible in this development environment**
(MoRTH/Kaggle/STATS19 all require a manual download), so a synthetic-but-structurally
-realistic data generator stands in for one — see `docs/PROJECT_OVERVIEW.md` for full
disclosure of this and every other honest limitation. Drop in a real CSV with the same
columns and the entire pipeline works unchanged.

## Features

- Synthetic accident data generator with realistic class imbalance (~2–3% Fatal) and
  deliberately injected geographic "black-spot" junctions
- Schema-validated ingestion that fails loudly on malformed data, never silently
- Four benchmarked severity classifiers (Logistic Regression, Random Forest, XGBoost,
  LightGBM) with SMOTE inside a leakage-safe pipeline, picked by Fatal-class recall
- A cost-sensitive decision-threshold rule for deployment, alongside the raw model output
- Segment-level, Laplace-smoothed risk scoring with a regressor that generalizes to
  unseen road/time/weather combinations
- K-Means + DBSCAN + KDE hotspot detection with a severity-weighted ranked black-spot
  table
- Dual explainability (SHAP *and* LIME) for cross-checked, human-readable predictions
- A four-tab interactive Streamlit dashboard
- A one-command, ~30-second reproducible pipeline (`python main.py`)
- 17 passing unit tests

## Tech Stack

Python 3.10+ · pandas · NumPy · scikit-learn · XGBoost · LightGBM ·
imbalanced-learn (SMOTE) · SHAP · LIME · Folium · Streamlit · streamlit-folium ·
joblib · python-dotenv · pytest · matplotlib/seaborn/Jupyter (EDA notebook only)

Full rationale for every choice: `docs/PROJECT_OVERVIEW.md`.

## Architecture Summary

Batch ML pipeline + read-only dashboard — no network API, no database server. Every
artifact is a plain file:

```
raw CSV -> validate -> clean -> feature-engineer
   |            |                |
   +-> Pillar 1: Severity (4 classifiers, SMOTE, SHAP/LIME)
   +-> Pillar 2: Risk (segment aggregation + regressor)
   +-> Pillar 3: Hotspot (K-Means + DBSCAN + KDE + ranking)
                    |
                    v
        models/*.joblib, *.csv, metrics.json
                    |
                    v
        dashboard/app.py  (reads only, never trains)
```

Full ASCII diagram and per-component breakdown: `docs/ARCHITECTURE.md`.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Requires **Python 3.10+**. Full step-by-step guide (including a real-dataset swap-in
and a troubleshooting list): `docs/SETUP_GUIDE.md`.

## How to Run

```bash
# 1. Run the full pipeline (generates data if none exists, trains everything, ~30s)
python main.py

# 2. Launch the dashboard
streamlit run dashboard/app.py
# -> open http://localhost:8501

# 3. (optional) Run the test suite
pytest

# 4. (optional) Explore the EDA notebook
jupyter notebook notebooks/eda.ipynb
```

## Environment Variables

All optional — copy `.env.example` to `.env` to override any default in `config.py`.
None are secrets; this project makes no external network calls at runtime.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_RAW_PATH` | `data/raw/accidents.csv` | Raw dataset location |
| `MODEL_DIR` | `models` | Where trained artifacts are saved |
| `RANDOM_SEED` | `42` | Reproducibility |
| `N_SYNTHETIC_RECORDS` | `20000` | Synthetic dataset size |
| `KMEANS_K` | `12` | K-Means cluster count |
| `DBSCAN_EPS_KM` / `DBSCAN_MIN_SAMPLES` | `1.5` / `8` | DBSCAN tuning |
| `TOP_N_HOTSPOTS` | `25` | Ranked black-spots kept |
| `STREAMLIT_SERVER_PORT` | `8501` | Dashboard port |

Full reference: `docs/SETUP_GUIDE.md`.

## Screenshots

> _Dashboard screenshots go here — run `streamlit run dashboard/app.py` locally and
> capture the four tabs (Severity Prediction, Risk Scoring, Hotspot Map, Model
> Performance) to fill this section in for a submission/report._

```
[ Severity Prediction tab ]     [ Risk Scoring tab ]
        (placeholder)                 (placeholder)

[ Hotspot Map tab ]              [ Model Performance tab ]
        (placeholder)                 (placeholder)
```

## API Summary

There is **no REST/HTTP API** in this implementation — everything runs as direct Python
function calls within a single process (`main.py` for training, `dashboard/app.py` for
serving predictions interactively). The project's companion SRS document specifies a
REST API with JWT auth for a future, fully productionised version; that's out of scope
for this ML-pipeline deliverable. See `docs/PROJECT_QA.md` → "Explain the APIs."

The closest thing to a stable "interface" is the set of importable functions other
code is meant to call:
- `src.models.severity_model.predict_single(pipeline, row_df)` → severity prediction
- `src.models.risk_model.score_segment(model, road_type, junction_type, weather, hour_bucket, is_weekend)` → risk score
- `src.hotspot.hotspot_detection.rank_hotspots(df, cluster_labels)` → ranked black-spots
- `src.explainability.explain.local_explanation_shap(...)` /
  `local_explanation_lime(...)` → per-prediction explanations

## Database Summary

There is **no database server** — every persisted artifact is a plain file:

| File | Format | Contents |
|---|---|---|
| `data/raw/accidents.csv` | CSV | Raw (synthetic or real) accident records |
| `data/processed/accidents_clean.csv` | CSV | Cleaned, validated dataset |
| `models/severity_model.joblib` | joblib | Best severity classifier pipeline + its name |
| `models/risk_model.joblib` | joblib | Fitted risk-scoring regressor |
| `models/risk_lookup.csv` | CSV | Aggregated risk score per road/time/weather segment |
| `models/hotspots_ranked.csv` | CSV | Ranked black-spot clusters |
| `models/shap_background.joblib` | joblib | Background sample for SHAP explainer construction |
| `models/metrics.json` | JSON | Full benchmark metrics for all 4 severity models + clustering quality |

Full explanation: `docs/ARCHITECTURE.md` → "Database: flat files, deliberately".

## Deployment Instructions

This ships as a local pipeline + local dashboard. To deploy the dashboard (e.g. to
Streamlit Community Cloud or your own VM):

1. Ensure `models/` is populated — either run `python main.py` as a build/startup step
   on the target, or intentionally commit a pre-trained `models/` folder for a demo
   deployment (it's `.gitignore`'d by default).
2. Set any needed environment variables (see above) through the platform's own
   environment/secrets configuration — most platforms don't read a committed `.env`
   file automatically.
3. Point the platform at `dashboard/app.py` as the entry point
   (`streamlit run dashboard/app.py`).
4. Confirm the target has enough memory for XGBoost/LightGBM/SHAP's dependencies plus
   the full dataset and every model artifact in memory (~1 GB free is a safe minimum
   at the default `N_SYNTHETIC_RECORDS=20000`).

Full deployment troubleshooting: `docs/TROUBLESHOOTING.md` → "Deployment
Considerations".

## Documentation Index

| File | Covers |
|---|---|
| `docs/PROJECT_OVERVIEW.md` | What/why, problem statement, objectives, tech choices |
| `docs/ARCHITECTURE.md` | Full system design + ASCII diagram |
| `docs/PROJECT_STRUCTURE.md` | Every folder/file, what's safe to edit |
| `docs/SETUP_GUIDE.md` | Full install walkthrough + common setup problems |
| `docs/DATA_FLOW.md` | Step-by-step data trace with real column/function names |
| `docs/CODE_EXPLANATION.md` | The non-obvious code, explained for a viva/interview |
| `docs/PROJECT_QA.md` | Core Q&A + 50 graded practice questions |
| `docs/TROUBLESHOOTING.md` | Dependency/env/data/model/dashboard/deployment issues |
