# Project Overview

## What This Project Does

This is a machine-learning system that takes historical road-accident records and turns
them into three separate, usable outputs instead of one:

1. **Severity Prediction** — given the conditions of a crash that already happened
   (road type, weather, lighting, speed limit, time of day...), predict whether it was
   **Minor, Serious, or Fatal**.
2. **Risk Prediction** — given a road type, junction type, weather condition and time
   window (a combination that *may not have had a crash yet*), produce a **0–100 risk
   score** ranking how dangerous that combination of conditions tends to be.
3. **Hotspot Detection** — using only GPS coordinates (no severity or time information),
   find the physical locations where accidents keep clustering, and rank them into a
   "black-spot" list by how severe the accidents at each location tend to be.

All three read from the same cleaned dataset, but they are trained, evaluated and used
completely independently — see `docs/ARCHITECTURE.md` for why they're split apart
instead of being one model.

A Streamlit dashboard (`dashboard/app.py`) exposes all three to a non-technical user:
type in some conditions and get a severity prediction with a plain-language explanation
of *why* (via SHAP/LIME); look up a risk score for a road/time/weather combination; or
browse a map of the highest-priority accident black-spots.

This implementation follows the project synopsis and presentation deck ("Reducing
Traffic Mortality Using Machine Learning") submitted for the B.Tech CSE (Data Science)
major project at Inderprastha Engineering College. It builds the core three-pillar ML
framework described there. The companion SRS document describes a larger, productionised
system (user accounts, role-based access, a full web front end, audit logging, a
relational database) — that is the target for a future, fully deployed version; this
repository is the ML pipeline and dashboard at the center of it, which is what a college
major-project submission is realistically expected to deliver and demonstrate.

## Problem Statement

Road-safety practice today is **reactive**: accidents are investigated after the fact,
and national statistics are compiled and published once a year. That tells you the
trend was bad — it does not tell a traffic officer which junction, at which hour, under
which weather, needs attention *tonight*.

Specifically:

- **1.19 million** people die in road crashes every year worldwide, and it is the
  leading cause of death for ages 5–29 (WHO, 2023).
- India recorded **172,890 deaths across 480,583 accidents** in 2023 — the highest ever
  — roughly 20 deaths every hour, with over-speeding the leading contributing factor
  (MoRTH, 2023).
- Less than 5% of road length (national/state highways) accounts for **more than half**
  of all fatalities.

Published academic work on this problem has three recurring gaps, which this project
was scoped to close simultaneously rather than pick one:

1. Severity prediction and hotspot detection are almost always studied **in isolation**,
   as two separate papers, never one connected pipeline.
2. Most published models are evaluated on **overall accuracy** — a metric that looks
   good while quietly missing the rare, high-stakes Fatal class (which is ~2–3% of any
   real accident dataset).
3. Very few studies go further than a classification result to something a
   **non-technical stakeholder** (a traffic police officer, a highway engineer) could
   actually act on.

## Project Objectives

1. Collect, clean and preprocess accident data, handling missing values, encoding
   categorical attributes, and correcting for severe class imbalance between
   fatal and non-fatal records.
2. Benchmark multiple ML classifiers (Logistic Regression, Random Forest, XGBoost,
   LightGBM) for severity prediction, optimizing for **recall on the Fatal class**
   rather than overall accuracy.
3. Score road/time/weather combinations with a continuous, exposure-based **risk
   score**, independent of any single completed accident.
4. Identify geographic accident hotspots ("black spots") using spatial clustering
   (K-Means, DBSCAN) and Kernel Density Estimation.
5. Make every prediction explainable (SHAP + LIME) so the factors driving a
   prediction are transparent to a non-technical reviewer.
6. Expose all three outputs through a single interactive dashboard.

## Target Users

| User | What they need from this system |
|---|---|
| Traffic police / highway patrol | "Where and when should I put a patrol tonight?" → Risk Scoring tab |
| Highway engineers / road-safety planners | "Which junctions need signage, lighting or redesign first?" → Hotspot Map tab |
| Policy makers / transport authorities | Evidence-based, ranked, explainable input for budget prioritization |
| Project evaluators / examiners (this being an academic submission) | A working, reproducible, well-documented pipeline they can run themselves |

## Main Features

- **Synthetic-but-realistic data generator** (`src/data/generate_sample_data.py`) that
  stands in for the real MoRTH/Kaggle/STATS19 datasets named in the synopsis (those
  require a manual download this environment doesn't have access to). It reproduces the
  same structure and the same real-world risk relationships (night + fog + highway +
  speed → much higher fatality odds), including a handful of deliberately dangerous,
  tightly-clustered "black-spot" junctions per city so hotspot detection has something
  genuine to find. **Swap this file's CSV output for a real dataset with the same
  column names and every downstream step keeps working unchanged.**
- **Data validation and cleaning** with explicit, raised errors for malformed input
  rather than silent corruption.
- **Four benchmarked severity classifiers** trained with SMOTE inside a leakage-safe
  pipeline, evaluated with a recall-first metric set, and a cost-sensitive decision
  threshold for deployment.
- **Segment-level risk scoring** with Laplace/credibility smoothing (so a segment with
  2 recorded accidents doesn't get an extreme score just because both were severe) and
  a regressor that generalizes to combinations not well-represented in the historical
  data.
- **Three complementary spatial methods** (K-Means, DBSCAN, KDE) plus a
  severity-weighted ranking, because each method answers a slightly different spatial
  question (see `docs/ARCHITECTURE.md`).
- **Dual explainability** (SHAP *and* LIME) so a single local explanation can be
  cross-checked by an independent method.
- **A four-tab Streamlit dashboard** requiring no code, notebook or model file access
  from the end user.
- **A reproducible one-command pipeline** (`python main.py`) that regenerates every
  artifact the dashboard needs from raw data in under 30 seconds.
- **17 passing unit tests** covering cleaning, feature engineering, severity/risk/hotspot
  correctness properties (not just "does it run").

## Technologies Used, and Why

| Technology | Used for | Why this one |
|---|---|---|
| **Python 3.10+** | Everything | The de-facto language for this stack (pandas/sklearn/XGBoost/SHAP all Python-first); matches the synopsis's own Tools & Technologies slide. |
| **pandas / NumPy** | Data loading, cleaning, feature engineering | Standard tabular data toolkit; every other library in this stack is built to interoperate with pandas DataFrames. |
| **scikit-learn** | Preprocessing (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), Logistic Regression, Random Forest, `GradientBoostingRegressor`, `KMeans`, `DBSCAN`, `KernelDensity`, evaluation metrics | One consistent, well-documented API for classical ML, clustering *and* density estimation — avoids pulling in three different libraries for those three jobs. |
| **XGBoost / LightGBM** | Two of the four severity classifiers | Named explicitly in the synopsis as the ensemble methods expected to capture non-linear interactions between speed/light/weather/road type; both are industry-standard gradient boosting implementations with native multi-class support. |
| **imbalanced-learn (SMOTE)** | Oversampling the Fatal class during training only | Fatal accidents are ~2–3% of the data; SMOTE is the specific technique named in the synopsis, and `imblearn.pipeline.Pipeline` lets it sit safely between the preprocessor and the classifier so it never touches validation/test data. |
| **SHAP** | Global + local explainability | Model-agnostic-enough to explain both a linear model and tree ensembles with the same downstream code; the de-facto standard for "why did the model say this" in tabular ML. |
| **LIME** | A second, independent local explainer | Named alongside SHAP in the synopsis; perturbs the input directly rather than relying on the model's internal structure, so it's a genuinely different cross-check, not just a second SHAP call. |
| **Folium** | The hotspot map | Renders an interactive Leaflet.js map from Python with no front-end code required — fits a dashboard meant to need no separate web development. |
| **Streamlit** + **streamlit-folium** | The dashboard itself | Turns the three pillars into a clickable UI in a few hundred lines of pure Python — no separate frontend/backend split needed for a project at this scope, matching the synopsis's own choice of Streamlit for deployment. |
| **joblib** | Saving/loading trained pipelines and models | The standard way to persist scikit-learn-compatible objects (including SMOTE/ColumnTransformer pipelines) without re-training on every dashboard restart. |
| **python-dotenv** | Reading `.env` for configurable paths/seeds | Lets `config.py` be overridden per-machine without editing code. |
| **pytest** | The test suite | Standard, minimal-boilerplate Python testing framework. |
| **matplotlib / seaborn / Jupyter** | `notebooks/eda.ipynb` only | For the exploratory, look-at-the-data step described in the SRS section 2.2 — kept separate from the reproducible pipeline in `main.py` on purpose (see `docs/PROJECT_STRUCTURE.md`). |

See `docs/ARCHITECTURE.md` for how these pieces fit together, and
`docs/PROJECT_QA.md` for the honest limitations of the synthetic data and the
class-imbalance results (worth reading before a viva — it turns a potentially awkward
question into a rehearsed, confident answer).
