# Project Structure

```
traffic-mortality-ml/
├── config.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── data/
│   ├── raw/accidents.csv          (generated, not committed)
│   ├── processed/accidents_clean.csv  (generated, not committed)
│   └── raw/.gitkeep, processed/.gitkeep
├── src/
│   ├── data/
│   │   ├── generate_sample_data.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── severity_model.py
│   │   └── risk_model.py
│   ├── hotspot/
│   │   └── hotspot_detection.py
│   ├── explainability/
│   │   └── explain.py
│   └── utils/
│       └── metrics.py
├── dashboard/
│   └── app.py
├── notebooks/
│   └── eda.ipynb
├── models/                          (generated artifacts, not committed)
├── tests/
│   ├── test_preprocess.py
│   └── test_models.py
└── docs/
    └── (this documentation set)
```

## Root-level files

### `config.py`
**What it's for:** Every path, random seed, model hyperparameter and tuning constant
used anywhere in the project lives here (e.g. `RANDOM_SEED`, `KMEANS_K`,
`FATAL_DECISION_THRESHOLD`, `CITY_CENTERS`, `RISK_SEVERITY_WEIGHTS`). Every other
module imports from it instead of hard-coding values.
**Safe to modify:** Yes — this is the intended place to tune the project. Changing
`N_SYNTHETIC_RECORDS`, `KMEANS_K`, `DBSCAN_EPS_KM`, etc. and re-running `python main.py`
is the normal way to experiment.
**Careful with:** `RANDOM_SEED` (changing it changes every reported number in the docs
and dashboard — fine for experimentation, just know the exact metrics will shift)
and `SEVERITY_CLASSES` (the order `["Minor","Serious","Fatal"]` is assumed — index
2 = Fatal — throughout `severity_model.py`; reordering it without updating that
assumption will break the Fatal-recall logic).

### `main.py`
**What it's for:** The single entry point that runs the entire pipeline in order:
generate data (if missing) → validate → clean → engineer features → train and
benchmark all four severity models → train the risk regressor → run hotspot
clustering → save every artifact to `models/`.
**Safe to modify:** Yes, if you want to change *what gets printed/saved*, add a new
step, or change the order. It's meant to be read top-to-bottom as the project's table
of contents.
**Don't:** Delete the artifact-saving lines (the `joblib.dump(...)` / `.to_csv(...)`
calls) — the dashboard depends on every one of those files existing with those exact
names (see `config.py`'s `*_PATH` constants).

### `requirements.txt`
Pinned-loosely (`>=`) dependency list. Safe to bump versions; if you do, re-run
`pytest` and `python main.py` once to confirm nothing broke (a couple of the sklearn
APIs used here — e.g. `LogisticRegression`'s old `multi_class` parameter — have
actually been removed in recent scikit-learn versions during this project's own
development, so version drift is a real, not theoretical, risk here).

### `.env.example` / `.env`
Template for optional local overrides (paths, seed, clustering knobs). Copy to `.env`;
`config.py` loads it automatically via `python-dotenv`. **Never contains secrets** —
this project makes no paid/authenticated external calls.

### `pytest.ini`
Tells `pytest` to add the project root to `sys.path` (so `import config` and
`import src...` work from inside `tests/`) and where the tests live. Safe to leave
alone; only edit it if you restructure the `tests/` folder.

## `data/`
Two subfolders, both `.gitignore`'d except for `.gitkeep` placeholders — this is
generated data, not source code, so it isn't meant to be committed to version control.
- `data/raw/accidents.csv` — output of `src/data/generate_sample_data.py`, or a real
  dataset you drop in with the same column names (see `src/data/load_data.py`'s
  `REQUIRED_COLUMNS`).
- `data/processed/accidents_clean.csv` — output of `src/data/preprocess.py`; this is
  what `notebooks/eda.ipynb` reads.

## `src/` — all pipeline logic, one responsibility per file

### `src/data/generate_sample_data.py`
Synthetic data generator standing in for a real accident dataset (see
`docs/PROJECT_OVERVIEW.md` for why). **Safe to modify** — this is the first file to
change if you get access to a real MoRTH/Kaggle/STATS19 dataset: either replace its
output CSV directly, or rewrite `generate()` to load and reshape the real file into the
same columns.

### `src/data/load_data.py`
Schema validation (`validate_schema`, raises `DataValidationError`) and a lightweight
data profile (`profile`). **Modify with care**: `REQUIRED_COLUMNS` and `VALID_SEVERITY`
are the contract every other module assumes holds. If you add a new raw column, add it
here first.

### `src/data/preprocess.py`
Cleaning only — duplicates, missing values, dtypes. Deliberately does **not** create
new columns (that's `feature_engineering.py`'s job — see the module docstring for why
they're kept separate). Safe to modify the imputation strategy (e.g. swap mode/median
for a smarter method) without touching anything else.

### `src/features/feature_engineering.py`
Derives `hour_bucket`, `is_weekend`, `is_rush_hour`, `is_night`,
`weather_severity_index`, and defines `FEATURE_COLUMNS` / `CATEGORICAL_FEATURES` /
`NUMERIC_FEATURES` — the single source of truth for which columns every model actually
trains on. **Central file** — `severity_model.py`, `risk_model.py`'s regressor step, and
`explain.py` all import `FEATURE_COLUMNS`/`build_preprocessor()` from here rather than
redefining it. If you add a new feature, add it here and it becomes available to every
model automatically (as long as you also add it to `CATEGORICAL_FEATURES` or
`NUMERIC_FEATURES`).

### `src/models/severity_model.py`
Pillar 1. Defines the four candidate pipelines, `train_and_evaluate`,
`cross_validate`, `select_best_model`, the `apply_fatal_threshold` cost-sensitive
decision rule, and `predict_single` (used by the dashboard). See
`docs/CODE_EXPLANATION.md` for a walkthrough of the non-obvious parts (integer label
encoding, the threshold rule).

### `src/models/risk_model.py`
Pillar 2. `build_risk_lookup` (aggregation + smoothing), `train_risk_regressor`,
`score_segment` (used by the dashboard). `SEGMENT_KEYS` here defines the grain of a
"segment" — change it (e.g. add `city`) to make the risk score more granular, at the
cost of more segments having fewer observations each (the smoothing in
`RISK_SMOOTHING_K` exists to protect against exactly that trade-off).

### `src/hotspot/hotspot_detection.py`
Pillar 3. `run_kmeans`, `run_dbscan`, `fit_kde`/`kde_grid`, `rank_hotspots`. No target
label is used to *form* clusters — only afterward, in `rank_hotspots`, to score them.

### `src/explainability/explain.py`
`build_shap_explainer`, `global_feature_importance`, `local_explanation_shap`,
`build_lime_explainer`, `local_explanation_lime`. The LIME wrapper is the most
intricate file in the project (label-encodes categoricals for LIME's internals, then
decodes back to strings inside `predict_fn` before calling the real pipeline) — see
`docs/CODE_EXPLANATION.md`.

### `src/utils/metrics.py`
`evaluate_classifier` (the shared recall-first metric computation) and
`rank_models_by_fatal_recall`. If you want to change *how* the best model is picked
(e.g. weight precision more), this is the one place to change it.

## `dashboard/app.py`
The Streamlit UI. **Reads models/ only — never trains.** Safe to modify freely for
layout/UX changes; if you change what a pillar's module returns, update the
corresponding tab here to match.

## `notebooks/eda.ipynb`
Exploratory analysis only (severity distribution, hourly patterns, weather x light
fatal-rate heatmap, speed vs. severity, a spatial scatter). Deliberately **not** part
of the reproducible pipeline — notebooks are for one-off, human-driven exploration;
anything that needs to run the same way every time belongs in `src/` and `main.py`
instead. Safe to edit/extend freely; re-run cells top-to-bottom after editing
`src/data/preprocess.py` or `feature_engineering.py`.

## `models/` (generated, gitignored)
Every file here is an output of `python main.py` — safe to delete the whole folder at
any time; running `python main.py` again regenerates all of it (~30 seconds). Never
hand-edit these files.

## `tests/`
`test_preprocess.py` (cleaning + feature engineering correctness) and
`test_models.py` (severity/risk/hotspot correctness properties — bounded scores,
sorted rankings, threshold logic). Uses a small, fast synthetic sample
(`generate(n_records=1500)`), not the full 20,000-row dataset, so the suite runs in a
couple of seconds. **Safe and encouraged to extend** — add a test any time you add a
function with a non-trivial return value.

## `docs/`
This documentation set. Safe to edit for accuracy as the code evolves — these files
describe the *actual* code in this repository, not an aspirational version of it, so
if you change behaviour, update the matching doc in the same commit.

## Summary: what's safe vs. what to be careful with

| Safe to change freely | Change with care | Don't hand-edit |
|---|---|---|
| `config.py` tuning constants | `config.SEVERITY_CLASSES` order, `RANDOM_SEED` | Anything in `models/` |
| `dashboard/app.py` layout | `src/data/load_data.py`'s `REQUIRED_COLUMNS` | `data/processed/*.csv` |
| `notebooks/eda.ipynb` | `src/features/feature_engineering.py`'s `FEATURE_COLUMNS` (many files import it) | |
| Adding new tests | `requirements.txt` versions (re-test after bumping) | |
| Model hyperparameters in `severity_model.py`/`risk_model.py` | | |
