# Setup Guide

## Required Software

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10 or newer** (developed and tested on 3.12) | Older versions may hit the `LogisticRegression(multi_class=...)` issue described below in reverse (some very old sklearn releases require that argument) — 3.10+ with a recent scikit-learn avoids both directions of the problem. |
| pip | Any recent version | Comes with Python |
| Git | Any | Only needed if you're cloning from a repository |
| ~2 GB free disk | | For Python packages (XGBoost/LightGBM/SHAP pull in compiled binaries) |

No database server, no Docker, no GPU, and no API keys are required. Everything runs
locally.

## Step-by-Step Installation (Fresh Computer)

### 1. Get the code
```bash
# If you have the project as a folder already, just cd into it:
cd traffic-mortality-ml

# If it's in Git:
git clone <your-repo-url>
cd traffic-mortality-ml
```

### 2. Create a virtual environment (strongly recommended)
```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows (Command Prompt)
venv\Scripts\Activate.ps1       # Windows (PowerShell)
```
You should see `(venv)` appear at the start of your terminal prompt.

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This installs: pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, imbalanced-learn,
shap, lime, folium, streamlit, streamlit-folium, joblib, python-dotenv, pytest,
matplotlib, seaborn, jupyter.

On a typical laptop this takes 2–5 minutes (XGBoost/LightGBM/SHAP have compiled
components, so their wheels are a bit larger than a pure-Python package).

### 4. Set up environment variables (optional)
```bash
cp .env.example .env
```
Open `.env` in a text editor if you want to change anything — every value already has
a sensible default in `config.py`, so **this step can be skipped entirely** for a
normal first run. See the Environment Variables section below for what each one does.

### 5. Run the full pipeline
```bash
python main.py
```
On first run, this will:
1. Notice `data/raw/accidents.csv` doesn't exist and generate a 20,000-row synthetic
   dataset (a few seconds).
2. Validate, clean, and feature-engineer it.
3. Train and benchmark all four severity models.
4. Build the risk-scoring lookup table and regressor.
5. Run K-Means + DBSCAN hotspot clustering and rank the black-spots.
6. Save every artifact into `models/`.

Expect this to take **20–40 seconds** total on a normal laptop. You'll see a
step-by-step printed report ending in:
```
======================================================================
DONE
======================================================================
Total pipeline time: ...s
Artifacts in .../models/ are ready for the dashboard: streamlit run dashboard/app.py
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
This opens (or prints a URL for) the dashboard at `http://localhost:8501`. Streamlit
will print something like:
```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```
Open the Local URL in your browser.

### 7. (Optional) Run the test suite
```bash
pytest
```
Expected output: `17 passed` in a couple of seconds.

### 8. (Optional) Explore the EDA notebook
```bash
jupyter notebook notebooks/eda.ipynb
# or, if you use JupyterLab:
jupyter lab notebooks/eda.ipynb
```

## Environment Variables Reference

All optional — every one has a working default in `config.py`. Set these in `.env` (or
directly in your shell) only if you want to change the default behaviour.

| Variable | Default | What it controls |
|---|---|---|
| `DATA_RAW_PATH` | `data/raw/accidents.csv` | Where the raw dataset is read from / generated |
| `DATA_PROCESSED_PATH` | `data/processed/accidents_clean.csv` | Where cleaned data is written |
| `MODEL_DIR` | `models` | Where all trained artifacts are saved |
| `RANDOM_SEED` | `42` | Reproducibility seed for data generation, train/test splits, SMOTE, clustering |
| `N_SYNTHETIC_RECORDS` | `20000` | How many synthetic rows to generate (only used if no raw CSV exists yet) |
| `KMEANS_K` | `12` | Number of K-Means clusters for hotspot detection |
| `DBSCAN_EPS_KM` | `1.5` | DBSCAN neighbourhood radius, in kilometres |
| `DBSCAN_MIN_SAMPLES` | `8` | DBSCAN minimum points to form a dense region |
| `KDE_BANDWIDTH` | `0.02` | Kernel Density Estimation bandwidth, in degrees (~2 km) |
| `TOP_N_HOTSPOTS` | `25` | How many ranked black-spots to keep |
| `STREAMLIT_SERVER_PORT` | `8501` | Which port the dashboard runs on |

None of these are secrets — this project makes no external network calls at runtime.

## "Database Setup"

There is no database to set up. Every persisted artifact is a plain file under
`models/` and `data/` (see `docs/ARCHITECTURE.md` — "Database: flat files,
deliberately"). If you delete the `models/` folder entirely, just re-run
`python main.py` to regenerate everything from scratch.

## Using a Real Dataset Instead of Synthetic Data

1. Get a CSV with these columns (see `src/data/load_data.py`'s `REQUIRED_COLUMNS`):
   `accident_id, date, hour, city, latitude, longitude, road_type, junction_type,
   weather, light_condition, road_surface, speed_limit, num_vehicles_involved,
   severity` (severity must be one of `Minor`, `Serious`, `Fatal`).
2. Save it as `data/raw/accidents.csv` (or set `DATA_RAW_PATH` to point elsewhere).
3. Delete the old synthetic file first if one already exists (`main.py` only generates
   synthetic data when `data/raw/accidents.csv` is **missing**).
4. Run `python main.py` again.

## Common Setup Problems

### `ModuleNotFoundError: No module named 'config'` (or `src`) when running tests
**Cause:** pytest wasn't run from the project root, or `pytest.ini`'s `pythonpath = .`
line is missing/was deleted.
**Fix:** Run `pytest` from the project's top-level folder (the one containing
`config.py`), and confirm `pytest.ini` exists there.

### `pip install` fails on `lightgbm` or `xgboost` with a compiler error
**Cause:** No pre-built wheel is available for your exact OS/Python combination, so pip
tries to compile from source and needs a C++ compiler.
**Fix (Linux):** `sudo apt-get install build-essential cmake` then retry.
**Fix (macOS):** `xcode-select --install` then retry.
**Fix (Windows):** Install "Desktop development with C++" via the Visual Studio Build
Tools, or use a Python version for which prebuilt wheels exist (check
pypi.org/project/lightgbm and pypi.org/project/xgboost for supported versions).

### `TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`
**Cause:** This exact error was hit during this project's own development — newer
scikit-learn versions (1.5+) removed the `multi_class` parameter from
`LogisticRegression` because multinomial handling is now automatic. If you're using an
older scikit-learn that still expects it and see the *opposite* error instead
(`multi_class` required), check your installed version with
`python -c "import sklearn; print(sklearn.__version__)"` and update `requirements.txt`'s
pin accordingly.
**Fix:** This repository's code already omits that parameter (compatible with
scikit-learn 1.4+). If you're modifying `severity_model.py` and reintroduce it,
remove it again.

### Streamlit dashboard shows "No trained artifacts found in `models/`"
**Cause:** `python main.py` hasn't been run yet, or `models/` was deleted/moved.
**Fix:** Run `python main.py` from the project root first, then restart
`streamlit run dashboard/app.py`.

### Dashboard's Hotspot Map tab is blank or errors on `st_folium`
**Cause:** `streamlit-folium` isn't installed (it's a separate package from `folium`).
**Fix:** `pip install streamlit-folium` (already in `requirements.txt` — re-run
`pip install -r requirements.txt` if you installed packages individually before).

### `ValueError: Parameter 'labels' must be ordered` from `roc_auc_score`
**Cause:** This was hit during development too — `roc_auc_score(..., multi_class="ovr")`
requires the `labels` argument to be sorted ascending; the project's severity classes
are ordinal (`Minor, Serious, Fatal`), not alphabetical.
**Fix:** Already handled in `src/utils/metrics.py` (it reorders the probability columns
to match sorted label order internally before calling `roc_auc_score`). If you copy this
metric logic elsewhere, keep that reordering step.

### `SMOTE` raises an error about `k_neighbors` on a very small dataset
**Cause:** SMOTE needs at least `k_neighbors + 1` samples of the minority class in the
training fold. If you drastically shrink `N_SYNTHETIC_RECORDS` (e.g. below a few
hundred), the Fatal class may have too few rows.
**Fix:** Keep `N_SYNTHETIC_RECORDS` at 2,000+ (the default is 20,000), or lower
`k_neighbors` in `src/models/severity_model.py`'s `SMOTE(...)` call.

### Port 8501 already in use
**Cause:** Another Streamlit app (or a previous run) is still using the port.
**Fix:** `streamlit run dashboard/app.py --server.port 8502` (or any free port), or
stop the other process.

### Notebook (`eda.ipynb`) fails with `ModuleNotFoundError: No module named 'seaborn'`
**Cause:** The notebook-only dependencies (`matplotlib`, `seaborn`, `jupyter`) weren't
installed — this happens if you installed packages one-by-one instead of via
`requirements.txt`.
**Fix:** `pip install matplotlib seaborn jupyter` or simply
`pip install -r requirements.txt`.

See `docs/TROUBLESHOOTING.md` for a broader list of runtime (not just setup) issues.
