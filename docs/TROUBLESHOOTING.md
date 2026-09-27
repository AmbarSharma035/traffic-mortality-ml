# Troubleshooting

Organized by category. `docs/SETUP_GUIDE.md` also has a shorter "Common Setup Problems"
section focused specifically on first-time installation — this file is the broader
reference, including issues that can show up after setup, during normal use.

## Dependency Issues

### `pip install -r requirements.txt` fails on `lightgbm` or `xgboost`
**Symptom:** A compiler error (`error: command 'gcc' failed`, `CMake Error`, etc.)
**Cause:** No prebuilt wheel available for your OS/architecture/Python version
combination, so pip tries to compile from source.
**Fix:**
- Linux: `sudo apt-get install build-essential cmake libgomp1`, then retry.
- macOS: `xcode-select --install`, then retry. If it still fails on Apple Silicon,
  try `pip install lightgbm --no-binary lightgbm` or check for an ARM-specific wheel.
- Windows: Install the "Desktop development with C++" workload via Visual Studio
  Build Tools, then retry.

### `ImportError: cannot import name 'X' from 'sklearn...'`
**Cause:** scikit-learn API drift between versions. This exact project hit this during
development: `LogisticRegression(multi_class="multinomial")` raised
`TypeError: unexpected keyword argument 'multi_class'` on scikit-learn 1.8 (the
parameter was removed; multinomial handling is automatic now).
**Fix:** Check your installed version — `python -c "import sklearn; print(sklearn.__version__)"`
— and either upgrade to match `requirements.txt`'s expectations (`scikit-learn>=1.4`)
or adjust the offending call in `src/models/severity_model.py` to match your installed
version's API.

### `ModuleNotFoundError: No module named 'streamlit_folium'`
**Cause:** `streamlit-folium` is a separate PyPI package from `folium` — installing one
does not install the other.
**Fix:** `pip install streamlit-folium` or re-run `pip install -r requirements.txt`
(it's listed there).

### `ModuleNotFoundError: No module named 'lime'` (or `shap`)
**Cause:** Packages installed individually/manually instead of via
`requirements.txt`, and one was missed.
**Fix:** `pip install -r requirements.txt` from the project root.

### Notebook fails with `ModuleNotFoundError: No module named 'seaborn'` / `'matplotlib'` / `jupyter` not found
**Cause:** The notebook-only dependencies are separate from the core pipeline
dependencies and are easy to miss if installing packages one at a time.
**Fix:** `pip install matplotlib seaborn jupyter`, or `pip install -r requirements.txt`.

## Environment Variable Issues

### Changing `.env` doesn't seem to have any effect
**Cause:** `config.py` reads environment variables **once, at import time**
(`os.getenv(...)` calls at module load). If a Python process (e.g. a long-running
Streamlit server, or a Jupyter kernel) was already started before you edited `.env`,
it won't pick up the change.
**Fix:** Restart the process (`Ctrl+C` the Streamlit server and re-run
`streamlit run dashboard/app.py`; restart the Jupyter kernel) after editing `.env`.

### `N_SYNTHETIC_RECORDS` change has no effect
**Cause:** `main.py` only generates synthetic data when `data/raw/accidents.csv`
**doesn't already exist**. If you already have a raw CSV (synthetic or real), changing
this variable does nothing until that file is removed.
**Fix:** `rm data/raw/accidents.csv` (and probably `data/processed/accidents_clean.csv`
too), then re-run `python main.py`.

### A `KMEANS_K` / `DBSCAN_EPS_KM` change doesn't show up in the dashboard
**Cause:** These only take effect the next time `python main.py` is run (they control
training-time clustering, not something the dashboard recomputes on the fly).
**Fix:** Re-run `python main.py` after changing them, then restart the dashboard.

## Data / File Issues

### `DataValidationError: Dataset is missing required columns: [...]`
**Cause:** You dropped in a real dataset (or edited the synthetic generator) without
matching the expected column names in `src/data/load_data.py`'s `REQUIRED_COLUMNS`.
**Fix:** Rename/add columns to match exactly:
`accident_id, date, hour, city, latitude, longitude, road_type, junction_type, weather,
light_condition, road_surface, speed_limit, num_vehicles_involved, severity`.

### `DataValidationError: Unexpected severity labels found: [...]`
**Cause:** Your dataset's severity column uses different label text (e.g.
`"Fatal Injury"` instead of `"Fatal"`, or numeric codes).
**Fix:** Map your labels to exactly `Minor`, `Serious`, `Fatal` before saving the CSV
(a one-line `df["severity"] = df["severity"].map({...})` in a preprocessing script of
your own, before it reaches `data/raw/accidents.csv`).

### `DataValidationError: N rows have out-of-range GPS coordinates`
**Cause:** Latitude/longitude columns are swapped, or contain a sentinel value like
`-999` for missing GPS data, or are in a non-decimal-degree format.
**Fix:** Confirm latitude is in `[-90, 90]` and longitude in `[-180, 180]`; if GPS is
genuinely missing for some rows, impute or drop them before ingestion rather than using
a sentinel value.

### Pipeline runs but `data/processed/accidents_clean.csv` looks unexpectedly small
**Cause:** `remove_duplicates()` may be dropping more rows than expected if your real
dataset has many legitimately-identical rows (e.g. two different accidents that happen
to share every recorded attribute, including timestamp, by coincidence).
**Fix:** Check `src/data/preprocess.py`'s `remove_duplicates()` — it currently treats
any two rows with identical values in every column except `accident_id` as duplicates.
If your data can have legitimate exact duplicates, exclude fewer columns from the
duplicate check or add a stricter uniqueness key.

## Model / Training Issues

### `SMOTE` raises `ValueError: Expected n_neighbors <= n_samples_fit`
**Cause:** The Fatal class has fewer than `k_neighbors + 1` (default 5+1=6) examples in
a training fold — happens if you drastically shrink the dataset size
(`N_SYNTHETIC_RECORDS`) or use a real dataset where Fatal is extremely rare in absolute
terms.
**Fix:** Increase dataset size, or lower `k_neighbors` in the `SMOTE(...)` call inside
`src/models/severity_model.py`'s `_make_pipeline()`.

### All four severity models report `fatal_recall: 0.0`
**Cause:** Either a genuinely very hard dataset for this feature set, or (if you changed
`config.FATAL_DECISION_THRESHOLD`) an oversized threshold that never triggers.
**Fix:** First check `models/metrics.json`'s ROC-AUC — if it's meaningfully above 0.5,
the models can discriminate somewhat and the issue is likely the decision threshold
(see `docs/CODE_EXPLANATION.md` §3-4); try lowering `FATAL_DECISION_THRESHOLD` in
`config.py`, or inspect whether the training data's Fatal examples are genuinely
separable given the features available.

### `ValueError: Parameter 'labels' must be ordered` from inside `evaluate_classifier`
**Cause:** This was a real bug hit during development — `roc_auc_score` requires
`labels` sorted ascending for `multi_class="ovr"`.
**Fix:** Already handled in the shipped `src/utils/metrics.py` (it reorders columns
before calling `roc_auc_score`). If you're extending this function and see the error
again, you likely bypassed that reordering step — reapply it.

### `main.py` runs successfully but takes far longer than ~30 seconds
**Cause:** Usually `N_SYNTHETIC_RECORDS` set very high, or the DBSCAN/KMeans step
running on a much larger coordinate set than default.
**Fix:** This is expected — clustering and SMOTE-based training both scale with row
count. Reduce `N_SYNTHETIC_RECORDS` for faster iteration during development, and only
run the full-size dataset for final results.

## Dashboard Issues

### "No trained artifacts found in `models/`"
**Cause:** `python main.py` hasn't been run yet, or `models/` was deleted/moved/renamed.
**Fix:** Run `python main.py` from the project root, then restart the dashboard.

### Dashboard starts but the Hotspot Map tab is blank
**Cause:** `streamlit-folium` not installed, or `models/hotspots_ranked.csv` is empty
(e.g. every point was classified as DBSCAN noise because `DBSCAN_MIN_SAMPLES`/
`DBSCAN_EPS_KM` are misconfigured for your dataset's density).
**Fix:** Confirm `streamlit-folium` is installed; check
`pd.read_csv("models/hotspots_ranked.csv")` isn't empty; if it is, loosen
`DBSCAN_EPS_KM` (larger radius) or lower `DBSCAN_MIN_SAMPLES` in `config.py` and re-run
`python main.py`.

### `st_folium` raises a JSON-serialization error
**Cause:** Usually a `NaN` or `numpy` type (e.g. `numpy.int64`) being passed into a
Folium popup/tooltip string instead of a native Python type.
**Fix:** Cast values to native Python types before building f-strings for popups (the
shipped `dashboard/app.py` already does this via `int(...)`/`float(...)` — if you add
new fields to a popup, cast them the same way).

### Port 8501 already in use
**Cause:** Another Streamlit process (or a previous run that didn't shut down) is still
bound to the port.
**Fix:** `streamlit run dashboard/app.py --server.port 8502`, or find and stop the other
process (`lsof -i :8501` on macOS/Linux, then `kill <PID>`).

### Predictions in the dashboard seem to ignore some of the input fields
**Cause:** Most likely the `date` field used to derive `is_weekend` — the dashboard
constructs a fixed placeholder date (`2024-01-06` for weekend, `2024-01-01` for weekday)
rather than a real calendar date, since only the weekend/weekday distinction actually
matters to the model (see `FEATURE_COLUMNS` in `feature_engineering.py` — the actual
calendar date isn't a feature, only `is_weekend` derived from it is).
**Fix:** This is expected behaviour, not a bug — the model was never trained on the
specific date, only on day-of-week-derived features. If you want the model to be
sensitive to more time information (e.g. `day_of_week` as a category, or `month` for
seasonality), add it to `FEATURE_COLUMNS` and retrain.

## Deployment Considerations

This project ships as a local pipeline + local dashboard, not a hosted service — but if
deploying `dashboard/app.py` (e.g. to Streamlit Community Cloud or a VM):

### Models aren't present after deployment
**Cause:** `models/` is `.gitignore`'d (generated artifacts aren't committed to version
control) — if you deploy from a fresh git clone, `models/` will be empty.
**Fix:** Either run `python main.py` as a build/startup step on the deployment target
before starting the dashboard, or intentionally commit a pre-trained `models/` folder
for a demo deployment (remove it from `.gitignore` if you do this).

### Deployed dashboard is slow on first load
**Cause:** `@st.cache_resource` caches per-process, but the *first* request after a
fresh deploy/restart still pays the cost of loading every joblib artifact and building
the SHAP/LIME explainers from scratch.
**Fix:** Expected and generally acceptable (a few seconds); if it matters for your
use case, consider a lighter-weight background sample size in `main.py`'s
`background_sample = X_test.sample(min(300, ...))` line to speed up SHAP explainer
construction.

### Memory issues on a small deployment instance
**Cause:** XGBoost/LightGBM/SHAP all have meaningfully sized dependencies loaded at
import time, and `data/processed/accidents_clean.csv` plus every `models/*.joblib`
file are loaded fully into memory.
**Fix:** For a resource-constrained deployment target, reduce `N_SYNTHETIC_RECORDS` (or
use a smaller real dataset), and confirm the instance has at least ~1 GB free RAM
beyond what the base Python/OS footprint needs.

### Environment variables not respected in a deployment platform
**Cause:** Many hosting platforms (Streamlit Community Cloud, Docker, etc.) don't
automatically read a `.env` file the way `python-dotenv` does locally — they expect
variables set through the platform's own secrets/environment configuration UI.
**Fix:** Set the same variable names listed in `.env.example` through your deployment
platform's environment variable configuration instead of relying on a committed `.env`
file (and never commit a real `.env` file with anything sensitive in it, even though
this project's variables aren't secrets by default).
