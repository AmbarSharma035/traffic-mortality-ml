# Data Flow

This traces one piece of data all the way through the system, using the actual column
names and function names in this codebase — not a generic description.

## Stage 0: Where the data comes from

Either:
- **Synthetic** (default, no setup required): `src/data/generate_sample_data.py`'s
  `generate()` builds a 20,000-row dataset with realistic structure — a logistic
  "latent risk" model decides each row's severity based on speed limit, weather, light
  condition, road type, road surface and whether the point falls in one of 15
  deliberately-injected dangerous "black-spot" junctions (3 per city × 5 cities), then
  adds Gaussian noise so the relationship isn't perfectly clean. It also injects ~2%
  missing values into `weather`/`road_surface`/`speed_limit` and duplicates ~0.5% of
  rows, so the cleaning stage downstream has real (if synthetic) messiness to handle.
- **Real**: any CSV with the same 14 columns dropped into `data/raw/accidents.csv` (see
  `docs/SETUP_GUIDE.md` → "Using a Real Dataset Instead of Synthetic Data").

Either way, the pipeline from this point on is identical.

## Stage 1: Ingestion & Validation
**File:** `src/data/load_data.py` · **Function:** `load_raw_data()` → `validate_schema()`

```
data/raw/accidents.csv
  → pd.read_csv(parse_dates=["date"])
  → validate_schema(df):
        - all 14 REQUIRED_COLUMNS present?          -> else DataValidationError
        - df not empty?                               -> else DataValidationError
        - every severity value in {Minor,Serious,Fatal}? -> else DataValidationError
        - latitude in [-90,90] and longitude in [-180,180]? -> else DataValidationError
  → profile(df): row count, duplicate count, missing % per column,
                 severity class distribution, date range
```
**Why this matters:** every downstream model assumes these columns and this severity
vocabulary exist. Failing loudly here (via `DataValidationError`) instead of silently
continuing is what stops a malformed dataset from quietly producing a broken model 20
minutes into training.

## Stage 2: Preprocessing / Cleaning
**File:** `src/data/preprocess.py` · **Function:** `clean()`

```
raw df (20,100 rows, in the default synthetic run)
  → remove_duplicates(): drops exact duplicate rows (ignoring accident_id)
        20,100 -> 20,000 rows (100 synthetic duplicates removed)
  → impute_missing():
        categorical columns (weather, road_surface, junction_type, light_condition,
        road_type, city) -> filled with each column's mode
        numeric columns (speed_limit, num_vehicles_involved, hour, latitude,
        longitude) -> filled with each column's median
  → coerce_types(): date -> datetime64, hour/speed_limit/num_vehicles_involved -> int
  → data/processed/accidents_clean.csv (20,000 rows, zero missing values)
```
**Why mode/median instead of dropping rows:** the missing values are concentrated in
`weather`/`road_surface`/`speed_limit` — dropping those rows would disproportionately
remove exactly the bad-weather records the severity model most needs to learn from.

## Stage 3: Feature Engineering
**File:** `src/features/feature_engineering.py` · **Function:** `add_derived_features()`
/ `prepare_model_table()`

```
cleaned df
  → add_derived_features():
        day_of_week          <- date.dt.day_name()
        is_weekend           <- date.dt.dayofweek in {5,6}                (0/1)
        is_rush_hour         <- hour in config.RUSH_HOURS (8-10, 17-20)   (0/1)
        is_night             <- light_condition in {Dark-Lit, Dark-Unlit}(0/1)
        hour_bucket          <- pd.cut(hour, 5 named bins)
        weather_severity_index <- weather_weight + light_weight + surface_weight
                                   (0 = best case, higher = worse; see config.py's
                                    WEATHER_SEVERITY_WEIGHTS etc.)
  → prepare_model_table(): keeps only FEATURE_COLUMNS + "severity"
        FEATURE_COLUMNS = 7 categorical + 7 numeric = 14 columns
```
This is the last shared step — from here, the three pillars diverge.

## Stage 4a: Pillar 1 — Severity Prediction
**File:** `src/models/severity_model.py` · **Function:** `train_and_evaluate()`

```
model_table (14 features + severity)
  → X = FEATURE_COLUMNS, y = severity
  → y_encoded = encode_labels(y)          # "Minor"->0, "Serious"->1, "Fatal"->2
  → train_test_split(stratify=y_encoded, test_size=0.2)
  → for each of 4 candidate pipelines (LR / RF / XGBoost / LightGBM):
        ColumnTransformer: one-hot encode 7 categoricals, StandardScale 7 numerics
          -> SMOTE (fit ONLY on the training fold, after encoding, before the
             classifier - never touches the test fold, avoiding data leakage)
          -> classifier.fit(X_train_resampled, y_train_resampled)
        predict_proba(X_test) -> reordered to [P(Minor), P(Serious), P(Fatal)]
        evaluate_classifier(): accuracy, macro-F1, ROC-AUC (OvR), and critically
             fatal_recall / fatal_precision / fatal_f1
  → rank_models_by_fatal_recall(): sort by (fatal_recall, macro_f1) descending
  → best model selected (see docs/CODE_EXPLANATION.md for why Logistic Regression
    wins on this dataset)
  → joblib.dump({"pipeline":..., "model_name":...}) -> models/severity_model.joblib
```

**At inference time** (dashboard `predict_single()`):
```
one input row (from the Streamlit form)
  → add_derived_features()             # same function as training
  → pipeline.predict_proba(row)         # same fitted ColumnTransformer + classifier
  → apply_fatal_threshold():
        if P(Fatal) >= config.FATAL_DECISION_THRESHOLD (0.15): predict Fatal
        else: argmax(P(Minor), P(Serious), P(Fatal))
  → returned to dashboard as {probabilities, argmax_prediction, recall_oriented_prediction}
```

## Stage 4b: Pillar 2 — Risk Prediction
**File:** `src/models/risk_model.py` · **Function:** `build_risk_lookup()` →
`train_risk_regressor()`

```
featured df (accident-level rows, with derived features)
  → groupby(["road_type","junction_type","weather","hour_bucket","is_weekend"])
  → per group: n_accidents, weighted_sum (Minor=1, Serious=3, Fatal=6 per config.py),
               fatal_count, serious_count
  → smoothed_avg_weight = (weighted_sum + k * global_avg_weight) / (n_accidents + k)
        (k=8; pulls low-count segments toward the dataset-wide average instead of
         letting 2 accidents that happened to be Fatal produce an extreme score)
  → risk_score = smoothed_avg_weight / max_possible_weight * 100   (0-100, clipped)
  → risk_level = Low / Medium / High / Critical (config.RISK_LEVELS thresholds)
  → models/risk_lookup.csv (622 segments, in the default synthetic run)

risk_lookup
  → train_risk_regressor(): GradientBoostingRegressor on the same 5 categorical/
    binary segment keys (one-hot encoded), sample_weight = log1p(n_accidents) so
    well-observed segments count more during training
  → models/risk_model.joblib
```

**At inference time** (dashboard `score_segment()`):
```
(road_type, junction_type, weather, hour_bucket, is_weekend) chosen in the UI
  → one-row DataFrame -> risk_model.joblib.predict() -> clipped to [0,100]
  → risk_level looked up from the same thresholds
```
This is what lets the dashboard score a combination that has few or zero historical
accidents recorded — the regressor generalizes from segments that *were* observed.

## Stage 4c: Pillar 3 — Hotspot Detection
**File:** `src/hotspot/hotspot_detection.py`

```
cleaned df (latitude, longitude only - no severity or time used for clustering itself)
  → run_kmeans(k=12): sklearn.cluster.KMeans on raw (lat, lon) pairs
        -> cluster label per row + silhouette score
  → run_dbscan(eps=1.5km converted to degrees, min_samples=8): sklearn.cluster.DBSCAN
        -> cluster label per row (-1 = noise/outlier), silhouette score on non-noise
           points, noise-point count
  → fit_kde(bandwidth=0.02 deg): sklearn.neighbors.KernelDensity, for the dashboard's
        continuous density surface (evaluated on a grid via kde_grid())
  → rank_hotspots(df, dbscan_labels):
        drop noise (cluster == -1)
        per cluster: n_accidents, fatal_count, serious_count, centroid (mean lat/lon),
                     dominant_city (mode), severity_weighted_score (sum of
                     config.RISK_SEVERITY_WEIGHTS per accident in that cluster)
        sort by severity_weighted_score descending, keep top 25
  → models/hotspots_ranked.csv
```

## Stage 5: Explainability
**File:** `src/explainability/explain.py`

```
best severity pipeline + a 300-row background sample of X_test
  → build_shap_explainer():
        TreeExplainer for RF/XGBoost/LightGBM, or a generic Explainer wrapping
        predict_proba for Logistic Regression
  → global_feature_importance(): mean |SHAP value| across the background sample,
        for the Fatal class (class_idx=2) by default
  → local_explanation_shap(one row): per-feature SHAP contribution for that one
        prediction
  → build_lime_explainer(): label-encodes the 7 categorical columns into small
        integers (LIME's internal requirement), returns (explainer, encoders)
  → local_explanation_lime(one row): LIME perturbs the encoded row, predict_fn
        decodes categoricals back to strings before calling pipeline.predict_proba,
        LIME fits a local linear surrogate and returns per-feature weights
```
SHAP and LIME are computed independently and shown side-by-side in the dashboard
(SHAP as the primary bar chart, LIME inside an expander) so a viewer can see whether
two different explanation methods agree — in practice, on this dataset, they largely do
(see `docs/CODE_EXPLANATION.md`).

## Stage 6: Output — the Dashboard
**File:** `dashboard/app.py`

```
models/severity_model.joblib, risk_model.joblib, risk_lookup.csv,
hotspots_ranked.csv, shap_background.joblib, metrics.json
  → loaded ONCE per server process via @st.cache_resource (load_artifacts(),
    load_explainers())
  → Tab 1 (Severity): user inputs -> predict_single() -> SHAP/LIME explanation
  → Tab 2 (Risk): user inputs -> score_segment() ; also shows the full sorted
        risk_lookup.csv table
  → Tab 3 (Hotspot): reads hotspots_ranked.csv -> folium.CircleMarker per cluster,
        sized/coloured by severity_weighted_score and rank
  → Tab 4 (Performance): reads metrics.json -> benchmark table across all 4 models,
        5-fold CV fatal-recall bar chart, hotspot clustering quality numbers
```

## Validation and Transformation Checkpoints, Summarised

| Stage | Validation / transformation | Why |
|---|---|---|
| Ingestion | Schema + severity-label + GPS-bounds check | Fail fast on malformed input, never silently |
| Cleaning | Dedup, mode/median impute, dtype coercion | Guarantees zero missing values downstream |
| Feature engineering | Derive temporal + composite-risk features | Same feature set used identically at training AND inference |
| Severity training | SMOTE strictly inside the training fold | Prevents oversampling from leaking into evaluation |
| Severity inference | Cost-sensitive Fatal threshold (not plain argmax) | Missing a Fatal case costs more than a false alarm |
| Risk aggregation | Laplace/credibility smoothing | Prevents small-sample segments from getting extreme scores |
| Hotspot ranking | Severity weighting applied only AFTER unsupervised clustering | Keeps clustering purely spatial, ranking purpose-built |
