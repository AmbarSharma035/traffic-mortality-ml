# Using Real UK STATS19 Data in Traffic Mortality ML

This guide explains how to replace the project's synthetic data with **real UK Department for Transport (DfT) STATS19 road-collision data**. The intended dataset for this project is STATS19 only. Do not combine it with Indian, US, Kaggle, or synthetic accident records.

> **Coverage note:** DfT's published STATS19 open data covers reported personal-injury road collisions on public roads in **Great Britain (England, Scotland and Wales)**. It is not a complete record of every collision, and it does not cover Northern Ireland. It is based on collisions reported to and recorded by police.

## 1. Download the official data

1. Open the DfT [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) page.
2. Start with one or a few recent complete years to develop and test the pipeline. The page provides annual files, a last-five-years bundle, and full historical files.
3. Download the **Collisions** CSV file(s). This is the core table because it contains one record per collision and includes the collision circumstances, severity, date/time and location used by this project.
4. Download the matching **Vehicles** and/or **Casualties** files only if you decide to add vehicle- or casualty-level features. They are separate tables and have multiple rows per collision.
5. Download and keep the official **Road Safety Open Data Guide** for the same data release. Use its lookup tables to decode the coded values. Also read the [STATS19 guidance](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance).

The DfT data page is updated as releases are published. Use the latest **final validated** year available for a reproducible project, and record the exact years and download date in your report. Do not assume that a provisional release is equivalent to final data.

## 2. Understand the three STATS19 tables

| File | Row meaning | How this project should use it |
|---|---|---|
| Collisions | One row per collision | Primary modelling table. Start here. |
| Vehicles | One row per vehicle involved | Optional. Aggregate to one row per collision before joining. |
| Casualties | One row per casualty | Optional. Aggregate to one row per collision before joining. |

The common collision identifier is called `accident_index` in older releases and may be named differently in newer specifications. **Check the exact identifier and column names in the guide for the release you downloaded.** Never join vehicle or casualty rows directly to the collision table without aggregation: doing so duplicates collisions and biases the training data.

For the first working real-data version, use the collision table alone. It already contains the main circumstances needed for severity prediction and hotspot detection.

## 3. Put the downloaded files in the repository

Keep the original files unchanged so the data source remains auditable.

Suggested layout:

```text
traffic-mortality-ml/
├── data/
│   ├── raw/
│   │   └── stats19/
│   │       ├── collisions_2024.csv
│   │       └── collisions_2025.csv
│   └── processed/
├── models/
└── ...
```

Use the actual years you downloaded. Do not commit large raw datasets to GitHub unless you have checked repository size and the DfT licence/redistribution terms. Keep them local and add the relevant raw-data directory to `.gitignore` if necessary.

## 4. Convert STATS19 into the project's internal schema

The current project does **not** read STATS19 columns directly. Its validator in `src/data/load_data.py` expects these 14 internal columns:

```text
accident_id, date, hour, city, latitude, longitude,
road_type, junction_type, weather, light_condition,
road_surface, speed_limit, num_vehicles_involved, severity
```

Create a dedicated adapter, for example:

```text
src/data/stats19_adapter.py
```

Its job is to read the DfT collision CSV, normalize release-specific column names, decode coded values using the official lookup guide, and produce exactly one internal row per collision.

### Recommended field mapping

The names below describe the concepts to map. Exact raw column names and codes can vary by STATS19 specification/year, so verify them against the downloaded file and its matching DfT data guide.

| Project column | STATS19 source concept | Conversion |
|---|---|---|
| `accident_id` | Collision identifier (`accident_index` in older data) | Copy as a string; do not convert an identifier to a number. |
| `date` | Collision date | Parse using the format in the file. |
| `hour` | Collision time | Parse the hour from the time field; handle missing/invalid times explicitly. |
| `city` | Local authority / local authority district | Prefer renaming this concept to `local_authority` in the project. It is not necessarily a city. If retaining `city` temporarily for compatibility, document that it contains local-authority labels/codes, not city names. |
| `latitude`, `longitude` | Collision coordinates | Convert to numeric; remove or flag invalid/missing coordinates for spatial analysis. |
| `road_type` | Road type | Decode with the official lookup table. |
| `junction_type` | Junction detail | Decode with the official lookup table. |
| `weather` | Weather conditions | Decode with the official lookup table. |
| `light_condition` | Light conditions | Decode with the official lookup table. |
| `road_surface` | Road surface conditions | Decode with the official lookup table. |
| `speed_limit` | Speed limit | Convert to numeric; keep missing/unknown values as missing until the cleaning stage. |
| `num_vehicles_involved` | Number of vehicles | Convert to integer after validating missing values. |
| `severity` | Accident/collision severity | Map the official codes to the project's target labels as described below. |

### Severity mapping is especially important

In the traditional STATS19 coding, collision severity is represented as:

- `1` = Fatal
- `2` = Serious
- `3` = Slight

The existing model expects `Minor`, `Serious`, and `Fatal`. For a minimal compatibility change, map:

| STATS19 label | Existing project label |
|---|---|
| Slight | Minor |
| Serious | Serious |
| Fatal | Fatal |

This is a compatibility mapping, **not a claim that “Slight” and “Minor” are identical definitions**. Prefer changing the project's target vocabulary and dashboard display from `Minor` to `Slight` so the model reports the official STATS19 terminology consistently. If you keep `Minor`, explain the mapping in the report and dashboard.

Newer STATS19 specifications and severity-adjustment fields may differ from older files. Follow the DfT guide for the exact release. Do not mix adjusted and unadjusted severity definitions without documenting the choice. For a student project, state clearly whether the target uses the published collision-severity field or an adjusted field.

## 5. Do not invent fields that STATS19 does not provide

The synthetic generator creates convenient categories such as Indian city names, generic weather labels, and road-surface labels. Real STATS19 values are coded and have different category names.

- **Do not keep the synthetic Indian city coordinates or city names.** Remove the synthetic generator's geographic assumptions. Use the collision's actual Great Britain coordinates.
- **Do not randomly create missing weather, road, junction, or location values.** Decode the official codes and preserve unknown/not-recorded values as missing or an explicit `Unknown` category.
- **Do not assume every STATS19 release has exactly the same schema.** The 2024 specification introduced changes. Build and test the adapter against each release you intend to use.
- **Do not convert unknown or missing codes into a real condition** such as Clear, Dry, or Daylight. That would introduce false information.
- **Do not describe this as all road crashes in the UK.** The open data represents police-reported personal-injury collisions on public roads in Great Britain.

## 6. Update category handling in the project

The current `config.py` contains synthetic category lists and weights, for example `ROAD_TYPES`, `WEATHER_CONDITIONS`, `LIGHT_CONDITIONS`, `ROAD_SURFACE`, and the dictionaries used by `weather_severity_index`.

Update these to match the **decoded STATS19 categories actually present in your selected years**. In particular:

1. Use the official decoded road-type, junction, weather, light and surface labels.
2. Update `WEATHER_SEVERITY_WEIGHTS`, `LIGHT_SEVERITY_WEIGHTS`, and `SURFACE_SEVERITY_WEIGHTS` so every retained category has an intentional documented treatment.
3. Do not assign a severity weight to an unknown category as if it were a known condition. Keep an explicit unknown value or missing indicator.
4. Update dashboard dropdowns and validation lists to use the same categories as training.
5. If the project changes `city` to `local_authority`, update `src/data/load_data.py`, `src/data/preprocess.py`, `src/features/feature_engineering.py`, model feature lists, risk model inputs, and dashboard controls together.

The current `OneHotEncoder(handle_unknown="ignore")` can tolerate unseen category values at prediction time, but it does not fix incorrect labels or missing-value handling. Training and dashboard categories should still be consistent.

## 7. Add the STATS19 adapter to the pipeline

The current `main.py` automatically generates synthetic data whenever `config.DATA_RAW_PATH` does not exist. Merely downloading STATS19 somewhere else will not stop that behaviour.

Recommended implementation:

1. Add a setting such as `DATA_SOURCE=stats19` and a path for the converted collision-level CSV.
2. Add `src/data/stats19_adapter.py` to load and transform the original DfT collision file(s).
3. In `main.py`, make the selected source explicit:
   - If `DATA_SOURCE=stats19`, run the STATS19 adapter and fail with a clear error if the required raw files are missing.
   - Do not fall back to synthetic generation when STATS19 mode is selected.
   - Keep synthetic generation available only as an explicitly selected development/demo option, or remove it if the project must be real-data-only.
4. Pass the adapter output through the existing validation, cleaning, feature engineering, model training, risk scoring and hotspot stages.
5. Save the converted dataset to a separate processed path, for example `data/processed/stats19_collisions.csv`.
6. Log the source, years, row counts, severity distribution, missingness and any rows excluded during conversion.

**Important:** Do not simply set `DATA_RAW_PATH` to a raw STATS19 file and run `python main.py`. The current loader expects the project's 14-column schema and will reject raw STATS19 columns.

## 8. Check the three project outputs against the real data

### A. Severity prediction

- Train on the collision-level target (Fatal / Serious / Slight, or the documented compatibility mapping).
- Keep the stratified train/test split and fit SMOTE only on the training data, inside the training pipeline. Never apply SMOTE before splitting.
- Check class counts before training. Fatal collisions may be rare; report fatal precision, fatal recall and the confusion matrix, not accuracy alone.
- Use a held-out test set and, if possible, a chronological evaluation (train on earlier years, test on a later year) to check performance on future data.
- Recalculate and justify the fatal decision threshold using validation data; do not assume the synthetic-data threshold of 0.15 is appropriate for STATS19.

### B. Risk scoring

The current risk score groups collisions by road/junction/weather/time/weekend and uses the severity weights in `config.py`. Recheck these weights and the smoothing strength on the real class distribution. Describe the result as a historical segment-level score, not a guaranteed prediction that a collision will occur.

### C. Hotspot detection

Use valid STATS19 longitude/latitude values. Check coordinate ranges and missingness before clustering. Great Britain-wide data spans a large geographic area, so one global K-Means/DBSCAN run with the current default parameters may not produce useful local hotspots. Consider running clustering by region/local authority or using a distance-based method with a suitable projection (for example, British National Grid) and a documented distance threshold. Do not interpret clusters as official black spots without validation.

## 9. Run the project with STATS19

After the adapter and source-selection changes have been implemented:

```bash
# Activate your existing virtual environment first
pip install -r requirements.txt

# Select STATS19 in your .env (after adding the setting to the code)
DATA_SOURCE=stats19

# Run conversion/validation/training and rebuild all model artifacts
python main.py

# Start the dashboard only after training succeeds
streamlit run dashboard/app.py

# Run tests
pytest
```

The dashboard reads saved model artifacts. Every time you change the source data, mapping, feature engineering or model settings, rerun `python main.py` before restarting the dashboard. Do not reuse model files trained on synthetic data.

## 10. Tests to add before trusting the results

Add tests for at least the following:

- A small sample of each downloaded STATS19 release converts to the expected internal columns.
- Collision identifiers remain unique and one row represents one collision.
- Severity codes map correctly and unexpected codes cause a clear validation error.
- All coded categories are decoded using the correct release's lookup table.
- Missing and unknown values are not silently changed to Clear/Dry/Daylight.
- Dates, times, coordinates and speed limits are parsed correctly.
- Joining optional vehicle/casualty tables does not multiply collision rows.
- The STATS19 source mode never calls the synthetic generator.
- The final model table contains no missing target labels and has documented feature missingness.
- The dashboard loads artifacts retrained from STATS19 and displays the official severity wording.

## 11. Suggested implementation order

1. Download one complete-year collision file and its matching DfT data guide.
2. Inspect its column names, unique coded values, missing values and severity distribution.
3. Implement the adapter and export a small converted sample.
4. Add tests for schema, code decoding, severity mapping and one-row-per-collision.
5. Update category weights, feature definitions and dashboard controls.
6. Change `main.py` so STATS19 is an explicit source and synthetic generation cannot run in STATS19 mode.
7. Train and evaluate the models; review Fatal recall and precision.
8. Validate risk scores and geographically meaningful hotspots.
9. Retrain all artifacts and verify every dashboard tab.
10. Record the DfT download URL, release years, data-guide version, conversion rules, exclusions and limitations in the project report.

## 12. Official references

- [DfT Road safety open data — downloads, data guide and release notes](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)
- [DfT Road accident and safety statistics guidance — coverage, definitions and methodology](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
- [STATS19 forms and guidance](https://www.gov.uk/government/publications/stats19-forms-and-guidance)
- [STATS19 data structure overview (rOpenSci documentation)](https://docs.ropensci.org/stats19/articles/stats19.html)

---

## Definition of done

The project is genuinely using real STATS19 data only when:

- The model's training rows come from the selected DfT STATS19 collision files.
- The conversion process uses the official lookup guide for the relevant release.
- No synthetic or non-UK accident rows are present in the training data.
- STATS19 mode cannot silently generate or substitute synthetic data.
- All model artifacts have been retrained from the converted STATS19 data.
- The dashboard displays the same categories and severity definitions used during training.
- The report clearly states the Great Britain coverage and police-reported personal-injury collision limitations.
