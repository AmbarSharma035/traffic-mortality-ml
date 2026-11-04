
"""
End-to-end pipeline runner for real UK STATS19 accident data.

Pipeline:
1. Load and validate real STATS19 data.
2. Clean and save accident records.
3. Generate derived features.
4. Train and evaluate four severity classifiers.
5. Build the historical risk lookup and regressor.
6. Detect geographic hotspots.
7. Save the SHAP background sample.
8. Save model artifacts and evaluation metrics.

Synthetic data generation is disabled.
"""

import json
import time

import joblib
import pandas as pd

import config

from src.data.load_data import load_raw_data, profile
from src.data.preprocess import clean, save_cleaned_data

from src.features.feature_engineering import (
    add_derived_features,
    prepare_model_table,
)

from src.hotspot.hotspot_detection import (
    rank_hotspots,
    run_dbscan,
    run_kmeans,
)

from src.models.risk_model import (
    build_risk_lookup,
    train_risk_regressor,
)

from src.models.severity_model import (
    cross_validate,
    select_best_model,
    train_and_evaluate,
)


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main():

    start = time.time()

    # --------------------------------------------------
    # STEP 0 - LOAD REAL DATA
    # --------------------------------------------------

    section("STEP 0 - Loading REAL STATS19 data")

    if not config.DATA_RAW_PATH.exists():
        raise FileNotFoundError(
            "\nREAL STATS19 DATASET NOT FOUND.\n"
            f"Expected file: {config.DATA_RAW_PATH}\n"
            "Prepare the STATS19 dataset first.\n"
            "Synthetic data generation is disabled."
        )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(config.DATA_RAW_PATH)

    print("Dataset:", config.DATA_RAW_PATH)
    print(f"Loaded records: {len(raw):,}")
    print("\nDataset profile:")
    print(json.dumps(profile(raw), indent=2, default=str))

    actual_classes = set(raw["severity"].dropna().unique())
    expected_classes = set(config.SEVERITY_CLASSES)

    missing_classes = expected_classes - actual_classes

    if missing_classes:
        raise ValueError(
            f"Dataset is missing severity classes: {missing_classes}"
        )

    # --------------------------------------------------
    # STEP 1 - CLEAN DATA
    # --------------------------------------------------

    section("STEP 1 - Data cleaning")

    cleaned = clean(raw)

    save_cleaned_data(cleaned)

    print(f"Cleaned records: {len(cleaned):,}")

    print("\nSeverity distribution:")
    print(cleaned["severity"].value_counts())

    # --------------------------------------------------
    # STEP 2 - FEATURE ENGINEERING
    # --------------------------------------------------

    section("STEP 2 - Feature engineering")

    featured_full = add_derived_features(cleaned)

    model_table = prepare_model_table(cleaned)

    print("Feature engineering completed.")
    print("Model table shape:", model_table.shape)
    print("Model features:", model_table.columns.tolist())

    # --------------------------------------------------
    # STEP 3 - PILLAR 1: SEVERITY PREDICTION
    # --------------------------------------------------

    section("STEP 3 - Pillar 1: Severity Prediction")

    results, fitted, X_test, y_test = train_and_evaluate(
        model_table
    )

    best_name, best_pipeline, best_metrics, ranking = (
        select_best_model(results, fitted)
    )

    print("\nBest model by Fatal recall:", best_name)
    print("Model ranking:", ranking)

    print("\nBest model metrics:")
    print(best_metrics)

    print("\nRunning stratified cross-validation...")

    cv_scores = cross_validate(
        model_table,
        best_name,
    )

    print(
        "Cross-validation Fatal recall:",
        [round(score, 3) for score in cv_scores]
    )

    severity_bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
    }

    joblib.dump(
        severity_bundle,
        config.SEVERITY_MODEL_PATH,
    )

    print("Saved severity model:", config.SEVERITY_MODEL_PATH)

    # Save a background sample for dashboard explanations.
    background_sample = X_test.sample(
        n=min(300, len(X_test)),
        random_state=config.RANDOM_SEED,
    )

    joblib.dump(
        background_sample,
        config.SHAP_BACKGROUND_PATH,
    )

    print("Saved SHAP background:", config.SHAP_BACKGROUND_PATH)

    metrics_out = {
        "dataset": "UK STATS19",
        "dataset_records": len(cleaned),
        "severity": {
            name: {
                key: value
                for key, value in metrics.items()
                if key != "per_class"
            }
            for name, metrics in results.items()
        },
        "severity_best_model": best_name,
        "severity_cv_fatal_recall": cv_scores,
    }

    # --------------------------------------------------
    # STEP 4 - PILLAR 2: RISK PREDICTION
    # --------------------------------------------------

    section("STEP 4 - Pillar 2: Accident Risk Prediction")

    risk_lookup = build_risk_lookup(featured_full)

    if risk_lookup.empty:
        raise ValueError("Risk lookup is empty. Cannot train risk model.")

    risk_lookup.to_csv(
        config.RISK_LOOKUP_PATH,
        index=False,
    )

    print(f"Risk segments created: {len(risk_lookup):,}")
    print("\nHighest-scoring risk segments:")
    print(risk_lookup.head(5).to_string(index=False))

    risk_regressor = train_risk_regressor(risk_lookup)

    joblib.dump(
        risk_regressor,
        config.RISK_MODEL_PATH,
    )

    print("Saved risk model:", config.RISK_MODEL_PATH)

    # --------------------------------------------------
    # STEP 5 - PILLAR 3: HOTSPOT DETECTION
    # --------------------------------------------------

    section("STEP 5 - Pillar 3: Hotspot Detection")

    print("Running K-Means...")

    km_labels, km_model, km_silhouette = run_kmeans(cleaned)

    print(
        f"K-Means: {config.KMEANS_K} clusters, "
        f"sampled silhouette={km_silhouette}"
    )

    print("\nRunning sampled DBSCAN...")

    db_labels, db_model, db_silhouette, n_noise = run_dbscan(
        cleaned
    )

    # Count actual clusters from the labels fitted by DBSCAN.
    # -1 means noise and is not a cluster.
    sampled_labels = db_model.labels_

    n_clusters = len(set(sampled_labels)) - (
        1 if -1 in sampled_labels else 0
    )

    n_sampled = len(sampled_labels)
    n_not_sampled = int((db_labels == -2).sum())

    print(
        f"DBSCAN: {n_clusters} clusters, "
        f"{n_noise} noise points, "
        f"{n_not_sampled} records not sampled, "
        f"sampled silhouette={db_silhouette}"
    )

    print("\nRanking discovered DBSCAN clusters...")

    ranked = rank_hotspots(
        cleaned,
        db_labels,
    )

    if ranked.empty:
        raise ValueError(
            "DBSCAN did not produce any rankable clusters. "
            "Review DBSCAN parameters and coordinates."
        )

    ranked.to_csv(
        config.HOTSPOT_TABLE_PATH,
        index=False,
    )

    print(f"Saved ranked hotspots: {config.HOTSPOT_TABLE_PATH}")
    print("\nTop hotspots:")
    print(ranked.head(10).to_string(index=False))

    metrics_out["hotspot"] = {
        "kmeans_k": config.KMEANS_K,
        "kmeans_silhouette": km_silhouette,
        "dbscan_n_clusters": int(n_clusters),
        "dbscan_n_noise": int(n_noise),
        "dbscan_n_sampled": int(n_sampled),
        "dbscan_n_not_sampled": n_not_sampled,
        "dbscan_silhouette": db_silhouette,
        "hotspot_records_ranked": len(ranked),
    }

    # --------------------------------------------------
    # STEP 6 - SAVE METRICS
    # --------------------------------------------------

    section("STEP 6 - Saving evaluation metrics")

    with open(
        config.METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_out,
            file,
            indent=2,
            default=str,
        )

    print("Metrics saved:", config.METRICS_PATH)

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    section("STATS19 PIPELINE COMPLETED")

    print(f"Total pipeline time: {time.time() - start:.1f} seconds")

    print("\nReal STATS19 data was used.")
    print("Synthetic data generation was not used.")

    print("\nSaved artifacts:")
    print("Severity model:", config.SEVERITY_MODEL_PATH)
    print("Risk model:", config.RISK_MODEL_PATH)
    print("Risk lookup:", config.RISK_LOOKUP_PATH)
    print("Hotspot table:", config.HOTSPOT_TABLE_PATH)
    print("SHAP background:", config.SHAP_BACKGROUND_PATH)
    print("Metrics:", config.METRICS_PATH)

    print("\nDashboard command:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
