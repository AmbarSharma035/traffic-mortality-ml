
"""
End-to-end pipeline runner for real UK STATS19 accident data.

Run using:
    python main.py

Pipeline:
1. Load and validate real STATS19 data.
2. Clean accident records.
3. Generate derived features.
4. Train severity classification models.
5. Build accident risk lookup and regressor.
6. Detect accident hotspots.
7. Build SHAP explainability artifacts.
8. Save model artifacts and evaluation metrics.

IMPORTANT:
This pipeline does not generate synthetic data.
If the configured real dataset is missing, execution stops.
"""

import json
import time

import joblib
import pandas as pd

import config

from src.data.load_data import load_raw_data, profile
from src.data.preprocess import clean

from src.explainability.explain import build_shap_explainer

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
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():

    start = time.time()

    # --------------------------------------------------
    # STEP 0: CHECK REAL DATASET
    # --------------------------------------------------

    section("STEP 0 - Loading REAL STATS19 data")

    if not config.DATA_RAW_PATH.exists():
        raise FileNotFoundError(
            "\nREAL STATS19 DATASET NOT FOUND.\n"
            f"Expected file: {config.DATA_RAW_PATH}\n"
            "Please prepare the STATS19 dataset first.\n"
            "Synthetic data generation is disabled."
        )

    print("Dataset selected:", config.DATA_RAW_PATH)
    print("Using real STATS19 data. Synthetic generation is disabled.")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(config.DATA_RAW_PATH)

    print(f"\nLoaded {len(raw):,} real accident records.")

    print("\nDataset profile:")
    print(json.dumps(profile(raw), indent=2, default=str))

    # Check that all three target classes are available.
    actual_classes = set(raw["severity"].dropna().unique())
    expected_classes = set(config.SEVERITY_CLASSES)

    missing_classes = expected_classes - actual_classes

    if missing_classes:
        raise ValueError(
            f"Dataset is missing severity classes: {missing_classes}"
        )

    # --------------------------------------------------
    # STEP 1: CLEAN DATA
    # --------------------------------------------------

    section("STEP 1 - Data cleaning")

    cleaned = clean(raw)

    config.DATA_PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        config.DATA_PROCESSED_PATH,
        index=False,
    )

    print(
        f"Cleaned dataset saved: {config.DATA_PROCESSED_PATH}"
    )

    print(f"Cleaned records: {len(cleaned):,}")

    print("\nSeverity distribution:")
    print(cleaned["severity"].value_counts())

    # --------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # --------------------------------------------------

    section("STEP 2 - Feature engineering")

    featured_full = add_derived_features(cleaned)

    model_table = prepare_model_table(cleaned)

    print("Feature engineering completed.")

    print("Model table shape:", model_table.shape)

    print("\nModel features:")
    print(model_table.columns.tolist())

    # --------------------------------------------------
    # STEP 3: PILLAR 1 - SEVERITY PREDICTION
    # --------------------------------------------------

    section(
        "STEP 3 - Pillar 1: Severity Prediction"
    )

    results, fitted, X_test, y_test = train_and_evaluate(
        model_table
    )

    best_name, best_pipeline, best_metrics, ranking = (
        select_best_model(results, fitted)
    )

    print(f"\nBest model by Fatal-class recall: {best_name}")

    print("\nModel ranking:")
    print(ranking)

    print("\nBest model metrics:")
    print(best_metrics)

    print("\n5-fold stratified cross-validation:")

    cv_scores = cross_validate(
        model_table,
        best_name,
    )

    print([
        round(score, 3)
        for score in cv_scores
    ])

    # Save winning severity model.
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
        },
        config.SEVERITY_MODEL_PATH,
    )

    print(
        f"\nSaved severity model: {config.SEVERITY_MODEL_PATH}"
    )

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

    # Save a small background sample for SHAP.
    background_sample = X_test.sample(
        n=min(300, len(X_test)),
        random_state=config.RANDOM_SEED,
    )

    joblib.dump(
        background_sample,
        config.SHAP_BACKGROUND_PATH,
    )

    # --------------------------------------------------
    # STEP 4: PILLAR 2 - RISK PREDICTION
    # --------------------------------------------------

    section("STEP 4 - Pillar 2: Accident Risk Prediction")

    risk_lookup = build_risk_lookup(featured_full)

    risk_lookup.to_csv(
        config.RISK_LOOKUP_PATH,
        index=False,
    )

    print(
        f"Risk lookup created: {len(risk_lookup):,} segments"
    )

    print("\nSample risk records:")

    print(
        risk_lookup.head(5)
    )

    risk_regressor = train_risk_regressor(
        risk_lookup
    )

    joblib.dump(
        risk_regressor,
        config.RISK_MODEL_PATH,
    )

    print(
        f"Saved risk model: {config.RISK_MODEL_PATH}"
    )

    # --------------------------------------------------
    # STEP 5: PILLAR 3 - HOTSPOT DETECTION
    # --------------------------------------------------

    section("STEP 5 - Pillar 3: Hotspot Detection")

    km_labels, km_model, km_silhouette = run_kmeans(
        cleaned
    )

    print(
        f"K-Means (k={config.KMEANS_K}): "
        f"silhouette={km_silhouette}"
    )

    db_labels, db_model, db_silhouette, n_noise = (
        run_dbscan(cleaned)
    )

    n_clusters = (
        len(set(db_labels))
        - (1 if -1 in db_labels else 0)
    )

    print(
        f"DBSCAN: {n_clusters} clusters found, "
        f"{n_noise} noise points, "
        f"silhouette={db_silhouette}"
    )

    ranked = rank_hotspots(
        cleaned,
        db_labels,
    )

    ranked.to_csv(
        config.HOTSPOT_TABLE_PATH,
        index=False,
    )

    print(
        f"Saved {len(ranked)} ranked hotspots."
    )

    metrics_out["hotspot"] = {
        "kmeans_k": config.KMEANS_K,
        "kmeans_silhouette": km_silhouette,
        "dbscan_n_clusters": n_clusters,
        "dbscan_n_noise": int(n_noise),
        "dbscan_silhouette": db_silhouette,
    }

    # --------------------------------------------------
    # STEP 6: EXPLAINABILITY
    # --------------------------------------------------

    section("STEP 6 - SHAP Explainability")

    explainer = build_shap_explainer(
        best_pipeline,
        background_sample,
    )

    joblib.dump(
        True,
        config.MODEL_DIR / "_shap_ok.flag",
    )

    print(
        "SHAP explainer built successfully."
    )

    # --------------------------------------------------
    # STEP 7: SAVE METRICS
    # --------------------------------------------------

    section("STEP 7 - Saving evaluation metrics")

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

    print(
        f"Metrics saved: {config.METRICS_PATH}"
    )

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    section("STATS19 PIPELINE COMPLETED")

    print(
        f"Total pipeline time: {time.time() - start:.1f} seconds"
    )

    print("\nReal STATS19 data was used.")
    print("Synthetic data generation was not used.")

    print(
        f"\nModel artifacts saved in: {config.MODEL_DIR}"
    )

    print(
        "\nDashboard command: streamlit run dashboard/app.py"
    )


if __name__ == "__main__":
    main()
