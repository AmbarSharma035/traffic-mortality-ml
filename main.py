"""
End-to-end pipeline runner.

    python main.py

Runs, in order: data generation (only if no raw CSV exists yet) ->
ingestion/validation -> cleaning -> feature engineering -> Pillar 1
(severity models, benchmarked, best one kept) -> Pillar 2 (risk lookup +
regressor) -> Pillar 3 (hotspot clustering + ranking) -> saves every
artifact the Streamlit dashboard needs into models/.

Re-run any time the synthetic-data seed, feature set, or model
hyperparameters change in config.py - everything downstream regenerates
from the same entry point instead of requiring several manual scripts.
"""
import json
import time

import joblib
import pandas as pd

import config
from src.data.load_data import load_raw_data, profile
from src.data.preprocess import clean
from src.explainability.explain import build_shap_explainer
from src.features.feature_engineering import add_derived_features, prepare_model_table
from src.hotspot.hotspot_detection import rank_hotspots, run_dbscan, run_kmeans
from src.models.risk_model import build_risk_lookup, train_risk_regressor
from src.models.severity_model import cross_validate, select_best_model, train_and_evaluate


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    start = time.time()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 0: data ----------------------------------------------------
    section("STEP 0 - Data")
    if not config.DATA_RAW_PATH.exists():
        print("No raw dataset found - generating synthetic data (see "
              "src/data/generate_sample_data.py to swap in a real CSV).")
        from src.data.generate_sample_data import generate
        config.DATA_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.DATA_RAW_PATH, index=False)

    raw = load_raw_data(config.DATA_RAW_PATH)
    print(f"Loaded {len(raw):,} raw records.")
    print(json.dumps(profile(raw), indent=2, default=str))

    cleaned = clean(raw)
    config.DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.DATA_PROCESSED_PATH, index=False)
    print(f"Cleaned dataset saved -> {config.DATA_PROCESSED_PATH} ({len(cleaned):,} rows)")

    featured_full = add_derived_features(cleaned)
    model_table = prepare_model_table(cleaned)

    # ---- Step 1: Pillar 1 - Severity Prediction --------------------------
    section("STEP 1 - Pillar 1: Severity Prediction (benchmarking 4 models)")
    results, fitted, X_test, y_test = train_and_evaluate(model_table)
    best_name, best_pipeline, best_metrics, ranking = select_best_model(results, fitted)
    print(f"\nBest model by Fatal-class recall: {best_name}")
    print(f"Ranking (best first): {ranking}")

    print("\n5-fold stratified CV (fatal recall per fold) for the winning model:")
    cv_scores = cross_validate(model_table, best_name)
    print([round(s, 3) for s in cv_scores])

    joblib.dump({"pipeline": best_pipeline, "model_name": best_name}, config.SEVERITY_MODEL_PATH)
    print(f"Saved severity model -> {config.SEVERITY_MODEL_PATH}")

    metrics_out = {
        "severity": {name: {k: v for k, v in m.items() if k != "per_class"} for name, m in results.items()},
        "severity_best_model": best_name,
        "severity_cv_fatal_recall": cv_scores,
    }

    # A small background sample is stored so the dashboard can build a SHAP
    # explainer at startup without needing the full training set on disk.
    background_sample = X_test.sample(min(300, len(X_test)), random_state=config.RANDOM_SEED)
    joblib.dump(background_sample, config.SHAP_BACKGROUND_PATH)

    # ---- Step 2: Pillar 2 - Risk Prediction -------------------------------
    section("STEP 2 - Pillar 2: Risk Prediction")
    risk_lookup = build_risk_lookup(featured_full)
    risk_lookup.to_csv(config.RISK_LOOKUP_PATH, index=False)
    print(f"Built risk lookup for {len(risk_lookup)} road/time/weather segments "
          f"-> {config.RISK_LOOKUP_PATH}")
    print(risk_lookup.head(5)[["road_type", "junction_type", "weather", "hour_bucket",
                                "risk_score", "risk_level"]])

    risk_regressor = train_risk_regressor(risk_lookup)
    joblib.dump(risk_regressor, config.RISK_MODEL_PATH)
    print(f"Saved risk regressor -> {config.RISK_MODEL_PATH}")

    # ---- Step 3: Pillar 3 - Hotspot Detection -----------------------------
    section("STEP 3 - Pillar 3: Hotspot Detection")
    km_labels, km_model, km_silhouette = run_kmeans(cleaned)
    print(f"K-Means (k={config.KMEANS_K}): silhouette={km_silhouette:.3f}")

    db_labels, db_model, db_silhouette, n_noise = run_dbscan(cleaned)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    print(f"DBSCAN: {n_clusters} clusters found, {n_noise} noise points, "
          f"silhouette={db_silhouette}")

    ranked = rank_hotspots(cleaned, db_labels)
    ranked.to_csv(config.HOTSPOT_TABLE_PATH, index=False)
    print(f"Saved top {len(ranked)} ranked black-spots -> {config.HOTSPOT_TABLE_PATH}")

    metrics_out["hotspot"] = {
        "kmeans_k": config.KMEANS_K,
        "kmeans_silhouette": km_silhouette,
        "dbscan_n_clusters": n_clusters,
        "dbscan_n_noise": int(n_noise),
        "dbscan_silhouette": db_silhouette,
    }

    # ---- Step 4: Trust layer sanity check ---------------------------------
    section("STEP 4 - Explainability sanity check")
    explainer = build_shap_explainer(best_pipeline, background_sample)
    joblib.dump(True, config.MODEL_DIR / "_shap_ok.flag")
    print("SHAP explainer builds successfully on the saved model + background sample.")

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nSaved consolidated metrics -> {config.METRICS_PATH}")

    section("DONE")
    print(f"Total pipeline time: {time.time() - start:.1f}s")
    print(f"Artifacts in {config.MODEL_DIR}/ are ready for the dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
