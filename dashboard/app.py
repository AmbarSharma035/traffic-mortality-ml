"""
Stakeholder-facing dashboard (slide 12 / SRS section 3.9).

    streamlit run dashboard/app.py

Loads the artifacts main.py already trained and saved into models/ -
it does NOT retrain anything, so it starts in a couple of seconds. Run
`python main.py` at least once before this.

Four tabs, one per audience-facing concern:
  1. Severity Prediction  - score one hypothetical accident + SHAP/LIME "why"
  2. Risk Scoring         - score a road/time/weather combination
  3. Hotspot Map          - Folium map of ranked black-spots
  4. Model Performance    - the benchmark table used to pick the winning model
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.explainability.explain import (
    build_lime_explainer, build_shap_explainer, global_feature_importance,
    local_explanation_lime, local_explanation_shap,
)
from src.models.risk_model import score_segment
from src.models.severity_model import predict_single

st.set_page_config(page_title="Traffic Mortality Reduction Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    missing = [p for p in [config.SEVERITY_MODEL_PATH, config.RISK_MODEL_PATH,
                            config.HOTSPOT_TABLE_PATH, config.SHAP_BACKGROUND_PATH]
               if not p.exists()]
    if missing:
        return None
    severity_bundle = joblib.load(config.SEVERITY_MODEL_PATH)
    risk_model = joblib.load(config.RISK_MODEL_PATH)
    hotspots = pd.read_csv(config.HOTSPOT_TABLE_PATH)
    background = joblib.load(config.SHAP_BACKGROUND_PATH)
    return {
        "severity_pipeline": severity_bundle["pipeline"],
        "severity_model_name": severity_bundle["model_name"],
        "risk_model": risk_model,
        "hotspots": hotspots,
        "background": background,
    }


@st.cache_resource
def load_explainers(_pipeline, _background):
    shap_explainer = build_shap_explainer(_pipeline, _background)
    lime_explainer, lime_encoders = build_lime_explainer(_pipeline, _background)
    return shap_explainer, lime_explainer, lime_encoders


artifacts = load_artifacts()

st.title("Reducing Traffic Mortality Using Machine Learning")
st.caption("Severity Prediction · Risk Prediction · Hotspot Detection — one pipeline, three coordinated outcomes.")

if artifacts is None:
    st.error(
        "No trained artifacts found in `models/`. Run `python main.py` from the "
        "project root first, then restart this dashboard."
    )
    st.stop()

tab_severity, tab_risk, tab_hotspot, tab_perf = st.tabs(
    ["🚑 Severity Prediction", "⚠️ Risk Scoring", "🗺️ Hotspot Map", "📊 Model Performance"]
)

# ---------------------------------------------------------------- Severity
with tab_severity:
    st.subheader("How severe is a crash likely to be under these conditions?")
    st.caption(f"Model in use: **{artifacts['severity_model_name']}** "
               f"(picked automatically for the best Fatal-class recall — see the Model Performance tab)")

    col1, col2, col3 = st.columns(3)
    with col1:
        road_type = st.selectbox("Road type", config.ROAD_TYPES)
        junction_type = st.selectbox("Junction type", config.JUNCTION_TYPES)
        city = st.selectbox("City", list(config.CITY_CENTERS.keys()))
    with col2:
        weather = st.selectbox("Weather", config.WEATHER_CONDITIONS)
        light_condition = st.selectbox("Light condition", config.LIGHT_CONDITIONS)
        road_surface = st.selectbox("Road surface", config.ROAD_SURFACE)
    with col3:
        speed_limit = st.select_slider("Speed limit (km/h)", options=[30, 40, 50, 60, 80, 100], value=60)
        hour = st.slider("Hour of day", 0, 23, 20)
        num_vehicles = st.select_slider("Vehicles involved", options=[1, 2, 3, 4], value=1)

    is_weekend = st.checkbox("Weekend")

    if st.button("Predict severity", type="primary"):
        from src.features.feature_engineering import add_derived_features

        raw_row = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-06" if is_weekend else "2024-01-01"),  # Sat vs Mon
            "hour": hour, "city": city, "road_type": road_type, "junction_type": junction_type,
            "weather": weather, "light_condition": light_condition, "road_surface": road_surface,
            "speed_limit": speed_limit, "num_vehicles_involved": num_vehicles,
        }])
        row = add_derived_features(raw_row)

        result = predict_single(artifacts["severity_pipeline"], row)

        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Model's top prediction (argmax)", result["argmax_prediction"])
            st.metric("Recall-oriented prediction", result["recall_oriented_prediction"],
                       help=f"Flags Fatal whenever P(Fatal) >= {config.FATAL_DECISION_THRESHOLD:.0%}, "
                            "since a missed fatal case is far costlier than a false alarm.")
            st.write("Class probabilities:")
            st.bar_chart(pd.Series(result["probabilities"]))

        with r2:
            st.write("**Why this prediction? (SHAP)**")
            shap_explainer, lime_explainer, lime_encoders = load_explainers(
                artifacts["severity_pipeline"], artifacts["background"]
            )
            shap_contrib = local_explanation_shap(artifacts["severity_pipeline"], shap_explainer, row)
            st.bar_chart(shap_contrib)
            st.caption("Positive = pushes toward Fatal. Negative = pushes away from Fatal.")

            with st.expander("Cross-check with LIME"):
                lime_contrib = local_explanation_lime(lime_explainer, lime_encoders,
                                                       artifacts["severity_pipeline"], row)
                st.bar_chart(pd.Series(lime_contrib))

# -------------------------------------------------------------------- Risk
with tab_risk:
    st.subheader("How risky are these road/time/weather conditions, before anything happens?")
    c1, c2, c3 = st.columns(3)
    with c1:
        r_road = st.selectbox("Road type ", config.ROAD_TYPES, key="risk_road")
        r_junction = st.selectbox("Junction type ", config.JUNCTION_TYPES, key="risk_junction")
    with c2:
        r_weather = st.selectbox("Weather ", config.WEATHER_CONDITIONS, key="risk_weather")
        r_bucket = st.selectbox("Time window", list(config.HOUR_BUCKETS.keys()), key="risk_bucket")
    with c3:
        r_weekend = st.checkbox("Weekend ", key="risk_weekend")

    if st.button("Score this segment"):
        result = score_segment(artifacts["risk_model"], r_road, r_junction, r_weather, r_bucket, r_weekend)
        st.metric("Risk score (0-100)", result["risk_score"])
        st.metric("Risk level", result["risk_level"])
        st.progress(min(int(result["risk_score"]), 100))

    st.divider()
    st.write("**Highest-risk segments currently on record:**")
    lookup = pd.read_csv(config.RISK_LOOKUP_PATH) if config.RISK_LOOKUP_PATH.exists() else None
    if lookup is not None:
        st.dataframe(
            lookup.sort_values("risk_score", ascending=False)
            .head(15)[["road_type", "junction_type", "weather", "hour_bucket", "is_weekend",
                       "n_accidents", "fatal_count", "risk_score", "risk_level"]],
            use_container_width=True,
        )

# ----------------------------------------------------------------- Hotspot
with tab_hotspot:
    st.subheader("Where do crashes keep concentrating?")
    hotspots = artifacts["hotspots"]

    n_show = st.slider("Number of black-spots to show", 5, len(hotspots), min(15, len(hotspots)))
    shown = hotspots.head(n_show)

    import folium

    m = folium.Map(location=[shown["centroid_lat"].mean(), shown["centroid_lon"].mean()], zoom_start=5)
    for _, row in shown.iterrows():
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=6 + min(row["severity_weighted_score"] / 200, 14),
            popup=(f"Rank #{int(row['rank'])} — {row['dominant_city']}<br>"
                   f"{int(row['n_accidents'])} accidents, {int(row['fatal_count'])} fatal<br>"
                   f"Weighted score: {row['severity_weighted_score']}"),
            color="darkred" if row["rank"] <= 5 else "orange",
            fill=True, fill_opacity=0.7,
        ).add_to(m)

    st_folium(m, width=1100, height=520)

    st.write("**Ranked black-spot table:**")
    st.dataframe(
        shown[["rank", "dominant_city", "n_accidents", "fatal_count", "serious_count",
               "severity_weighted_score"]],
        use_container_width=True,
    )

# ------------------------------------------------------------ Model perf
with tab_perf:
    st.subheader("Severity model benchmark (why this model was picked)")
    import json
    if config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text())
        rows = []
        for name, m in metrics["severity"].items():
            rows.append({
                "model": name,
                "fatal_recall": round(m["fatal_recall"], 3),
                "fatal_precision": round(m["fatal_precision"], 3),
                "macro_f1": round(m["macro_f1"], 3),
                "roc_auc_ovr": round(m["roc_auc_ovr"], 3) if m["roc_auc_ovr"] else None,
                "accuracy": round(m["accuracy"], 3),
            })
        df_metrics = pd.DataFrame(rows).sort_values("fatal_recall", ascending=False)
        st.dataframe(df_metrics, use_container_width=True)
        st.caption(f"Winning model: **{metrics['severity_best_model']}** "
                   f"(highest Fatal-class recall — the project's primary metric, not accuracy).")

        st.write("5-fold cross-validated Fatal-class recall for the winning model:")
        st.bar_chart(pd.Series(metrics["severity_cv_fatal_recall"],
                                index=[f"fold {i+1}" for i in range(len(metrics["severity_cv_fatal_recall"]))]))

        st.write("Hotspot clustering quality:")
        st.json(metrics["hotspot"])
    else:
        st.warning("No metrics.json found — run `python main.py` first.")
