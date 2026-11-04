
"""
Stakeholder-facing dashboard for the UK STATS19 Traffic Mortality project.

Run from the project root:
    streamlit run dashboard/app.py

The dashboard loads saved model artifacts. It does not retrain models.
"""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

from src.models.risk_model import score_segment
from src.models.severity_model import predict_single
from src.features.feature_engineering import add_derived_features


st.set_page_config(
    page_title="Traffic Mortality Reduction Dashboard",
    layout="wide",
)

st.title("Reducing Traffic Mortality Using Machine Learning")


# --------------------------------------------------
# LOAD DATA AND MODEL ARTIFACTS
# --------------------------------------------------

@st.cache_data(show_spinner="Loading district names...")
def load_districts():
    """Load local authority district names from the cleaned STATS19 CSV."""

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "stats19_clean.csv"
    )

    if not csv_path.is_file():
        st.warning(
            "District list unavailable: "
            "data/processed/stats19_clean.csv "
            "was not found in the deployed repository."
        )
        return ["Unknown"]

    try:
        # Read the CSV header first.
        header = pd.read_csv(csv_path, nrows=0)

        columns_by_lower = {
            str(column).strip().lower(): column
            for column in header.columns
        }

        # Support the possible district column names.
        district_column = next(
            (
                columns_by_lower[name]
                for name in (
                    "city",
                    "district",
                    "local_authority_district",
                )
                if name in columns_by_lower
            ),
            None,
        )

        if district_column is None:
            st.warning(
                "District list unavailable. The cleaned CSV "
                "does not contain a recognised district column. "
                f"Available columns: {', '.join(map(str, header.columns))}"
            )
            return ["Unknown"]

        district_series = pd.read_csv(
            csv_path,
            usecols=[district_column],
            low_memory=False,
        )[district_column]

        districts = (
            district_series
            .dropna()
            .astype(str)
            .str.strip()
        )

        districts = sorted(
            value
            for value in districts.unique().tolist()
            if value
            and value.lower() not in {"unknown", "nan", "none"}
        )

        if not districts:
            st.warning(
                "The district column was found, "
                "but it contains no usable district names."
            )
            return ["Unknown"]

        return districts

    except Exception as error:
        st.warning(
            f"Could not load district names from the cleaned CSV: {error}"
        )
        return ["Unknown"]


@st.cache_resource
def load_artifacts():

    required_files = [
        config.SEVERITY_MODEL_PATH,
        config.RISK_MODEL_PATH,
        config.RISK_LOOKUP_PATH,
        config.HOTSPOT_TABLE_PATH,
        config.SHAP_BACKGROUND_PATH,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        return None, missing

    severity_bundle = joblib.load(
        config.SEVERITY_MODEL_PATH
    )

    risk_model = joblib.load(
        config.RISK_MODEL_PATH
    )

    risk_lookup = pd.read_csv(
        config.RISK_LOOKUP_PATH
    )

    hotspots = pd.read_csv(
        config.HOTSPOT_TABLE_PATH
    )

    background = joblib.load(
        config.SHAP_BACKGROUND_PATH
    )

    return {
        "severity_pipeline": severity_bundle["pipeline"],
        "severity_model_name": severity_bundle["model_name"],
        "risk_model": risk_model,
        "risk_lookup": risk_lookup,
        "hotspots": hotspots,
        "background": background,
    }, []


@st.cache_resource
def load_explainers(_pipeline, _background):

    from src.explainability.explain import (
        build_shap_explainer,
        build_lime_explainer,
    )

    shap_explainer = build_shap_explainer(
        _pipeline,
        _background,
    )

    lime_explainer, lime_encoders = build_lime_explainer(
        _pipeline,
        _background,
    )

    return shap_explainer, lime_explainer, lime_encoders


artifacts, missing_files = load_artifacts()

if artifacts is None:

    st.error(
        "Required trained model artifacts were not found."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "The trained model artifacts must be made available "
        "to the application. They are not automatically created "
        "when Streamlit starts."
    )

    st.stop()


district_options = load_districts()


# --------------------------------------------------
# FOUR DASHBOARD TABS
# --------------------------------------------------

tab_severity, tab_risk, tab_hotspot, tab_performance = st.tabs(
    [
        "🚑 Severity Prediction",
        "⚠️ Risk Scoring",
        "🗺️ Hotspot Map",
        "📊 Model Performance",
    ]
)


# ==================================================
# TAB 1 - SEVERITY PREDICTION
# ==================================================

with tab_severity:

    st.header("Accident Severity Prediction")

    st.write(
        "Estimate the severity of an accident using its "
        "recorded road, weather, lighting and time conditions."
    )

    selected_model_display = (
        str(artifacts["severity_model_name"])
        .replace("_", " ")
        .title()
    )

    st.caption(
        f"Selected model: {selected_model_display}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        road_type = st.selectbox(
            "Road type",
            config.ROAD_TYPES,
        )

        junction_type = st.selectbox(
            "Junction type",
            config.JUNCTION_TYPES,
        )

        district = st.selectbox(
            "Local authority district",
            district_options,
        )

    with col2:

        weather = st.selectbox(
            "Weather",
            config.WEATHER_CONDITIONS,
        )

        light_condition = st.selectbox(
            "Light condition",
            config.LIGHT_CONDITIONS,
        )

        road_surface = st.selectbox(
            "Road surface",
            config.ROAD_SURFACE,
        )

    with col3:

        speed_limit = st.select_slider(
            "Speed limit (mph)",
            options=[20, 30, 40, 50, 60, 70],
            value=30,
        )

        hour = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=20,
        )

        num_vehicles = st.select_slider(
            "Number of vehicles involved",
            options=[1, 2, 3, 4, 5],
            value=2,
        )

    is_weekend = st.checkbox("Weekend")

    if st.button(
        "Predict accident severity",
        type="primary",
    ):

        # Saturday represents weekend; Monday represents weekday.
        selected_date = (
            pd.Timestamp("2024-01-06")
            if is_weekend
            else pd.Timestamp("2024-01-08")
        )

        raw_row = pd.DataFrame([{
            "date": selected_date,
            "hour": hour,
            "city": district,
            "road_type": road_type,
            "junction_type": junction_type,
            "weather": weather,
            "light_condition": light_condition,
            "road_surface": road_surface,
            "speed_limit": speed_limit,
            "num_vehicles_involved": num_vehicles,
        }])

        row = add_derived_features(raw_row)

        try:

            result = predict_single(
                artifacts["severity_pipeline"],
                row,
            )

            left, right = st.columns(2)

            with left:

                st.subheader("Prediction")

                st.metric(
                    "Highest probability class",
                    result["argmax_prediction"],
                )

                st.metric(
                    "Recall-oriented prediction",
                    result["recall_oriented_prediction"],
                )

                st.caption(
                    "The recall-oriented decision flags Fatal when "
                    f"P(Fatal) >= "
                    f"{config.FATAL_DECISION_THRESHOLD:.0%}."
                )

                st.write("Class probabilities")

                probabilities = pd.Series(
                    result["probabilities"]
                )

                st.bar_chart(probabilities)

            with right:

                st.subheader(
                    "Why did the model predict this?"
                )

                try:

                    (
                        shap_explainer,
                        lime_explainer,
                        lime_encoders,
                    ) = load_explainers(
                        artifacts["severity_pipeline"],
                        artifacts["background"],
                    )

                    from src.explainability.explain import (
                        local_explanation_shap,
                        local_explanation_lime,
                    )

                    shap_values = local_explanation_shap(
                        artifacts["severity_pipeline"],
                        shap_explainer,
                        row,
                        class_idx=2,
                    )

                    st.write(
                        "SHAP explanation for Fatal class"
                    )

                    st.bar_chart(shap_values)

                    st.caption(
                        "Positive values push the model output "
                        "toward Fatal; negative values push it away."
                    )

                    with st.expander("LIME explanation"):

                        lime_values = local_explanation_lime(
                            lime_explainer,
                            lime_encoders,
                            artifacts["severity_pipeline"],
                            row,
                            class_idx=2,
                        )

                        st.write(lime_values)

                except Exception as explain_error:

                    st.warning(
                        "Prediction succeeded, but the explanation "
                        "could not be generated."
                    )

                    st.caption(str(explain_error))

        except Exception as prediction_error:

            st.error("Severity prediction failed.")

            st.exception(prediction_error)


# ==================================================
# TAB 2 - RISK SCORING
# ==================================================

with tab_risk:

    st.header("Accident Risk Scoring")

    st.write(
        "Estimate the historical severity-weighted score "
        "associated with a road and weather condition combination."
    )

    st.warning(
        "This is a historical relative score, not a real-time "
        "accident probability. Traffic-volume exposure data is unavailable."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        risk_road = st.selectbox(
            "Road type",
            config.ROAD_TYPES,
            key="risk_road",
        )

        risk_junction = st.selectbox(
            "Junction type",
            config.JUNCTION_TYPES,
            key="risk_junction",
        )

    with c2:

        risk_weather = st.selectbox(
            "Weather",
            config.WEATHER_CONDITIONS,
            key="risk_weather",
        )

        risk_bucket = st.selectbox(
            "Time window",
            list(config.HOUR_BUCKETS.keys()),
            key="risk_bucket",
        )

    with c3:

        risk_weekend = st.checkbox(
            "Weekend",
            key="risk_weekend",
        )

    if st.button("Calculate historical risk score"):

        try:

            result = score_segment(
                artifacts["risk_model"],
                risk_road,
                risk_junction,
                risk_weather,
                risk_bucket,
                risk_weekend,
            )

            st.metric(
                "Relative risk score",
                result["risk_score"],
            )

            st.metric(
                "Risk category",
                result["risk_level"],
            )

            st.progress(
                min(
                    max(int(result["risk_score"]), 0),
                    100,
                )
            )

        except Exception as risk_error:

            st.error("Risk scoring failed.")

            st.exception(risk_error)

    st.divider()

    st.subheader(
        "Highest-scoring historical segments"
    )

    lookup = artifacts["risk_lookup"]

    if not lookup.empty:

        st.dataframe(
            lookup.sort_values(
                "risk_score",
                ascending=False,
            ).head(15),
            use_container_width=True,
        )

    else:

        st.info(
            "No historical risk segments are available."
        )


# ==================================================
# TAB 3 - HOTSPOT MAP
# ==================================================

with tab_hotspot:

    st.header("Geographic Accident Hotspots")

    st.write(
        "Geographic clusters of recorded accidents identified "
        "using DBSCAN."
    )

    hotspots = artifacts["hotspots"]

    if hotspots.empty:

        st.warning(
            "No hotspot records are available."
        )

    else:

        n_show = st.slider(
            "Number of hotspots to display",
            min_value=1,
            max_value=len(hotspots),
            value=min(15, len(hotspots)),
        )

        shown = hotspots.head(n_show)

        import folium

        map_center = [
            shown["centroid_lat"].mean(),
            shown["centroid_lon"].mean(),
        ]

        map_object = folium.Map(
            location=map_center,
            zoom_start=5,
            tiles="OpenStreetMap",
        )

        for _, hotspot in shown.iterrows():

            district_name = hotspot["dominant_district"]

            folium.CircleMarker(
                location=[
                    hotspot["centroid_lat"],
                    hotspot["centroid_lon"],
                ],
                radius=6 + min(
                    hotspot["severity_weighted_score"] / 200,
                    14,
                ),
                popup=(
                    f"Rank #{int(hotspot['rank'])}<br>"
                    f"District: {district_name}<br>"
                    f"Accidents in sampled cluster: "
                    f"{int(hotspot['n_accidents'])}<br>"
                    f"Fatal: {int(hotspot['fatal_count'])}<br>"
                    f"Serious: {int(hotspot['serious_count'])}<br>"
                    f"Weighted score: "
                    f"{hotspot['severity_weighted_score']}"
                ),
                color=(
                    "darkred"
                    if hotspot["rank"] <= 5
                    else "orange"
                ),
                fill=True,
                fill_opacity=0.7,
            ).add_to(map_object)

        st_folium(
            map_object,
            width=1100,
            height=520,
        )

        st.subheader("Ranked hotspot table")

        st.dataframe(
            shown[
                [
                    "rank",
                    "dominant_district",
                    "n_accidents",
                    "fatal_count",
                    "serious_count",
                    "severity_weighted_score",
                ]
            ],
            use_container_width=True,
        )

        st.caption(
            "Hotspots are based on the sampled DBSCAN records. "
            "Cluster counts do not represent every record in the dataset."
        )


# ==================================================
# TAB 4 - MODEL PERFORMANCE
# ==================================================

with tab_performance:

    st.header("Model Performance")

    if config.METRICS_PATH.exists():

        with open(
            config.METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            metrics = json.load(file)

        st.subheader(
            "Severity classifier benchmark"
        )

        rows = []

        for name, model_metrics in metrics["severity"].items():

            roc_auc = model_metrics.get(
                "roc_auc_ovr"
            )

            rows.append({
                "Model": name,
                "Fatal recall": model_metrics.get("fatal_recall"),
                "Fatal precision": model_metrics.get("fatal_precision"),
                "Macro F1": model_metrics.get("macro_f1"),
                "ROC-AUC OVR": roc_auc,
                "Accuracy": model_metrics.get("accuracy"),
            })

        metrics_df = pd.DataFrame(rows)

        st.dataframe(
            metrics_df.sort_values(
                "Fatal recall",
                ascending=False,
            ),
            use_container_width=True,
        )

        selected_model_display = (
            str(metrics["severity_best_model"])
            .replace("_", " ")
            .title()
        )

        st.success(
            f"Selected model: {selected_model_display}"
        )

        st.caption(
            "The selected classifier is ranked primarily by Fatal recall."
        )

        st.subheader(
            "Cross-validation Fatal recall"
        )

        cv_scores = metrics.get(
            "severity_cv_fatal_recall",
            [],
        )

        if cv_scores:

            st.bar_chart(
                pd.Series(
                    cv_scores,
                    index=[
                        f"Fold {i + 1}"
                        for i in range(len(cv_scores))
                    ],
                )
            )

        st.subheader(
            "Hotspot clustering results"
        )

        hotspot_metrics = metrics.get("hotspot")

        if hotspot_metrics:

            st.json(hotspot_metrics)

    else:

        st.warning(
            "metrics.json was not found. "
            "Run the training pipeline first."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research prototype using real UK STATS19 "
    "police-reported personal-injury collision records, 2021–2025. "
    "Model outputs are analytical estimates, not causal conclusions "
    "or official road-safety risk rates."
)
