"""
Traffic Mortality ML — Interactive Streamlit Dashboard

Multi-page dashboard with:
1. Overview - KPIs and summary
2. Severity Prediction - User input prediction
3. Risk Map - Geographic risk visualization
4. Hotspot Analysis - K-Means, DBSCAN, KDE
5. Temporal Analysis - Time-based patterns
6. Explainability - SHAP & LIME
7. Model Comparison - Metrics and confusion matrices
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR, MAPS_DIR,
    REPORTS_DIR, setup_logging, get_config_section, load_config
)

logger = setup_logging("dashboard")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Mortality ML Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_unified_data():
    """Load the unified accidents dataset."""
    path = DATA_PROCESSED / "unified_accidents.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


@st.cache_data
def load_risk_data():
    """Load accidents with risk scores."""
    path = DATA_PROCESSED / "accidents_with_risk.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    return df


@st.cache_data
def load_model_comparison():
    """Load model comparison results."""
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_temporal_stats():
    """Load temporal analysis results."""
    path = DATA_PROCESSED / "temporal_stats.json"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_hotspot_labels():
    """Load hotspot cluster labels."""
    path = DATA_PROCESSED / "hotspot_labels.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and feature names."""
    model_path = MODELS_DIR / "selected_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    features_path = MODELS_DIR / "feature_names.json"

    if not all(p.exists() for p in [model_path, scaler_path, features_path]):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


@st.cache_data
def load_evaluation_report():
    """Load detailed evaluation report."""
    path = REPORTS_DIR / "evaluation_report.txt"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚗 Traffic Mortality ML")
st.sidebar.markdown("---")

pages = {
    "📊 Overview": "overview",
    "🔮 Severity Prediction": "prediction",
    "🗺️ Risk Map": "risk_map",
    "📍 Hotspot Analysis": "hotspots",
    "⏰ Temporal Analysis": "temporal",
    "🔍 Explainability": "explainability",
    "📈 Model Comparison": "model_comparison"
}

selected_page = st.sidebar.radio("Navigate", list(pages.keys()))
page_key = pages[selected_page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** Reducing Traffic Mortality\n\n"
    "**Framework:** Accident Severity Classification\n"
    "& Risk Hotspot Identification"
)


# ===================================================================
# PAGE: Overview
# ===================================================================
def page_overview():
    st.title("📊 Dashboard Overview")
    st.markdown("### Reducing Traffic Mortality Using Machine Learning")

    df = load_unified_data()
    if df is None:
        st.error("⚠️ No data available. Please run the preprocessing pipeline first.")
        st.code("python -m src.data.download\npython -m src.data.preprocess", language="bash")
        return

    comparison = load_model_comparison()

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    fatal_count = len(df[df['severity'] == 'Fatal']) if 'severity' in df.columns else 0
    serious_count = len(df[df['severity'] == 'Serious']) if 'severity' in df.columns else 0
    fatality_pct = (fatal_count / total * 100) if total > 0 else 0

    col1.metric("Total Accidents", f"{total:,}")
    col2.metric("Fatal Accidents", f"{fatal_count:,}")
    col3.metric("Serious Accidents", f"{serious_count:,}")
    col4.metric("Fatality Rate", f"{fatality_pct:.2f}%")

    st.markdown("---")

    # --- Model info ---
    if comparison is not None and not comparison.empty:
        best_model = comparison.iloc[0]
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🏆 Selected Model")
            st.success(f"**{best_model.get('Model', 'N/A')}**")
            st.write(f"- Fatal Recall: **{best_model.get('Fatal_Recall', 0):.4f}**")
            st.write(f"- Accuracy: **{best_model.get('Accuracy', 0):.4f}**")
            st.write(f"- Macro F1: **{best_model.get('Macro_F1', 0):.4f}**")

        with col_b:
            st.subheader("📊 Severity Distribution")
            if 'severity' in df.columns:
                sev_counts = df['severity'].value_counts()
                fig = px.pie(
                    values=sev_counts.values,
                    names=sev_counts.index,
                    color=sev_counts.index,
                    color_discrete_map={
                        'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                    },
                    hole=0.4
                )
                fig.update_layout(height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run model training to see results: `python -m src.models.train`")

    # --- Country breakdown ---
    st.markdown("---")
    if 'country' in df.columns:
        st.subheader("🌍 Data by Country")
        country_sev = df.groupby(['country', 'severity']).size().reset_index(name='count')
        fig = px.bar(
            country_sev, x='country', y='count', color='severity',
            barmode='group',
            color_discrete_map={
                'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Quick temporal snapshot ---
    if 'hour' in df.columns:
        st.subheader("⏰ Accidents by Hour of Day")
        hourly = df['hour'].value_counts().sort_index()
        fig = px.bar(x=hourly.index, y=hourly.values,
                     labels={'x': 'Hour', 'y': 'Accidents'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: Severity Prediction
# ===================================================================
def page_prediction():
    st.title("🔮 Accident Severity Prediction")
    st.markdown("Enter accident conditions to predict severity and risk score.")

    model, scaler, feature_names = load_model_artifacts()
    if model is None:
        st.error("⚠️ No trained model found. Run: `python -m src.models.train`")
        return

    severity_labels = ['Minor', 'Serious', 'Fatal']

    with st.form("prediction_form"):
        st.subheader("Input Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Temporal**")
            hour = st.slider("Hour of Day", 0, 23, 14)
            day_of_week = st.selectbox("Day of Week",
                                       options=list(range(7)),
                                       format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
            month = st.slider("Month", 1, 12, 6)

        with col2:
            st.markdown("**Weather & Environment**")
            temperature = st.slider("Temperature (°F)", -20, 120, 65)
            visibility = st.slider("Visibility (miles)", 0.0, 10.0, 5.0, step=0.5)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
            wind_speed = st.slider("Wind Speed (mph)", 0.0, 80.0, 10.0, step=1.0)
            weather_severity = st.selectbox("Weather Severity",
                                            [0, 1, 2, 3],
                                            format_func=lambda x: ['Clear', 'Light Rain/Cloud',
                                                                     'Rain/Fog', 'Severe Storm'][x])

        with col3:
            st.markdown("**Road Conditions**")
            is_night = st.checkbox("Night Time", value=False)
            is_weekend = st.checkbox("Weekend", value=False)
            rush_hour = st.checkbox("Rush Hour", value=False)
            has_junction = st.checkbox("Near Junction", value=False)
            has_crossing = st.checkbox("Near Crossing", value=False)
            has_traffic_signal = st.checkbox("Traffic Signal Present", value=False)

        submitted = st.form_submit_button("🔮 Predict Severity", use_container_width=True)

    if submitted:
        # Build feature dict
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': int(is_weekend),
            'is_night': int(is_night),
            'rush_hour': int(rush_hour),
            'temperature': temperature,
            'visibility': visibility,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'weather_severity': weather_severity,
            'has_junction': int(has_junction),
            'has_crossing': int(has_crossing),
            'has_traffic_signal': int(has_traffic_signal),
            'has_road_feature': int(has_junction or has_crossing or has_traffic_signal),
        }

        # Derived features
        if hour in [7, 8, 9, 16, 17, 18]:
            features['rush_hour'] = 1

        time_map = {range(6, 12): 0, range(12, 17): 1, range(17, 21): 2}
        tod = 3  # night
        for r, v in time_map.items():
            if hour in r:
                tod = v
                break
        features['time_of_day'] = tod

        season_map = {12: 3, 1: 3, 2: 3, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2}
        features['season'] = season_map.get(month, 0)

        if visibility > 5:
            features['visibility_category'] = 0
        elif visibility >= 2:
            features['visibility_category'] = 1
        else:
            features['visibility_category'] = 2

        if temperature < 32:
            features['temperature_category'] = 0
        elif temperature < 60:
            features['temperature_category'] = 1
        elif temperature < 80:
            features['temperature_category'] = 2
        else:
            features['temperature_category'] = 3

        # Create feature vector in correct order
        try:
            feature_vector = pd.DataFrame([{fn: features.get(fn, 0) for fn in feature_names}])
            scaled = scaler.transform(feature_vector)

            prediction = model.predict(scaled)[0]
            probabilities = model.predict_proba(scaled)[0]

            pred_label = severity_labels[int(prediction)]
            confidence = float(probabilities[int(prediction)])

            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)

            color_map = {'Minor': '🟢', 'Serious': '🟡', 'Fatal': '🔴'}

            res_col1.metric("Predicted Severity",
                            f"{color_map.get(pred_label, '')} {pred_label}")
            res_col2.metric("Confidence", f"{confidence:.1%}")

            # Risk score (simplified)
            risk_cfg = get_config_section('risk')
            weights = risk_cfg['weights']
            sev_score = float(probabilities[2]) * 100  # Fatal probability
            temp_score = 70 if rush_hour else 30
            env_score = weather_severity * 25 + (25 if is_night else 0)
            env_score = min(env_score, 100)

            risk_score = (
                weights['severity_probability'] * sev_score +
                weights['temporal_risk'] * temp_score +
                weights['environmental_risk'] * env_score +
                weights['geographic_density'] * 50  # neutral without KDE
            )
            risk_score = np.clip(risk_score, 0, 100)

            cats = risk_cfg['categories']
            if risk_score <= cats['low'][1]:
                risk_cat = 'Low'
            elif risk_score <= cats['moderate'][1]:
                risk_cat = 'Moderate'
            elif risk_score <= cats['high'][1]:
                risk_cat = 'High'
            else:
                risk_cat = 'Critical'

            res_col3.metric("Risk Score", f"{risk_score:.1f} ({risk_cat})")

            # Probability distribution
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                'Severity': severity_labels,
                'Probability': probabilities
            })
            fig = px.bar(prob_df, x='Severity', y='Probability',
                         color='Severity',
                         color_discrete_map={
                             'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                         })
            fig.update_layout(height=300, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            logger.error(f"Prediction failed: {e}", exc_info=True)


# ===================================================================
# PAGE: Risk Map
# ===================================================================
def page_risk_map():
    st.title("🗺️ Risk Map")

    risk_df = load_risk_data()
    df = load_unified_data()

    data = risk_df if risk_df is not None else df
    if data is None:
        st.error("⚠️ No data available.")
        return

    # Filters
    st.sidebar.markdown("### Map Filters")
    if 'severity' in data.columns:
        severities = st.sidebar.multiselect(
            "Severity", data['severity'].unique().tolist(),
            default=data['severity'].unique().tolist()
        )
        data = data[data['severity'].isin(severities)]

    if 'country' in data.columns:
        countries = st.sidebar.multiselect(
            "Country", data['country'].unique().tolist(),
            default=data['country'].unique().tolist()
        )
        data = data[data['country'].isin(countries)]

    # Limit points for performance
    max_points = st.sidebar.slider("Max points on map", 100, 10000, 2000, step=100)
    sample = data.dropna(subset=['latitude', 'longitude']).head(max_points)

    if sample.empty:
        st.warning("No data with valid coordinates.")
        return

    # Create map
    center_lat = sample['latitude'].mean()
    center_lon = sample['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    # Color by risk or severity
    color_col = 'risk_category' if 'risk_category' in sample.columns else 'severity'
    color_map_risk = {
        'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Critical': 'darkred',
        'Minor': 'green', 'Serious': 'orange', 'Fatal': 'red'
    }

    from folium.plugins import MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in sample.iterrows():
        cat = str(row.get(color_col, 'Minor'))
        color = color_map_risk.get(cat, 'blue')

        popup_text = f"Severity: {row.get('severity', 'N/A')}"
        if 'risk_score' in row:
            popup_text += f"<br>Risk: {row['risk_score']:.1f}"
        if 'risk_category' in row:
            popup_text += f" ({row['risk_category']})"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(marker_cluster)

    st_folium(m, width=None, height=600)

    # Legend
    st.markdown("""
    **Legend:** 🟢 Low/Minor | 🟠 Moderate/Serious | 🔴 High/Fatal | ⚫ Critical
    """)

    if 'risk_category' in data.columns:
        st.subheader("Risk Distribution")
        risk_counts = data['risk_category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     color=risk_counts.index,
                     color_discrete_map={
                         'Low': '#2ecc71', 'Moderate': '#f39c12',
                         'High': '#e74c3c', 'Critical': '#8b0000'
                     })
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: Hotspot Analysis
# ===================================================================
def page_hotspots():
    st.title("📍 Hotspot Analysis")

    hotspot_df = load_hotspot_labels()
    df = load_unified_data()

    if df is None:
        st.error("⚠️ No data available.")
        return

    tab_kmeans, tab_dbscan, tab_kde, tab_ranked = st.tabs(
        ["K-Means", "DBSCAN", "KDE Heatmap", "Ranked Hotspots"]
    )

    coords_path = DATA_PROCESSED / "coordinates.csv"
    if coords_path.exists():
        coords_df = pd.read_csv(coords_path)
    else:
        coords_df = df[['latitude', 'longitude']].dropna()

    with tab_kmeans:
        st.subheader("K-Means Clustering")
        if hotspot_df is not None and 'kmeans_label' in hotspot_df.columns:
            merged = pd.concat([coords_df.reset_index(drop=True),
                                hotspot_df[['kmeans_label']].reset_index(drop=True)], axis=1)
            merged = merged.dropna(subset=['latitude', 'longitude'])
            sample = merged.sample(min(3000, len(merged)), random_state=42)

            fig = px.scatter_mapbox(
                sample, lat='latitude', lon='longitude',
                color='kmeans_label',
                mapbox_style='open-street-map',
                zoom=3, height=500,
                title="K-Means Accident Clusters"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(f"**Clusters:** {merged['kmeans_label'].nunique()}")
            st.dataframe(merged['kmeans_label'].value_counts().head(10).reset_index(
                ).rename(columns={'index': 'Cluster', 'kmeans_label': 'Count'}))
        else:
            st.info("Run hotspot detection: `python -m src.hotspots.run`")

    with tab_dbscan:
        st.subheader("DBSCAN Clustering")
        if hotspot_df is not None and 'dbscan_label' in hotspot_df.columns:
            merged = pd.concat([coords_df.reset_index(drop=True),
                                hotspot_df[['dbscan_label']].reset_index(drop=True)], axis=1)
            merged = merged.dropna(subset=['latitude', 'longitude'])

            # Separate noise from clusters
            noise = merged[merged['dbscan_label'] == -1]
            clusters = merged[merged['dbscan_label'] >= 0]

            st.write(f"**Clusters:** {clusters['dbscan_label'].nunique()} | "
                     f"**Noise points:** {len(noise):,}")

            sample = clusters.sample(min(3000, len(clusters)), random_state=42) if len(clusters) > 0 else clusters
            if not sample.empty:
                fig = px.scatter_mapbox(
                    sample, lat='latitude', lon='longitude',
                    color='dbscan_label',
                    mapbox_style='open-street-map',
                    zoom=3, height=500,
                    title="DBSCAN Dense Regions"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run hotspot detection: `python -m src.hotspots.run`")

    with tab_kde:
        st.subheader("Kernel Density Estimation Heatmap")

        valid_coords = coords_df.dropna(subset=['latitude', 'longitude'])
        sample_kde = valid_coords.sample(min(5000, len(valid_coords)), random_state=42)

        m = folium.Map(
            location=[sample_kde['latitude'].mean(), sample_kde['longitude'].mean()],
            zoom_start=4
        )
        from folium.plugins import HeatMap
        heat_data = sample_kde[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

        st_folium(m, width=None, height=500)

    with tab_ranked:
        st.subheader("Top Accident Hotspot Locations")
        if hotspot_df is not None and 'kmeans_label' in hotspot_df.columns:
            merged = pd.concat([coords_df.reset_index(drop=True),
                                hotspot_df[['kmeans_label']].reset_index(drop=True)], axis=1)
            cluster_stats = merged.groupby('kmeans_label').agg(
                count=('latitude', 'size'),
                avg_lat=('latitude', 'mean'),
                avg_lon=('longitude', 'mean')
            ).sort_values('count', ascending=False).reset_index()
            cluster_stats.columns = ['Cluster', 'Accident Count', 'Latitude', 'Longitude']
            st.dataframe(cluster_stats.head(20), use_container_width=True)
        else:
            st.info("Run hotspot detection first.")


# ===================================================================
# PAGE: Temporal Analysis
# ===================================================================
def page_temporal():
    st.title("⏰ Temporal Analysis")

    df = load_unified_data()
    temporal = load_temporal_stats()

    if df is None:
        st.error("⚠️ No data available.")
        return

    tab_hourly, tab_daily, tab_monthly, tab_patterns = st.tabs(
        ["Hourly", "Daily", "Monthly", "Patterns"]
    )

    with tab_hourly:
        st.subheader("Accidents by Hour of Day")
        if 'hour' in df.columns:
            hourly = df.groupby(['hour', 'severity']).size().reset_index(name='count')
            fig = px.bar(hourly, x='hour', y='count', color='severity',
                         barmode='stack',
                         color_discrete_map={
                             'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                         },
                         labels={'hour': 'Hour of Day', 'count': 'Number of Accidents'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            if temporal and 'high_risk_hours' in temporal:
                st.warning(f"⚠️ **High-risk hours:** {temporal['high_risk_hours']}")

    with tab_daily:
        st.subheader("Accidents by Day of Week")
        if 'day_of_week' in df.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily = df.groupby(['day_of_week', 'severity']).size().reset_index(name='count')
            daily['day_name'] = daily['day_of_week'].map(lambda x: day_names[x] if x < 7 else 'Unk')
            fig = px.bar(daily, x='day_name', y='count', color='severity',
                         barmode='stack',
                         color_discrete_map={
                             'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                         })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab_monthly:
        st.subheader("Accidents by Month")
        if 'month' in df.columns:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly = df.groupby(['month', 'severity']).size().reset_index(name='count')
            monthly['month_name'] = monthly['month'].map(
                lambda x: month_names[x-1] if 1 <= x <= 12 else 'Unk')
            fig = px.bar(monthly, x='month_name', y='count', color='severity',
                         barmode='stack',
                         color_discrete_map={
                             'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                         })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab_patterns:
        st.subheader("Risk Patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Weekend vs Weekday")
            if 'is_weekend' in df.columns:
                wk = df.groupby(['is_weekend', 'severity']).size().reset_index(name='count')
                wk['type'] = wk['is_weekend'].map({0: 'Weekday', 1: 'Weekend',
                                                    False: 'Weekday', True: 'Weekend'})
                fig = px.bar(wk, x='type', y='count', color='severity',
                             barmode='group',
                             color_discrete_map={
                                 'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                             })
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Day vs Night")
            if 'is_night' in df.columns:
                dn = df.groupby(['is_night', 'severity']).size().reset_index(name='count')
                dn['type'] = dn['is_night'].map({0: 'Day', 1: 'Night',
                                                  False: 'Day', True: 'Night'})
                fig = px.bar(dn, x='type', y='count', color='severity',
                             barmode='group',
                             color_discrete_map={
                                 'Minor': '#2ecc71', 'Serious': '#f39c12', 'Fatal': '#e74c3c'
                             })
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # High-risk time windows
        if temporal:
            st.markdown("---")
            st.subheader("🚨 High-Risk Time Windows")
            if 'high_risk_hours' in temporal:
                st.write(f"**Peak Hours:** {temporal['high_risk_hours']}")
            if 'high_risk_days' in temporal:
                st.write(f"**Peak Days:** {temporal['high_risk_days']}")
            if 'high_risk_months' in temporal:
                st.write(f"**Peak Months:** {temporal['high_risk_months']}")


# ===================================================================
# PAGE: Explainability
# ===================================================================
def page_explainability():
    st.title("🔍 Model Explainability")

    tab_shap, tab_lime, tab_importance = st.tabs(
        ["SHAP Analysis", "LIME Analysis", "Feature Importance"]
    )

    with tab_shap:
        st.subheader("SHAP — Global Feature Importance")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** uses game theory to explain how
        each feature contributes to the model's prediction.
        """)

        # Show saved SHAP plots
        shap_summary = FIGURES_DIR / "shap_summary.png"
        shap_importance = FIGURES_DIR / "shap_feature_importance.png"

        if shap_summary.exists():
            st.image(str(shap_summary), caption="SHAP Summary Plot", use_container_width=True)
        if shap_importance.exists():
            st.image(str(shap_importance), caption="SHAP Feature Importance",
                     use_container_width=True)

        # Individual explanations
        st.markdown("---")
        st.subheader("Individual Prediction Explanations")
        for i in range(3):
            path = FIGURES_DIR / f"shap_individual_{i}.png"
            if path.exists():
                st.image(str(path), caption=f"SHAP Explanation — Sample {i+1}",
                         use_container_width=True)

        if not shap_summary.exists() and not shap_importance.exists():
            st.info("Run explainability: `python -m src.explainability.explain`")

    with tab_lime:
        st.subheader("LIME — Local Interpretable Explanations")
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)** creates a
        simple local surrogate model to explain individual predictions.
        """)

        for i in range(3):
            path = FIGURES_DIR / f"lime_individual_{i}.png"
            if path.exists():
                st.image(str(path), caption=f"LIME Explanation — Sample {i+1}",
                         use_container_width=True)

        if not (FIGURES_DIR / "lime_individual_0.png").exists():
            st.info("Run explainability: `python -m src.explainability.explain`")

    with tab_importance:
        st.subheader("Model Feature Importance")

        # Try to load model and show built-in feature importance
        model, scaler, feature_names = load_model_artifacts()
        if model is not None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(20)

            fig = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h', title="Top 20 Feature Importances")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance available for tree-based models.")


# ===================================================================
# PAGE: Model Comparison
# ===================================================================
def page_model_comparison():
    st.title("📈 Model Comparison")

    comparison = load_model_comparison()
    if comparison is None:
        st.error("⚠️ No evaluation results. Run: `python -m src.models.evaluate`")
        return

    # Highlight best model
    st.subheader("📊 Comparison Table")
    st.dataframe(
        comparison.style.highlight_max(
            subset=['Fatal_Recall', 'Accuracy', 'Macro_F1'],
            color='lightgreen'
        ),
        use_container_width=True
    )

    st.markdown(f"**Primary selection metric:** Fatal-Class Recall")
    st.markdown(f"**Best model:** {comparison.iloc[0]['Model']}")

    st.markdown("---")

    # Fatal Recall comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fatal-Class Recall")
        fig = px.bar(
            comparison.sort_values('Fatal_Recall'),
            x='Fatal_Recall', y='Model',
            orientation='h',
            color='Fatal_Recall',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Overall Accuracy")
        fig = px.bar(
            comparison.sort_values('Accuracy'),
            x='Accuracy', y='Model',
            orientation='h',
            color='Accuracy',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # All metrics comparison
    st.subheader("All Metrics Comparison")
    metric_cols = [c for c in comparison.columns if c != 'Model']
    melted = comparison.melt(id_vars='Model', value_vars=metric_cols,
                             var_name='Metric', value_name='Score')
    fig = px.bar(melted, x='Metric', y='Score', color='Model',
                 barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_path = FIGURES_DIR / "confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Run evaluation to generate confusion matrices.")

    # Full report
    report = load_evaluation_report()
    if report:
        with st.expander("📄 Full Evaluation Report"):
            st.text(report)


# ===================================================================
# Main router
# ===================================================================
PAGE_MAP = {
    "overview": page_overview,
    "prediction": page_prediction,
    "risk_map": page_risk_map,
    "hotspots": page_hotspots,
    "temporal": page_temporal,
    "explainability": page_explainability,
    "model_comparison": page_model_comparison,
}

PAGE_MAP[page_key]()
