
"""
Central configuration for the Traffic Mortality ML project.

Adapted for real UK STATS19 accident data.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# DATA PATHS
# --------------------------------------------------

DATA_RAW_PATH = Path(
    os.getenv(
        "DATA_RAW_PATH",
        BASE_DIR / "data/processed/stats19_prepared.csv"
    )
)

DATA_PROCESSED_PATH = Path(
    os.getenv(
        "DATA_PROCESSED_PATH",
        BASE_DIR / "data/processed/stats19_clean.csv"
    )
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

METRICS_PATH = MODEL_DIR / "metrics.json"

SEVERITY_MODEL_PATH = MODEL_DIR / "severity_model.joblib"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
RISK_LOOKUP_PATH = MODEL_DIR / "risk_lookup.csv"
HOTSPOT_TABLE_PATH = MODEL_DIR / "hotspots_ranked.csv"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.joblib"


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


# --------------------------------------------------
# STATS19 DATA CATEGORIES
# --------------------------------------------------

SEVERITY_CLASSES = ["Minor", "Serious", "Fatal"]

ROAD_TYPES = [
    "Roundabout",
    "One way street",
    "Dual carriageway",
    "Single carriageway",
    "Slip road",
    "Unknown",
]

JUNCTION_TYPES = [
    "Not at or within 20 metres of junction",
    "T or staggered junction",
    "Crossroads",
    "Junction with more than four arms",
    "Private drive or entrance",
    "Other junction",
    "Unknown",
]

WEATHER_CONDITIONS = [
    "Fine no high winds",
    "Raining no high winds",
    "Snowing no high winds",
    "Fine high winds",
    "Raining high winds",
    "Snowing high winds",
    "Fog or mist",
    "Other",
    "Unknown",
]

LIGHT_CONDITIONS = [
    "Daylight",
    "Darkness - lights lit",
    "Darkness - lights unlit",
    "Darkness - no lighting",
    "Darkness - lighting unknown",
    "Unknown",
]

ROAD_SURFACE = [
    "Dry",
    "Wet or damp",
    "Snow",
    "Frost or ice",
    "Flood",
    "Oil or diesel",
    "Mud",
    "Unknown",
]


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

# Illustrative UK commuting hours.
# These are assumptions for feature engineering,
# not proven accident-risk thresholds.

RUSH_HOURS = (
    set(range(7, 10)) |
    set(range(16, 19))
)

# Heuristic weather severity weights.
# These are not probabilities or causal measurements.

WEATHER_SEVERITY_WEIGHTS = {
    "Fine no high winds": 0,
    "Fine high winds": 1,
    "Raining no high winds": 2,
    "Raining high winds": 3,
    "Snowing no high winds": 3,
    "Snowing high winds": 4,
    "Fog or mist": 3,
    "Other": 1,
    "Unknown": 1,
}

LIGHT_SEVERITY_WEIGHTS = {
    "Daylight": 0,
    "Darkness - lights lit": 1,
    "Darkness - lights unlit": 2,
    "Darkness - no lighting": 2,
    "Darkness - lighting unknown": 2,
    "Unknown": 1,
}

SURFACE_SEVERITY_WEIGHTS = {
    "Dry": 0,
    "Wet or damp": 1,
    "Snow": 2,
    "Frost or ice": 2,
    "Flood": 3,
    "Oil or diesel": 2,
    "Mud": 2,
    "Unknown": 1,
}

HOUR_BUCKETS = {
    "Night (00-05)": range(0, 6),
    "Morning (06-11)": range(6, 12),
    "Afternoon (12-16)": range(12, 17),
    "Evening (17-21)": range(17, 22),
    "Late Night (22-23)": range(22, 24),
}


# --------------------------------------------------
# RISK SCORING
# --------------------------------------------------

RISK_SEVERITY_WEIGHTS = {
    "Minor": 1,
    "Serious": 3,
    "Fatal": 6,
}

RISK_SMOOTHING_K = 8

RISK_LEVELS = [
    ("Low", 0, 25),
    ("Medium", 25, 50),
    ("High", 50, 75),
    ("Critical", 75, 101),
]


# --------------------------------------------------
# HOTSPOT DETECTION
# --------------------------------------------------

KMEANS_K = int(os.getenv("KMEANS_K", 12))

DBSCAN_EPS_KM = float(
    os.getenv("DBSCAN_EPS_KM", 1.5)
)

DBSCAN_MIN_SAMPLES = int(
    os.getenv("DBSCAN_MIN_SAMPLES", 8)
)

KDE_BANDWIDTH = float(
    os.getenv("KDE_BANDWIDTH", 2.0)
)


TOP_N_HOTSPOTS = int(
    os.getenv("TOP_N_HOTSPOTS", 25)
)


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

TEST_SIZE = 0.2
CV_FOLDS = 5

FATAL_DECISION_THRESHOLD = 0.15
