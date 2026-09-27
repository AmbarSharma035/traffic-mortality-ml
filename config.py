"""
Central configuration for the Traffic Mortality ML project.
All paths and tunable constants live here so every module (and the
Streamlit dashboard) reads the same values instead of hard-coding them.
Values can be overridden via environment variables (see .env.example) -
copy that file to .env and this module will pick it up automatically.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present - every var below still has a default

BASE_DIR = Path(__file__).resolve().parent

# ---- Paths -----------------------------------------------------------------
DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", BASE_DIR / "data/raw/accidents.csv"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", BASE_DIR / "data/processed/accidents_clean.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
METRICS_PATH = MODEL_DIR / "metrics.json"

SEVERITY_MODEL_PATH = MODEL_DIR / "severity_model.joblib"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
RISK_LOOKUP_PATH = MODEL_DIR / "risk_lookup.csv"
HOTSPOT_TABLE_PATH = MODEL_DIR / "hotspots_ranked.csv"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.joblib"

# ---- Reproducibility ---------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# ---- Synthetic data generation -----------------------------------------
N_SYNTHETIC_RECORDS = int(os.getenv("N_SYNTHETIC_RECORDS", 20000))

# Rough bounding box used to scatter synthetic GPS points around a handful
# of Indian metro areas (lat, lon, spread-in-degrees)
CITY_CENTERS = {
    "Delhi": (28.6139, 77.2090, 0.35),
    "Mumbai": (19.0760, 72.8777, 0.30),
    "Bengaluru": (12.9716, 77.5946, 0.30),
    "Chennai": (13.0827, 80.2707, 0.25),
    "Pune": (18.5204, 73.8567, 0.20),
}

ROAD_TYPES = ["National Highway", "State Highway", "Urban Road", "Rural Road"]
JUNCTION_TYPES = ["No Junction", "T-Junction", "Four Arm Junction", "Roundabout"]
WEATHER_CONDITIONS = ["Clear", "Rainy", "Foggy", "Windy"]
LIGHT_CONDITIONS = ["Daylight", "Dark-Lit", "Dark-Unlit"]
ROAD_SURFACE = ["Dry", "Wet", "Icy"]
SEVERITY_CLASSES = ["Minor", "Serious", "Fatal"]  # ordinal, index 2 = rare/critical class

# ---- Feature engineering ------------------------------------------------
RUSH_HOURS = set(list(range(8, 11)) + list(range(17, 21)))  # 8-10am, 5-8pm

# Weights used to build the composite weather-severity index (higher = worse)
WEATHER_SEVERITY_WEIGHTS = {"Clear": 0, "Windy": 1, "Rainy": 2, "Foggy": 3}
LIGHT_SEVERITY_WEIGHTS = {"Daylight": 0, "Dark-Lit": 1, "Dark-Unlit": 2}
SURFACE_SEVERITY_WEIGHTS = {"Dry": 0, "Wet": 1, "Icy": 2}

HOUR_BUCKETS = {
    "Night (00-05)": range(0, 6),
    "Morning (06-11)": range(6, 12),
    "Afternoon (12-16)": range(12, 17),
    "Evening (17-21)": range(17, 22),
    "Late Night (22-23)": range(22, 24),
}

# ---- Risk scoring (Pillar 2) --------------------------------------------
# Severity weights used to turn a mix of Minor/Serious/Fatal accidents at a
# segment/time/weather combo into a single continuous risk score.
RISK_SEVERITY_WEIGHTS = {"Minor": 1, "Serious": 3, "Fatal": 6}
RISK_SMOOTHING_K = 8  # Laplace/credibility smoothing strength (higher = trusts the prior more for low-count groups)
RISK_LEVELS = [("Low", 0, 25), ("Medium", 25, 50), ("High", 50, 75), ("Critical", 75, 101)]

# ---- Hotspot detection (Pillar 3) ---------------------------------------
KMEANS_K = int(os.getenv("KMEANS_K", 12))
DBSCAN_EPS_KM = float(os.getenv("DBSCAN_EPS_KM", 1.5))   # neighbourhood radius in km
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 8))
KDE_BANDWIDTH = float(os.getenv("KDE_BANDWIDTH", 0.02))  # degrees, ~2km
TOP_N_HOTSPOTS = int(os.getenv("TOP_N_HOTSPOTS", 25))

# ---- Model training -------------------------------------------------------
TEST_SIZE = 0.2
CV_FOLDS = 5

# Below this predicted probability for the Fatal class, the model's own
# argmax decision is trusted; at/above it, the prediction is overridden to
# Fatal. Lower than 0.5 (the implicit argmax threshold) on purpose - see
# docs/CODE_EXPLANATION.md "Cost-sensitive decision thresholding" for why.
FATAL_DECISION_THRESHOLD = 0.15
