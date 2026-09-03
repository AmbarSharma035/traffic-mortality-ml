"""
Central configuration loader for the Traffic Mortality ML project.
Loads paths, YAML config, and environment variables.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MAPS_DIR = OUTPUTS_DIR / "maps"

# Source-specific raw data paths
RAW_US_ACCIDENTS = DATA_RAW / "us_accidents"
RAW_UK_STATS19 = DATA_RAW / "uk_stats19"
RAW_MORTH = DATA_RAW / "morth"
RAW_WEATHER = DATA_RAW / "weather"
RAW_GIS = DATA_RAW / "gis"

# Key output files
UNIFIED_DATA_PATH = DATA_PROCESSED / "unified_accidents.csv"
FEATURE_MATRIX_PATH = DATA_PROCESSED / "feature_matrix.csv"
FEATURE_COLUMNS_PATH = DATA_PROCESSED / "feature_columns.json"
TARGET_PATH = DATA_PROCESSED / "target.csv"

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging(name: str = "traffic_ml") -> logging.Logger:
    """Configure and return a logger."""
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------
_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """Load and cache the risk_config.yaml."""
    global _config_cache
    if _config_cache is None:
        config_path = CONFIG_DIR / "risk_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_config_section(section: str) -> Dict[str, Any]:
    """Return a specific section from the config."""
    cfg = load_config()
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found. "
                       f"Available: {list(cfg.keys())}")
    return cfg[section]


# ---------------------------------------------------------------------------
# Directory creation helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all required output directories."""
    for d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR,
              FIGURES_DIR, REPORTS_DIR, MAPS_DIR,
              RAW_US_ACCIDENTS, RAW_UK_STATS19, RAW_MORTH,
              RAW_WEATHER, RAW_GIS]:
        d.mkdir(parents=True, exist_ok=True)
