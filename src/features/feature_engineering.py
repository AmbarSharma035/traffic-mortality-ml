
"""
Feature engineering module for real STATS19 accident data.

Responsibilities:
1. Create temporal features.
2. Create rush-hour and night-time indicators.
3. Calculate a composite adverse-condition index.
4. Prepare categorical and numerical features for ML models.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


# --------------------------------------------------
# 1. ADD DERIVED FEATURES
# --------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and environmental features
    to the cleaned STATS19 accident dataset.
    """

    df = df.copy()

    # Day of the week
    df["day_of_week"] = df["date"].dt.day_name()

    # Weekend indicator
    df["is_weekend"] = (
        df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    # Rush-hour indicator
    df["is_rush_hour"] = (
        df["hour"].isin(config.RUSH_HOURS).astype(int)
    )

    # Night-time indicator using actual STATS19 labels
    df["is_night"] = (
        df["light_condition"]
        .str.startswith("Darkness", na=False)
        .astype(int)
    )

    # Time-of-day categories
    df["hour_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 16, 21, 23],
        labels=[
            "Night (00-05)",
            "Morning (06-11)",
            "Afternoon (12-16)",
            "Evening (17-21)",
            "Late Night (22-23)",
        ],
    ).astype("string").fillna("Unknown")

    # Weather severity score
    weather_score = (
        df["weather"]
        .map(config.WEATHER_SEVERITY_WEIGHTS)
        .fillna(0)
    )

    # Lighting severity score
    light_score = (
        df["light_condition"]
        .map(config.LIGHT_SEVERITY_WEIGHTS)
        .fillna(0)
    )

    # Road surface severity score
    surface_score = (
        df["road_surface"]
        .map(config.SURFACE_SEVERITY_WEIGHTS)
        .fillna(0)
    )

    # Combined adverse-condition score
    df["weather_severity_index"] = (
        weather_score + light_score + surface_score
    ).astype(int)

    return df


# --------------------------------------------------
# 2. MODEL FEATURE DEFINITIONS
# --------------------------------------------------

CATEGORICAL_FEATURES = [
    "road_type",
    "junction_type",
    "weather",
    "light_condition",
    "road_surface",
    "hour_bucket",
    "city",
]

NUMERIC_FEATURES = [
    "speed_limit",
    "num_vehicles_involved",
    "hour",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "weather_severity_index",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES
)

TARGET_COLUMN = "severity"


# --------------------------------------------------
# 3. BUILD MACHINE LEARNING PREPROCESSOR
# --------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    One-hot encode categorical features
    and scale numerical features.

    Unknown categories are ignored safely.
    """

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


# --------------------------------------------------
# 4. PREPARE MODEL TABLE
# --------------------------------------------------

def prepare_model_table(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate derived features and return
    model-ready columns plus severity target.
    """

    df = add_derived_features(df)

    return df[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ]


# --------------------------------------------------
# 5. TEST FEATURE ENGINEERING
# --------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(
        config.DATA_RAW_PATH,
        parse_dates=["date"],
    )

    featured = prepare_model_table(df)

    print("\nFEATURE ENGINEERING SUCCESSFUL")

    print("\nSample records:")
    print(featured.head())

    print(
        "\nNumber of model features:",
        len(FEATURE_COLUMNS),
    )

    print("\nNight-time distribution:")
    print(featured["is_night"].value_counts())

    print("\nRush-hour distribution:")
    print(featured["is_rush_hour"].value_counts())

    print("\nWeather severity index:")
    print(
        featured["weather_severity_index"]
        .value_counts()
        .sort_index()
    )

    print("\nSeverity distribution:")
    print(featured["severity"].value_counts())
