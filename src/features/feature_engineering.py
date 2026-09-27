"""
Feature engineering module (SRS 3.2, FR-3.2-04..07).

Two responsibilities:
1. add_derived_features(): create the new columns described in the
   synopsis (rush-hour flag, weekend flag, weather-severity index, hour
   bucket) from the already-cleaned table.
2. build_preprocessor(): the sklearn ColumnTransformer that one-hot-encodes
   categoricals and scales numerics, shared by every model so severity
   prediction, risk scoring and hotspot ranking all see features built the
   exact same way.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and composite-risk columns on top of the cleaned data."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin(config.RUSH_HOURS).astype(int)
    df["is_night"] = df["light_condition"].isin(["Dark-Lit", "Dark-Unlit"]).astype(int)

    df["hour_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 16, 21, 23],
        labels=["Night (00-05)", "Morning (06-11)", "Afternoon (12-16)",
                "Evening (17-21)", "Late Night (22-23)"],
    ).astype(str)

    # Composite weather-severity index: 0 (best case) upwards, combining
    # weather condition, light condition and road surface into one score
    # so models/humans get a single "how nasty are conditions" number.
    df["weather_severity_index"] = (
        df["weather"].map(config.WEATHER_SEVERITY_WEIGHTS).fillna(0)
        + df["light_condition"].map(config.LIGHT_SEVERITY_WEIGHTS).fillna(0)
        + df["road_surface"].map(config.SURFACE_SEVERITY_WEIGHTS).fillna(0)
    ).astype(int)

    return df


CATEGORICAL_FEATURES = [
    "road_type", "junction_type", "weather", "light_condition",
    "road_surface", "hour_bucket", "city",
]
NUMERIC_FEATURES = [
    "speed_limit", "num_vehicles_involved", "hour", "is_weekend",
    "is_rush_hour", "is_night", "weather_severity_index",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "severity"


def build_preprocessor() -> ColumnTransformer:
    """One shared ColumnTransformer: one-hot encode categoricals, scale numerics.

    unknown categories are ignored (rather than raising) so the dashboard can
    be handed a category combination it hasn't seen in training data.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def prepare_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: derive features and return only the model-ready columns + target."""
    df = add_derived_features(df)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


if __name__ == "__main__":
    import config
    df = pd.read_csv(config.DATA_PROCESSED_PATH, parse_dates=["date"])
    featured = prepare_model_table(df)
    print(featured.head())
    print(f"\nFeature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
