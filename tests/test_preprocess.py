import pandas as pd
import pytest

from src.data.load_data import DataValidationError, validate_schema
from src.data.preprocess import clean, impute_missing, remove_duplicates
from src.features.feature_engineering import add_derived_features


def _sample_df():
    return pd.DataFrame({
        "accident_id": [1, 2, 3],
        "date": pd.to_datetime(["2023-01-02", "2023-06-17", "2023-11-05"]),  # Mon, Sat, Sun
        "hour": [8, 23, 14],
        "city": ["Delhi", "Mumbai", "Delhi"],
        "latitude": [28.6, 19.0, 28.7],
        "longitude": [77.2, 72.8, 77.1],
        "road_type": ["Urban Road", "National Highway", None],
        "junction_type": ["T-Junction", "No Junction", "No Junction"],
        "weather": ["Clear", None, "Foggy"],
        "light_condition": ["Daylight", "Dark-Unlit", "Daylight"],
        "road_surface": ["Dry", "Wet", None],
        "speed_limit": [40, None, 60],
        "num_vehicles_involved": [1, 2, 1],
        "severity": ["Minor", "Fatal", "Serious"],
    })


def test_validate_schema_passes_on_good_data():
    validate_schema(_sample_df())  # should not raise


def test_validate_schema_rejects_missing_columns():
    df = _sample_df().drop(columns=["severity"])
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_validate_schema_rejects_bad_severity_label():
    df = _sample_df()
    df.loc[0, "severity"] = "Catastrophic"
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_validate_schema_rejects_out_of_range_coordinates():
    df = _sample_df()
    df.loc[0, "latitude"] = 200
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_impute_missing_fills_categoricals_and_numerics():
    df = _sample_df()
    out = impute_missing(df)
    assert out.isna().sum().sum() == 0
    # road_type had one None among Urban Road/National Highway -> filled with the mode
    assert out.loc[2, "road_type"] in {"Urban Road", "National Highway"}


def test_remove_duplicates_drops_exact_repeats():
    df = _sample_df()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out = remove_duplicates(dup)
    assert len(out) == len(df)


def test_clean_pipeline_produces_no_missing_values():
    out = clean(_sample_df())
    assert out.isna().sum().sum() == 0
    assert out["hour"].dtype.kind in "iu"


def test_add_derived_features_flags_weekend_and_rush_hour():
    df = clean(_sample_df())
    out = add_derived_features(df)
    # 2023-06-17 is a Saturday
    saturday_row = out[out["date"] == "2023-06-17"].iloc[0]
    assert saturday_row["is_weekend"] == 1
    # hour=8 is inside config.RUSH_HOURS (8-10am)
    monday_row = out[out["hour"] == 8].iloc[0]
    assert monday_row["is_rush_hour"] == 1


def test_weather_severity_index_orders_conditions_correctly():
    df = clean(_sample_df())
    out = add_derived_features(df)
    clear_idx = out.loc[out["weather"] == "Clear", "weather_severity_index"].iloc[0]
    foggy_idx = out.loc[out["weather"] == "Foggy", "weather_severity_index"].iloc[0]
    assert foggy_idx > clear_idx
