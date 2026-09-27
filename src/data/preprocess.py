"""
Preprocessing / cleaning module (SRS section 3.2, FR-3.2-01..03).

Responsibilities kept strictly separate from feature_engineering.py:
this module only CLEANS the raw table (missing values, duplicates,
dtypes). Deriving new columns (rush-hour flag, weather index, etc.)
happens in src/features/feature_engineering.py so each module has one job.
"""
import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (FR-3.2-02). Keeps the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "accident_id"])
    removed = before - len(df)
    if removed:
        print(f"[preprocess] removed {removed} duplicate row(s)")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values (FR-3.2-01): mode for categoricals, median for numerics.

    Mode/median imputation is used (rather than dropping rows) because the
    missing fields here are weather/road_surface/speed_limit - dropping
    those rows would silently and disproportionately remove bad-weather
    records, which are exactly the ones the severity model most needs to see.
    """
    df = df.copy()
    categorical_cols = ["weather", "road_surface", "junction_type", "light_condition", "road_type", "city"]
    numeric_cols = ["speed_limit", "num_vehicles_involved", "hour", "latitude", "longitude"]

    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(mode_val)

    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure dtypes are what downstream code expects."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["speed_limit"] = df["speed_limit"].astype(int)
    df["num_vehicles_involved"] = df["num_vehicles_involved"].astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedupe -> impute -> coerce dtypes."""
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = coerce_types(df)
    return df


if __name__ == "__main__":
    import config
    from src.data.load_data import load_raw_data

    raw = load_raw_data(config.DATA_RAW_PATH)
    cleaned = clean(raw)
    config.DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.DATA_PROCESSED_PATH, index=False)
    print(f"Cleaned {len(raw)} -> {len(cleaned)} rows. Saved to {config.DATA_PROCESSED_PATH}")
