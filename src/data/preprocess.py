
"""
Preprocessing module for real UK STATS19 accident data.

Responsibilities:
- Remove duplicate accident IDs.
- Preserve unknown categorical values.
- Handle missing numeric values.
- Validate and convert data types.
- Save the cleaned dataset to config.DATA_PROCESSED_PATH.
"""

import pandas as pd
import config


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate accident IDs without deleting distinct accidents."""

    before = len(df)

    if "accident_id" in df.columns:
        df = df.drop_duplicates(
            subset=["accident_id"],
            keep="first"
        )

    removed = before - len(df)

    if removed:
        print(f"[preprocess] removed {removed} duplicate accident ID(s)")

    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with Unknown and numeric values with medians."""

    df = df.copy()

    categorical_cols = [
        "weather",
        "road_surface",
        "junction_type",
        "light_condition",
        "road_type",
        "city"
    ]

    numeric_cols = [
        "speed_limit",
        "num_vehicles_involved",
        "hour"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            if df[col].isna().any():
                median_value = df[col].median()

                if pd.isna(median_value):
                    raise ValueError(
                        f"Cannot impute {col}: no valid numeric values."
                    )

                df[col] = df[col].fillna(median_value)

    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to suitable data types."""

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Some accident dates are invalid or missing."
        )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="raise"
    ).astype(int)

    df["speed_limit"] = pd.to_numeric(
        df["speed_limit"],
        errors="raise"
    ).round().astype(int)

    df["num_vehicles_involved"] = pd.to_numeric(
        df["num_vehicles_involved"],
        errors="raise"
    ).astype(int)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete preprocessing pipeline."""

    df = remove_duplicates(df)
    df = impute_missing(df)
    df = coerce_types(df)

    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset to the configured output location."""

    config.DATA_PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        config.DATA_PROCESSED_PATH,
        index=False
    )

    print(f"\nCleaned dataset saved to: {config.DATA_PROCESSED_PATH}")


if __name__ == "__main__":

    from src.data.load_data import load_raw_data

    raw = load_raw_data(config.DATA_RAW_PATH)

    cleaned = clean(raw)

    save_cleaned_data(cleaned)

    print("\nPreprocessing completed successfully.")

    print("Records before cleaning:", len(raw))
    print("Records after cleaning:", len(cleaned))

    print("\nSeverity distribution:")
    print(cleaned["severity"].value_counts())

    print("\nRemaining missing values:")
    print(cleaned.isna().sum())
