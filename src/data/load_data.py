"""
Data ingestion module.

Mirrors the "Accident Data Ingestion and Profiling" requirements from the
project's SRS (FR-3.1-*): validate structure before anything downstream
touches the data, and produce a quick profile so problems are caught early.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "accident_id", "date", "hour", "city", "latitude", "longitude",
    "road_type", "junction_type", "weather", "light_condition",
    "road_surface", "speed_limit", "num_vehicles_involved", "severity",
]

VALID_SEVERITY = {"Minor", "Serious", "Fatal"}


class DataValidationError(ValueError):
    """Raised when the uploaded/loaded dataset fails a structural check."""


def load_raw_data(path) -> pd.DataFrame:
    """Read the raw accident CSV and run structural validation (FR-3.1-01/02)."""
    df = pd.read_csv(path, parse_dates=["date"])
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns exist and severity labels are within the known set.

    Raises DataValidationError instead of silently continuing, because a
    silently-mismatched column here would corrupt every model trained later.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {sorted(missing)}")

    if df.empty:
        raise DataValidationError("Dataset has no rows.")

    bad_labels = set(df["severity"].dropna().unique()) - VALID_SEVERITY
    if bad_labels:
        raise DataValidationError(f"Unexpected severity labels found: {sorted(bad_labels)}")

    out_of_range = ~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)
    if out_of_range.any():
        raise DataValidationError(f"{out_of_range.sum()} rows have out-of-range GPS coordinates.")


def profile(df: pd.DataFrame) -> dict:
    """Return a lightweight profile dict (FR-3.1-04/05/06): counts, missing %,
    duplicate rows and the severity-class distribution."""
    return {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "severity_distribution": df["severity"].value_counts(normalize=True).round(4).to_dict(),
        "date_range": (str(df["date"].min().date()), str(df["date"].max().date())),
    }


if __name__ == "__main__":
    import config
    data = load_raw_data(config.DATA_RAW_PATH)
    from pprint import pprint
    pprint(profile(data))
