
from pathlib import Path
import pandas as pd

# Find the project folder
ROOT = Path(__file__).resolve().parents[1]

# Load the prepared dataset
FILE = ROOT / "data" / "processed" / "stats19_prepared.csv"

print("Loading prepared STATS19 dataset...")

df = pd.read_csv(FILE)

# 1. Total records
print("\n========== DATASET SIZE ==========")
print("Total accident records:", len(df))
print("Total columns:", len(df.columns))

# 2. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 3. Severity distribution
print("\n========== SEVERITY DISTRIBUTION ==========")
print(df["severity"].value_counts(dropna=False))

# 4. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 5. Unknown categories
print("\n========== UNKNOWN VALUES ==========")

for column in [
    "city",
    "road_type",
    "junction_type",
    "weather",
    "light_condition",
    "road_surface"
]:
    print(
        column,
        "Unknown:",
        (df[column] == "Unknown").sum()
    )

# 6. Coordinate checks
print("\n========== LOCATION CHECK ==========")
print("Latitude range:", df["latitude"].min(), df["latitude"].max())
print("Longitude range:", df["longitude"].min(), df["longitude"].max())

# 7. Sample records
print("\n========== SAMPLE RECORDS ==========")
print(df.head(5).to_string(index=False))

print("\n========== VALIDATION FINISHED ==========")
