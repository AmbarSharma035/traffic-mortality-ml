
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. FILE LOCATIONS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "stats19_collision.csv"
GUIDE_FILE = PROJECT_ROOT / "docs" / "stats19_data_guide.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "stats19_prepared.csv"


# --------------------------------------------------
# 2. CHECK THAT THE REQUIRED FILES EXIST
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"STATS19 file not found: {INPUT_FILE}")

if not GUIDE_FILE.exists():
    raise FileNotFoundError(f"Excel guide not found: {GUIDE_FILE}")


# --------------------------------------------------
# 3. LOAD THE ORIGINAL STATS19 DATA
# --------------------------------------------------

print("Loading STATS19 accident data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    dtype={
        "collision_index": "string",
        "local_authority_ons_district": "string"
    }
)

print("Original records:", len(df))


# --------------------------------------------------
# 4. LOAD OFFICIAL DISTRICT NAMES FROM EXCEL GUIDE
# --------------------------------------------------

print("Loading official district names...")

guide = pd.read_excel(
    GUIDE_FILE,
    sheet_name="2024_code_list"
)

district_rows = guide[
    guide["field name"] == "local_authority_ons_district"
]

district_mapping = dict(
    zip(
        district_rows["code/format"].astype(str).str.strip(),
        district_rows["label"].astype(str).str.strip()
    )
)


# --------------------------------------------------
# 5. CONVERT ACCIDENT SEVERITY
# --------------------------------------------------

severity_mapping = {
    1: "Fatal",
    2: "Serious",
    3: "Minor"
}

df["severity"] = pd.to_numeric(
    df["collision_severity"],
    errors="coerce"
).map(severity_mapping)


# --------------------------------------------------
# 6. CONVERT ROAD TYPE
# --------------------------------------------------

road_type_mapping = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
    12: "One way street / Slip road",
    -1: "Unknown"
}

df["road_type"] = pd.to_numeric(
    df["road_type"],
    errors="coerce"
).map(road_type_mapping).fillna("Unknown")


# --------------------------------------------------
# 7. CONVERT JUNCTION DETAIL
# --------------------------------------------------

junction_mapping = {
    0: "Not at or within 20 metres of junction",
    13: "T or staggered junction",
    16: "Crossroads",
    17: "Junction with more than four arms",
    18: "Private drive or entrance",
    19: "Other junction",
    99: "Unknown",
    -1: "Unknown"
}

df["junction_type"] = pd.to_numeric(
    df["junction_detail"],
    errors="coerce"
).map(junction_mapping).fillna("Unknown")


# --------------------------------------------------
# 8. CONVERT WEATHER CONDITIONS
# --------------------------------------------------

weather_mapping = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine high winds",
    5: "Raining high winds",
    6: "Snowing high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Unknown"
}

df["weather"] = pd.to_numeric(
    df["weather_conditions"],
    errors="coerce"
).map(weather_mapping).fillna("Unknown")


# --------------------------------------------------
# 9. CONVERT LIGHT CONDITIONS
# --------------------------------------------------

light_mapping = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Unknown"
}

df["light_condition"] = pd.to_numeric(
    df["light_conditions"],
    errors="coerce"
).map(light_mapping).fillna("Unknown")


# --------------------------------------------------
# 10. CONVERT ROAD SURFACE CONDITIONS
# --------------------------------------------------

surface_mapping = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood",
    6: "Oil or diesel",
    7: "Mud",
    9: "Unknown",
    -1: "Unknown"
}

df["road_surface"] = pd.to_numeric(
    df["road_surface_conditions"],
    errors="coerce"
).map(surface_mapping).fillna("Unknown")


# --------------------------------------------------
# 11. CONVERT DISTRICT CODES INTO OFFICIAL NAMES
# --------------------------------------------------

district_codes = (
    df["local_authority_ons_district"]
    .astype("string")
    .str.strip()
)

df["city"] = district_codes.map(district_mapping).fillna("Unknown")


# --------------------------------------------------
# 12. EXTRACT DATE AND HOUR
# --------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

df["hour"] = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
).dt.hour


# --------------------------------------------------
# 13. CONVERT NUMERIC FEATURES
# --------------------------------------------------

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df["speed_limit"] = pd.to_numeric(
    df["speed_limit"],
    errors="coerce"
).replace({-1: float("nan"), 99: float("nan")})

df["num_vehicles_involved"] = pd.to_numeric(
    df["number_of_vehicles"],
    errors="coerce"
)


# --------------------------------------------------
# 14. CREATE THE PROJECT DATASET
# --------------------------------------------------

prepared = pd.DataFrame({
    "accident_id": df["collision_index"],
    "date": df["date"],
    "hour": df["hour"],
    "city": df["city"],
    "latitude": df["latitude"],
    "longitude": df["longitude"],
    "road_type": df["road_type"],
    "junction_type": df["junction_type"],
    "weather": df["weather"],
    "light_condition": df["light_condition"],
    "road_surface": df["road_surface"],
    "speed_limit": df["speed_limit"],
    "num_vehicles_involved": df["num_vehicles_involved"],
    "severity": df["severity"]
})


# --------------------------------------------------
# 15. REMOVE INVALID TARGETS AND COORDINATES
# --------------------------------------------------

before_filtering = len(prepared)

prepared = prepared.dropna(
    subset=["severity", "latitude", "longitude"]
).copy()

prepared = prepared[
    prepared["latitude"].between(-90, 90)
    & prepared["longitude"].between(-180, 180)
].copy()

print("Records before filtering:", before_filtering)
print("Records after filtering:", len(prepared))
print("Records removed:", before_filtering - len(prepared))


# --------------------------------------------------
# 16. SAVE THE PREPARED DATASET
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

prepared.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("STATS19 PREPARATION COMPLETED")
print("===================================")

print("Output file:", OUTPUT_FILE)
print("Total prepared records:", len(prepared))

print("\nSeverity distribution:")
print(prepared["severity"].value_counts())

print("\nMissing values:")
print(prepared.isna().sum())

print("\nSample records:")
print(prepared.head().to_string(index=False))
