"""
Synthetic accident-data generator.

The real project (see docs/PROJECT_OVERVIEW.md) is designed to run on
MoRTH/data.gov.in, Kaggle US-Accidents, or UK STATS19 records. Those
datasets require manual download / an API key we don't have here, so this
module generates a *realistic, statistically-structured* stand-in dataset
with the same columns and the same underlying risk relationships
(night + fog + highway + speed => much higher fatal probability) that the
real data would show. Swap this file's output for a real CSV with the same
column names and every downstream step keeps working unchanged.
"""
import numpy as np
import pandas as pd

import config


def _build_blackspot_junctions(rng, n_per_city=3):
    """A handful of deliberately dangerous, tightly-clustered junctions per
    city (e.g. a specific highway interchange with poor lighting). Without
    these, synthetic accidents would just scatter uniformly around each
    city center and hotspot detection would have nothing real to find -
    this is what makes DBSCAN/K-Means/KDE recover genuine "black spots"
    instead of only rediscovering the city centers themselves.
    """
    spots = []
    for city, (lat0, lon0, spread) in config.CITY_CENTERS.items():
        for _ in range(n_per_city):
            offset_lat = rng.uniform(-spread * 0.6, spread * 0.6)
            offset_lon = rng.uniform(-spread * 0.6, spread * 0.6)
            spots.append({"city": city, "lat": lat0 + offset_lat, "lon": lon0 + offset_lon})
    return pd.DataFrame(spots)


def _sample_city_points(n, rng, blackspots: pd.DataFrame, blackspot_share: float = 0.30):
    cities = list(config.CITY_CENTERS.keys())
    weights = np.array([1.0, 0.9, 0.8, 0.6, 0.5])
    weights = weights / weights.sum()
    chosen = rng.choice(cities, size=n, p=weights)
    lats, lons = np.empty(n), np.empty(n)
    is_blackspot = rng.random(n) < blackspot_share

    for city in cities:
        city_mask = chosen == city
        bg_mask = city_mask & ~is_blackspot
        bs_mask = city_mask & is_blackspot

        lat0, lon0, spread = config.CITY_CENTERS[city]
        lats[bg_mask] = rng.normal(lat0, spread / 3, size=bg_mask.sum())
        lons[bg_mask] = rng.normal(lon0, spread / 3, size=bg_mask.sum())

        city_spots = blackspots[blackspots["city"] == city].reset_index(drop=True)
        n_bs = bs_mask.sum()
        if n_bs and len(city_spots):
            picked = rng.integers(0, len(city_spots), size=n_bs)
            lats[bs_mask] = city_spots.loc[picked, "lat"].values + rng.normal(0, 0.01, size=n_bs)
            lons[bs_mask] = city_spots.loc[picked, "lon"].values + rng.normal(0, 0.01, size=n_bs)

    return chosen, lats, lons, is_blackspot


def generate(n_records: int = None, seed: int = None) -> pd.DataFrame:
    """Build a synthetic accident-level dataset with realistic risk structure."""
    n_records = n_records or config.N_SYNTHETIC_RECORDS
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.default_rng(seed)

    blackspots = _build_blackspot_junctions(rng)
    city, lat, lon, is_blackspot = _sample_city_points(n_records, rng, blackspots)

    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=n_records), unit="D"
    )
    # Accidents cluster around rush hours and late night, not uniform
    hour_weights = np.array([
        2, 1, 1, 1, 1, 2,           # 0-5
        4, 7, 9, 8, 6, 5,           # 6-11
        5, 5, 5, 6, 7, 9,           # 12-17
        10, 9, 7, 5, 4, 3,          # 18-23
    ], dtype=float)
    hour_weights /= hour_weights.sum()
    hour = rng.choice(24, size=n_records, p=hour_weights)

    road_type = rng.choice(config.ROAD_TYPES, size=n_records, p=[0.30, 0.20, 0.35, 0.15])
    junction_type = rng.choice(config.JUNCTION_TYPES, size=n_records, p=[0.45, 0.25, 0.20, 0.10])
    weather = rng.choice(config.WEATHER_CONDITIONS, size=n_records, p=[0.60, 0.20, 0.12, 0.08])
    road_surface = rng.choice(config.ROAD_SURFACE, size=n_records, p=[0.70, 0.27, 0.03])

    # Light condition correlated with hour, not independent
    light_condition = np.empty(n_records, dtype=object)
    is_daylight_hour = (hour >= 6) & (hour <= 18)
    light_condition[is_daylight_hour] = "Daylight"
    dark_idx = ~is_daylight_hour
    n_dark = dark_idx.sum()
    light_condition[dark_idx] = rng.choice(
        ["Dark-Lit", "Dark-Unlit"], size=n_dark, p=[0.55, 0.45]
    )

    speed_limit = rng.choice([30, 40, 50, 60, 80, 100], size=n_records,
                              p=[0.15, 0.20, 0.20, 0.20, 0.15, 0.10])
    num_vehicles = rng.choice([1, 2, 3, 4], size=n_records, p=[0.35, 0.45, 0.15, 0.05])

    # ---- Latent fatality risk: this is what makes the synthetic data
    # behave like real crash data instead of random noise. Each factor
    # nudges log-odds of a severe/fatal outcome up or down.
    logit = -2.2 + 0.014 * (speed_limit - 50)
    logit += np.where(weather == "Foggy", 0.55, 0)
    logit += np.where(weather == "Rainy", 0.30, 0)
    logit += np.where(light_condition == "Dark-Unlit", 0.65, 0)
    logit += np.where(light_condition == "Dark-Lit", 0.25, 0)
    logit += np.where(road_type == "National Highway", 0.60, 0)
    logit += np.where(road_type == "State Highway", 0.30, 0)
    logit += np.where(road_surface == "Icy", 0.50, 0)
    logit += np.where(road_surface == "Wet", 0.20, 0)
    logit += 0.10 * (num_vehicles - 1)
    logit += np.where(is_blackspot, 0.55, 0)  # black-spot junctions are riskier by design
    logit += rng.normal(0, 0.35, size=n_records)  # unobserved noise

    p_severe = 1 / (1 + np.exp(-logit))       # probability of Serious-or-worse
    p_fatal_given_severe = 1 / (1 + np.exp(-(logit - 1.1)))  # conditional escalation to Fatal

    is_severe = rng.random(n_records) < p_severe
    is_fatal = is_severe & (rng.random(n_records) < p_fatal_given_severe)

    severity = np.where(is_fatal, "Fatal", np.where(is_severe, "Serious", "Minor"))

    df = pd.DataFrame({
        "accident_id": np.arange(1, n_records + 1),
        "date": dates,
        "hour": hour,
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "road_type": road_type,
        "junction_type": junction_type,
        "weather": weather,
        "light_condition": light_condition,
        "road_surface": road_surface,
        "speed_limit": speed_limit,
        "num_vehicles_involved": num_vehicles,
        "severity": severity,
    })

    # Inject realistic messiness: missing values and a few duplicate rows,
    # so the preprocessing module in src/data/preprocess.py has real work to do.
    for col, frac in [("weather", 0.02), ("road_surface", 0.015), ("speed_limit", 0.01)]:
        mask = rng.random(n_records) < frac
        df.loc[mask, col] = np.nan

    dup_rows = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    data = generate()
    config.DATA_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.DATA_RAW_PATH, index=False)
    print(f"Wrote {len(data):,} synthetic accident records -> {config.DATA_RAW_PATH}")
    print(data["severity"].value_counts(normalize=True).round(3))
