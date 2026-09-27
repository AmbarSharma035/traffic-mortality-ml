"""
Pillar 2 - Accident Risk Prediction.

"For a given road, time and weather - before anything has happened - how
dangerous are conditions right now?" (slide 6).

Different grain from Pillar 1 on purpose: severity_model.py scores one
already-occurred accident record. This module scores a *combination* of
conditions (road_type x junction_type x weather x hour_bucket x weekend) -
a place/time/weather cell that may or may not have a crash in it yet.

Honest limitation (documented in docs/PROJECT_OVERVIEW.md and
docs/PROJECT_QA.md "what could be improved"): we only have accident
records, not a traffic-volume/exposure baseline, so the empirical risk
score below measures "how severe are the accidents that DO happen in this
cell", not a true incidence rate per vehicle-km travelled. A production
version would need traffic-count data to divide by.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import config

SEGMENT_KEYS = ["road_type", "junction_type", "weather", "hour_bucket", "is_weekend"]


def _severity_weighted_score(group: pd.DataFrame) -> pd.Series:
    weights = group["severity"].map(config.RISK_SEVERITY_WEIGHTS)
    return pd.Series({
        "n_accidents": len(group),
        "weighted_sum": weights.sum(),
        "fatal_count": (group["severity"] == "Fatal").sum(),
        "serious_count": (group["severity"] == "Serious").sum(),
    })


def build_risk_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accident-level rows into segment/time/weather cells and
    compute a smoothed 0-100 risk score per cell (SRS FR-3.8-01/02).

    Credibility (Laplace) smoothing: a cell with only 2 recorded accidents
    shouldn't get a wilder score than a cell with 500, just because both
    happened to be Fatal. Each cell's raw rate is pulled toward the
    dataset-wide average, in proportion to how few observations back it up.
    """
    global_avg_weight = df["severity"].map(config.RISK_SEVERITY_WEIGHTS).mean()
    max_weight = max(config.RISK_SEVERITY_WEIGHTS.values())

    agg = df.groupby(SEGMENT_KEYS, observed=True).apply(_severity_weighted_score, include_groups=False)
    agg = agg.reset_index()

    k = config.RISK_SMOOTHING_K
    agg["smoothed_avg_weight"] = (
        (agg["weighted_sum"] + k * global_avg_weight) / (agg["n_accidents"] + k)
    )
    # Normalise to a 0-100 scale using the theoretical max weight (all-Fatal cell)
    agg["risk_score"] = (agg["smoothed_avg_weight"] / max_weight * 100).clip(0, 100).round(2)
    agg["risk_level"] = agg["risk_score"].apply(_risk_level)
    return agg.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _risk_level(score: float) -> str:
    for level, low, high in config.RISK_LEVELS:
        if low <= score < high:
            return level
    return config.RISK_LEVELS[-1][0]


RISK_FEATURES = SEGMENT_KEYS


def train_risk_regressor(risk_lookup: pd.DataFrame):
    """Fit a small regressor that generalises the empirical lookup table to
    segment/time/weather combinations that weren't seen often (or at all) in
    the historical data - this is what lets the dashboard score a
    hypothetical "this junction, tonight, in the rain" query (slide 6).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    X = risk_lookup[RISK_FEATURES]
    y = risk_lookup["risk_score"]
    sample_weight = np.log1p(risk_lookup["n_accidents"])  # trust well-observed cells more

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), RISK_FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            random_state=config.RANDOM_SEED,
        )),
    ])
    model.fit(X, y, regressor__sample_weight=sample_weight)
    return model


def score_segment(model, road_type, junction_type, weather, hour_bucket, is_weekend) -> dict:
    """Score one arbitrary road/time/weather combination (used by the dashboard)."""
    row = pd.DataFrame([{
        "road_type": road_type, "junction_type": junction_type, "weather": weather,
        "hour_bucket": hour_bucket, "is_weekend": int(is_weekend),
    }])
    score = float(model.predict(row)[0])
    score = float(np.clip(score, 0, 100))
    return {"risk_score": round(score, 2), "risk_level": _risk_level(score)}


if __name__ == "__main__":
    from src.features.feature_engineering import add_derived_features

    df = pd.read_csv(config.DATA_PROCESSED_PATH, parse_dates=["date"])
    df = add_derived_features(df)
    lookup = build_risk_lookup(df)
    print(lookup.head(10))
    print(f"\n{len(lookup)} unique road/time/weather segments scored.")
