import numpy as np
import pandas as pd
import pytest

import config
from src.data.generate_sample_data import generate
from src.data.preprocess import clean
from src.features.feature_engineering import add_derived_features, prepare_model_table
from src.hotspot.hotspot_detection import rank_hotspots, run_dbscan, run_kmeans
from src.models.risk_model import build_risk_lookup, score_segment, train_risk_regressor
from src.models.severity_model import (
    apply_fatal_threshold, encode_labels, get_candidate_models, predict_single,
)


@pytest.fixture(scope="module")
def small_clean_df():
    raw = generate(n_records=1500, seed=1)
    return clean(raw)


@pytest.fixture(scope="module")
def model_table(small_clean_df):
    return prepare_model_table(small_clean_df)


def test_encode_labels_maps_to_expected_integers():
    y = pd.Series(["Minor", "Serious", "Fatal"])
    encoded = encode_labels(y)
    assert list(encoded) == [0, 1, 2]


def test_severity_pipeline_fits_and_predicts(model_table):
    from src.features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

    X = model_table[FEATURE_COLUMNS]
    y = encode_labels(model_table[TARGET_COLUMN])
    pipeline = get_candidate_models()["logistic_regression"]
    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(10))
    assert len(preds) == 10
    assert set(preds).issubset({0, 1, 2})


def test_apply_fatal_threshold_flags_fatal_when_probability_clears_bar():
    # 3 rows: low fatal prob, borderline, clearly fatal
    proba = np.array([
        [0.9, 0.09, 0.01],
        [0.5, 0.35, 0.15],
        [0.2, 0.3, 0.5],
    ])
    preds = apply_fatal_threshold(proba, threshold=0.15)
    assert preds[0] == 0        # Minor: fatal prob (0.01) below threshold, argmax wins
    assert preds[1] == 2        # exactly at threshold -> overridden to Fatal
    assert preds[2] == 2        # clearly Fatal anyway


def test_predict_single_returns_all_three_probabilities(model_table):
    from src.features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

    X = model_table[FEATURE_COLUMNS]
    y = encode_labels(model_table[TARGET_COLUMN])
    pipeline = get_candidate_models()["logistic_regression"]
    pipeline.fit(X, y)

    result = predict_single(pipeline, X.iloc[[0]])
    assert set(result["probabilities"].keys()) == {"Minor", "Serious", "Fatal"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_risk_lookup_scores_are_bounded_0_to_100(small_clean_df):
    featured = add_derived_features(small_clean_df)
    lookup = build_risk_lookup(featured)
    assert lookup["risk_score"].between(0, 100).all()
    assert set(lookup["risk_level"]) <= {"Low", "Medium", "High", "Critical"}


def test_riskier_conditions_score_at_least_as_high_as_safer_ones(small_clean_df):
    featured = add_derived_features(small_clean_df)
    lookup = build_risk_lookup(featured)
    regressor = train_risk_regressor(lookup)

    risky = score_segment(regressor, "National Highway", "No Junction", "Foggy",
                           "Late Night (22-23)", True)
    safe = score_segment(regressor, "Urban Road", "Roundabout", "Clear",
                          "Afternoon (12-16)", False)
    assert risky["risk_score"] >= safe["risk_score"]


def test_kmeans_returns_a_label_per_row(small_clean_df):
    labels, model, score = run_kmeans(small_clean_df, k=5)
    assert len(labels) == len(small_clean_df)
    assert model.n_clusters == 5


def test_dbscan_runs_and_ranking_is_sorted_descending(small_clean_df):
    labels, model, score, n_noise = run_dbscan(small_clean_df)
    ranked = rank_hotspots(small_clean_df, labels, top_n=10)
    scores = ranked["severity_weighted_score"].tolist()
    assert scores == sorted(scores, reverse=True)
