"""
Pillar 1 - Accident Severity Prediction.

"Given the conditions logged for a crash that already occurred, how
severe is it - minor, serious, or fatal?" (slide 5 / SRS 3.4).

Four classifiers are benchmarked (Logistic Regression as an interpretable
baseline, then Random Forest / XGBoost / LightGBM for non-linear
interactions). SMOTE runs *inside* each imblearn Pipeline, after the
ColumnTransformer and only on the training fold, so oversampling never
leaks into the validation/test data (SRS FR-3.3-05).

Labels are integer-encoded (0=Minor, 1=Serious, 2=Fatal) before fitting
because XGBoost/LightGBM require integer class ids; metrics are reported
back out using the human-readable names via `target_names`.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import config
from src.features.feature_engineering import (
    FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor,
)
from src.utils.metrics import evaluate_classifier, rank_models_by_fatal_recall

CLASS_ORDER = config.SEVERITY_CLASSES  # ["Minor", "Serious", "Fatal"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}
LABEL_IDS = list(range(len(CLASS_ORDER)))  # [0, 1, 2]


def encode_labels(y: pd.Series) -> pd.Series:
    return y.map(LABEL_TO_IDX)


def _make_pipeline(classifier) -> ImbPipeline:
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=config.RANDOM_SEED, k_neighbors=5)),
        ("classifier", classifier),
    ])


def get_candidate_models() -> dict:
    """The four classifiers named in the synopsis (Objective 2 / slide 5)."""
    return {
        "logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
        ),
        "random_forest": _make_pipeline(
            RandomForestClassifier(n_estimators=300, max_depth=12,
                                    class_weight="balanced",
                                    random_state=config.RANDOM_SEED, n_jobs=-1)
        ),
        "xgboost": _make_pipeline(
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss", random_state=config.RANDOM_SEED,
            )
        ),
        "lightgbm": _make_pipeline(
            LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            objective="multiclass", num_class=len(CLASS_ORDER),
                            random_state=config.RANDOM_SEED, verbosity=-1)
        ),
    }


def train_and_evaluate(df: pd.DataFrame):
    """Split, train every candidate, evaluate on the held-out test set.

    Returns (results: {name: metrics}, fitted_pipelines: {name: pipeline},
    X_test, y_test) so callers (main.py, tests, the dashboard) can pick the
    winner and still have the held-out data for further inspection (e.g. SHAP).
    y_test is returned in its original ("Minor"/"Serious"/"Fatal") form.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    y_encoded = encode_labels(y)

    X_train, X_test, y_train, y_test_enc = train_test_split(
        X, y_encoded, test_size=config.TEST_SIZE, stratify=y_encoded,
        random_state=config.RANDOM_SEED,
    )

    results, fitted = {}, {}
    for name, pipeline in get_candidate_models().items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        # pipeline.classes_ is sorted ([0,1,2]) which already matches LABEL_IDS order
        proba_ordered = pd.DataFrame(y_proba, columns=pipeline.classes_)[LABEL_IDS].values

        metrics = evaluate_classifier(
            y_test_enc, y_pred, proba_ordered,
            labels=LABEL_IDS, target_names=CLASS_ORDER,
        )
        results[name] = metrics
        fitted[name] = pipeline
        print(f"[severity] {name:>20s}  fatal_recall={metrics['fatal_recall']:.3f}  "
              f"macro_f1={metrics['macro_f1']:.3f}  roc_auc={metrics['roc_auc_ovr']:.3f}")

    y_test_labels = y_test_enc.map(IDX_TO_LABEL)
    return results, fitted, X_test, y_test_labels


def cross_validate(df: pd.DataFrame, model_name: str, n_splits=config.CV_FOLDS):
    """Stratified k-fold CV (SRS Appendix B.6) preserving the Fatal ratio in
    every fold - returns the list of per-fold fatal-recall scores."""
    X = df[FEATURE_COLUMNS]
    y_encoded = encode_labels(df[TARGET_COLUMN])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)

    scores = []
    for train_idx, val_idx in skf.split(X, y_encoded):
        pipeline = get_candidate_models()[model_name]
        pipeline.fit(X.iloc[train_idx], y_encoded.iloc[train_idx])
        y_pred = pipeline.predict(X.iloc[val_idx])
        proba = pipeline.predict_proba(X.iloc[val_idx])
        proba_ordered = pd.DataFrame(proba, columns=pipeline.classes_)[LABEL_IDS].values
        metrics = evaluate_classifier(
            y_encoded.iloc[val_idx], y_pred, proba_ordered,
            labels=LABEL_IDS, target_names=CLASS_ORDER,
        )
        scores.append(metrics["fatal_recall"])
    return scores


def select_best_model(results: dict, fitted: dict):
    ranking = rank_models_by_fatal_recall(results)
    best_name = ranking[0]
    return best_name, fitted[best_name], results[best_name], ranking


FATAL_IDX = LABEL_TO_IDX["Fatal"]


def apply_fatal_threshold(proba_ordered: np.ndarray, threshold: float = None) -> np.ndarray:
    """Cost-sensitive decision rule (see config.FATAL_DECISION_THRESHOLD).

    A missed Fatal case is far costlier than a false Fatal alarm (slide 5:
    "What good means"), so instead of trusting plain argmax - which, under
    ~2% class prevalence, rarely picks Fatal even when the model's own
    P(Fatal) is meaningfully elevated - we flag Fatal whenever P(Fatal)
    clears a lower, deliberately-chosen bar. Argmax still decides between
    Minor/Serious when that bar isn't cleared. Returns integer-encoded labels.
    """
    threshold = config.FATAL_DECISION_THRESHOLD if threshold is None else threshold
    argmax_pred = np.argmax(proba_ordered, axis=1)
    fatal_triggered = proba_ordered[:, FATAL_IDX] >= threshold
    return np.where(fatal_triggered, FATAL_IDX, argmax_pred)


def predict_single(pipeline, row_df: pd.DataFrame, threshold: float = None) -> dict:
    """Score one dashboard-style input row. Returns both the model's raw
    argmax call and the recall-oriented thresholded call, plus class
    probabilities, so the UI can show all three."""
    proba = pipeline.predict_proba(row_df)
    proba_ordered = pd.DataFrame(proba, columns=pipeline.classes_)[LABEL_IDS].values[0]
    argmax_idx = int(np.argmax(proba_ordered))
    thresholded_idx = int(apply_fatal_threshold(proba_ordered.reshape(1, -1), threshold)[0])
    return {
        "probabilities": {IDX_TO_LABEL[i]: float(proba_ordered[i]) for i in LABEL_IDS},
        "argmax_prediction": IDX_TO_LABEL[argmax_idx],
        "recall_oriented_prediction": IDX_TO_LABEL[thresholded_idx],
    }
