"""
Evaluation helpers shared by the severity and risk models.

The synopsis is explicit that overall accuracy is the wrong headline
metric for this problem (Fatal is ~2% of records) - these helpers make
sure recall on the Fatal class is always computed and easy to find,
instead of buried inside a generic classification_report string.

Models are trained on integer-encoded labels (0/1/2) because XGBoost and
LightGBM require it; `target_names` maps those integers back to
"Minor"/"Serious"/"Fatal" so the returned metrics dict stays human-readable.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_classifier(y_true, y_pred, y_proba, labels, target_names, fatal_label="Fatal"):
    """Compute the metric set from SRS section 3.7 / Evaluation Strategy slide.

    labels: the integer class ids in the model's own order, e.g. [0, 1, 2].
    target_names: matching human-readable names, e.g. ["Minor","Serious","Fatal"].
    y_proba: array of shape (n_samples, len(labels)), columns in `labels` order.

    Returns a plain dict (JSON-serialisable) with per-class recall
    highlighted separately so `metrics["fatal_recall"]` is always a single
    number a caller can compare across models without parsing text.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    try:
        # roc_auc_score requires integer `labels` in sorted order.
        order = np.argsort(labels)
        sorted_labels = list(np.asarray(labels)[order])
        y_proba_sorted = np.asarray(y_proba)[:, order]
        roc_auc = roc_auc_score(y_true, y_proba_sorted, multi_class="ovr", labels=sorted_labels)
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "roc_auc_ovr": roc_auc,
        "fatal_recall": report.get(fatal_label, {}).get("recall"),
        "fatal_precision": report.get(fatal_label, {}).get("precision"),
        "fatal_f1": report.get(fatal_label, {}).get("f1-score"),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "class_order": target_names,
    }


def rank_models_by_fatal_recall(results: dict) -> list:
    """results: {model_name: metrics_dict}. Returns model names best-first.

    Fatal recall is the primary sort key (missing a fatal case is the
    costliest error); macro-F1 breaks ties so a model that recalls Fatal
    by predicting it everywhere doesn't automatically win.
    """
    return sorted(
        results.keys(),
        key=lambda name: (results[name]["fatal_recall"] or 0, results[name]["macro_f1"]),
        reverse=True,
    )
