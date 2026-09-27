"""
Trust layer - SHAP (global + local) and LIME (local), slide 11 / SRS 3.7.

"A model that a traffic officer can't question is a model they won't use."

SHAP gives:
  - a GLOBAL view: which features move severity predictions across the
    whole dataset (mean |SHAP value| per feature).
  - a LOCAL view: for one specific prediction, which conditions pushed
    that one score up or down.

LIME is offered as a second, model-agnostic local explainer, exactly as
named in the synopsis ("SHAP/LIME"), useful for double-checking SHAP's
story on a single instance with an independent method.

Works on any of the four fitted pipelines - preprocessing is a
ColumnTransformer, so SHAP explains the model in its own encoded feature
space and we map the encoded feature names back to human names for display.
"""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.features.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def _get_encoded_feature_names(pipeline) -> list:
    """Human-readable names for the one-hot + scaled columns SHAP actually sees,
    e.g. 'weather_Foggy' instead of 'cat__weather_Foggy'."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def build_shap_explainer(pipeline, background_df: pd.DataFrame, sample_size: int = 200):
    """Build a SHAP explainer for the pipeline's final classifier.

    A small random background sample (not the full training set) is used
    on purpose: SHAP's TreeExplainer/KernelExplainer runtime scales with
    background size, and a few hundred rows is enough to estimate a stable
    baseline for a project of this scope.
    """
    background = background_df.sample(
        n=min(sample_size, len(background_df)), random_state=42
    )
    encoded_background = _to_dense(pipeline.named_steps["preprocessor"].transform(background))
    classifier = pipeline.named_steps["classifier"]

    model_type = type(classifier).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(classifier)
    else:
        # Logistic Regression / anything else: a generic explainer that
        # works on predict_proba directly, at the cost of being slower.
        explainer = shap.Explainer(classifier.predict_proba, encoded_background)
    return explainer


def global_feature_importance(pipeline, explainer, sample_df: pd.DataFrame, class_idx: int = 2, top_n: int = 15):
    """Mean |SHAP value| per feature for one class (default: Fatal, class_idx=2),
    ranked descending - the "which factors drive severity across the whole
    dataset" view from slide 11.
    """
    encoded = _to_dense(pipeline.named_steps["preprocessor"].transform(sample_df))
    feature_names = _get_encoded_feature_names(pipeline)

    shap_values = explainer.shap_values(encoded) if hasattr(explainer, "shap_values") else explainer(encoded).values

    if isinstance(shap_values, list):  # older SHAP API: one array per class
        class_values = shap_values[class_idx]
    elif np.ndim(shap_values) == 3:    # shape (n_samples, n_features, n_classes)
        class_values = shap_values[:, :, class_idx]
    else:
        class_values = shap_values

    mean_abs = np.abs(class_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    return importance.head(top_n)


def local_explanation_shap(pipeline, explainer, row_df: pd.DataFrame, class_idx: int = 2, top_n: int = 8):
    """"Why THIS one": per-feature contribution for a single prediction."""
    encoded = _to_dense(pipeline.named_steps["preprocessor"].transform(row_df))
    feature_names = _get_encoded_feature_names(pipeline)

    shap_values = explainer.shap_values(encoded) if hasattr(explainer, "shap_values") else explainer(encoded).values
    if isinstance(shap_values, list):
        row_values = shap_values[class_idx][0]
    elif np.ndim(shap_values) == 3:
        row_values = shap_values[0, :, class_idx]
    else:
        row_values = shap_values[0]

    contributions = pd.Series(row_values, index=feature_names).sort_values(key=np.abs, ascending=False)
    return contributions.head(top_n)


def build_lime_explainer(pipeline, background_df: pd.DataFrame):
    """LIME's own perturbation/scaling machinery needs every column numeric,
    so categorical columns are label-encoded here (LIME calls them
    "categorical_features" and treats them as small integer codes rather
    than one-hot columns). `encoders` is returned alongside the explainer so
    local_explanation_lime() can decode LIME's perturbed integer rows back
    into the raw strings pipeline.predict_proba() actually expects.
    """
    from sklearn.preprocessing import LabelEncoder

    df = background_df[FEATURE_COLUMNS].copy()
    encoded = df.copy()
    encoders, categorical_idx, categorical_names = {}, [], {}

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(df[col].astype(str))
        idx = FEATURE_COLUMNS.index(col)
        categorical_idx.append(idx)
        categorical_names[idx] = list(le.classes_)
        encoders[col] = le

    explainer = LimeTabularExplainer(
        training_data=encoded.values.astype(float),
        feature_names=FEATURE_COLUMNS,
        categorical_features=categorical_idx,
        categorical_names=categorical_names,
        class_names=["Minor", "Serious", "Fatal"],
        mode="classification",
        random_state=42,
    )
    return explainer, encoders


def local_explanation_lime(lime_explainer, encoders: dict, pipeline, row_df: pd.DataFrame,
                            class_idx: int = 2, num_features: int = 8):
    row_raw = row_df[FEATURE_COLUMNS].iloc[0].copy()
    for col, le in encoders.items():
        row_raw[col] = le.transform([str(row_raw[col])])[0]
    row_encoded = row_raw.values.astype(float)

    def predict_fn(rows):
        rows_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        for col, le in encoders.items():
            # LIME perturbs categorical codes as floats close to valid indices -
            # round and clip back onto a real class before decoding to a string.
            idx = rows_df[col].round().clip(0, len(le.classes_) - 1).astype(int)
            rows_df[col] = le.inverse_transform(idx)
        for col in NUMERIC_FEATURES:
            rows_df[col] = pd.to_numeric(rows_df[col])
        return pipeline.predict_proba(rows_df)

    explanation = lime_explainer.explain_instance(
        row_encoded, predict_fn, labels=[class_idx], num_features=num_features
    )
    return dict(explanation.as_list(label=class_idx))


if __name__ == "__main__":
    import config
    from src.features.feature_engineering import prepare_model_table
    from src.models.severity_model import predict_single, train_and_evaluate, select_best_model

    df = pd.read_csv(config.DATA_PROCESSED_PATH, parse_dates=["date"])
    featured = prepare_model_table(df)
    results, fitted, X_test, y_test = train_and_evaluate(featured)
    best_name, best_pipe, _, _ = select_best_model(results, fitted)
    print(f"Explaining: {best_name}")

    explainer = build_shap_explainer(best_pipe, X_test)
    print("\nGlobal feature importance (Fatal class):")
    print(global_feature_importance(best_pipe, explainer, X_test.sample(300, random_state=1)))

    sample_row = X_test.iloc[[0]]
    print(f"\nLocal SHAP explanation for one row:\n{sample_row.iloc[0].to_dict()}")
    print(local_explanation_shap(best_pipe, explainer, sample_row))

    print(f"\nPrediction for that row: {predict_single(best_pipe, sample_row)}")

    lime_explainer, lime_encoders = build_lime_explainer(best_pipe, X_test)
    print("\nLocal LIME explanation for the same row:")
    print(local_explanation_lime(lime_explainer, lime_encoders, best_pipe, sample_row))
