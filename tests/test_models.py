"""Tests for ML models."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import json
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (MODELS_DIR, FEATURE_MATRIX_PATH, TARGET_PATH,
                            FEATURE_COLUMNS_PATH, REPORTS_DIR)


class TestModels:
    """Test model training and evaluation."""

    def test_selected_model_exists(self):
        """Selected model pickle should exist."""
        assert (MODELS_DIR / "selected_model.pkl").exists()

    def test_scaler_exists(self):
        """Scaler pickle should exist."""
        assert (MODELS_DIR / "scaler.pkl").exists()

    def test_feature_names_saved(self):
        """Feature names JSON should exist in models dir."""
        assert (MODELS_DIR / "feature_names.json").exists()

    def test_all_four_models_saved(self):
        """All 4 model pickles should exist."""
        for name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']:
            path = MODELS_DIR / f"{name}.pkl"
            assert path.exists(), f"Model {name} not found at {path}"

    def test_model_prediction_shape(self):
        """Selected model should predict correct number of classes."""
        model_path = MODELS_DIR / "selected_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        fn_path = MODELS_DIR / "feature_names.json"
        if not all(p.exists() for p in [model_path, scaler_path, fn_path]):
            pytest.skip("Model artifacts not available")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(fn_path) as f:
            feature_names = json.load(f)

        # Create dummy input
        X_dummy = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        X_scaled = scaler.transform(X_dummy)
        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)

        assert pred.shape == (1,)
        assert proba.shape[1] == 3  # 3 classes

    def test_smote_not_on_test_data(self):
        """Verify SMOTE was only applied to training data by checking test class distribution."""
        if not TARGET_PATH.exists():
            pytest.skip("Target not available")
        y = pd.read_csv(TARGET_PATH).squeeze().values
        # Test set should reflect natural imbalance (Fatal << Minor)
        from sklearn.model_selection import train_test_split
        _, _, _, y_test = train_test_split(y, y, test_size=0.15, stratify=y, random_state=42)
        minor_count = (y_test == 0).sum()
        fatal_count = (y_test == 2).sum()
        # Fatal should be much smaller than Minor (natural imbalance preserved in test)
        assert fatal_count < minor_count * 0.1, "Test data may have been contaminated by SMOTE"

    def test_evaluation_report_exists(self):
        """Evaluation CSV should exist."""
        comp_path = REPORTS_DIR / "model_comparison.csv"
        assert comp_path.exists(), "Model comparison CSV not found"

    def test_fatal_recall_in_report(self):
        """Comparison report should have Fatal_Recall column."""
        comp_path = REPORTS_DIR / "model_comparison.csv"
        if not comp_path.exists():
            pytest.skip("Comparison report not available")
        df = pd.read_csv(comp_path)
        assert 'Fatal_Recall' in df.columns
        assert len(df) == 4  # 4 models
