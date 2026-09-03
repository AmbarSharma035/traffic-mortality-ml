"""Tests for feature engineering."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FEATURE_MATRIX_PATH, TARGET_PATH, FEATURE_COLUMNS_PATH, DATA_PROCESSED


class TestFeatures:
    """Test feature engineering output."""

    @pytest.fixture
    def feature_matrix(self):
        if not FEATURE_MATRIX_PATH.exists():
            pytest.skip("Feature matrix not available")
        return pd.read_csv(FEATURE_MATRIX_PATH)

    @pytest.fixture
    def target(self):
        if not TARGET_PATH.exists():
            pytest.skip("Target not available")
        return pd.read_csv(TARGET_PATH).squeeze()

    @pytest.fixture
    def feature_names(self):
        if not FEATURE_COLUMNS_PATH.exists():
            pytest.skip("Feature names not available")
        with open(FEATURE_COLUMNS_PATH) as f:
            return json.load(f)

    def test_feature_matrix_shape(self, feature_matrix, target):
        """Feature matrix rows should match target length."""
        assert len(feature_matrix) == len(target)

    def test_no_nan_in_features(self, feature_matrix):
        """Feature matrix should have no NaN."""
        assert not feature_matrix.isnull().any().any()

    def test_feature_names_match_columns(self, feature_matrix, feature_names):
        """Feature names JSON should match DataFrame columns."""
        assert list(feature_matrix.columns) == feature_names

    def test_target_three_classes(self, target):
        """Target should have exactly 3 classes: 0, 1, 2."""
        assert set(target.unique()) == {0, 1, 2}

    def test_coordinates_exist(self):
        """Coordinates file should exist for hotspot analysis."""
        coords_path = DATA_PROCESSED / "coordinates.csv"
        assert coords_path.exists()

    def test_coordinates_have_lat_lon(self):
        """Coordinates should have latitude and longitude."""
        coords_path = DATA_PROCESSED / "coordinates.csv"
        if not coords_path.exists():
            pytest.skip("Coordinates not available")
        df = pd.read_csv(coords_path, nrows=5)
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
