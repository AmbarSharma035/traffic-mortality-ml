"""Tests for data loading and schema validation."""
import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (DATA_PROCESSED, UNIFIED_DATA_PATH, FEATURE_MATRIX_PATH,
                            TARGET_PATH, FEATURE_COLUMNS_PATH, RAW_UK_STATS19)


class TestDataLoading:
    """Test data loading and file existence."""

    def test_uk_stats19_raw_exists(self):
        """Check that UK STATS19 raw data was downloaded."""
        collision_file = RAW_UK_STATS19 / "dft-road-casualty-statistics-collision-2023.csv"
        assert collision_file.exists(), f"UK collision file not found at {collision_file}"

    def test_unified_data_exists(self):
        """Check that unified processed data exists."""
        assert UNIFIED_DATA_PATH.exists(), "Unified accidents CSV not found"

    def test_unified_data_not_empty(self):
        """Check that unified data has records."""
        if not UNIFIED_DATA_PATH.exists():
            pytest.skip("Unified data not available")
        df = pd.read_csv(UNIFIED_DATA_PATH, nrows=5)
        assert len(df) > 0, "Unified data is empty"

    def test_unified_data_has_severity(self):
        """Check severity column exists and has expected values."""
        if not UNIFIED_DATA_PATH.exists():
            pytest.skip("Unified data not available")
        df = pd.read_csv(UNIFIED_DATA_PATH)
        assert 'severity' in df.columns, "Missing 'severity' column"
        valid = {'Minor', 'Serious', 'Fatal'}
        actual = set(df['severity'].dropna().unique())
        assert actual.issubset(valid), f"Unexpected severity values: {actual - valid}"

    def test_unified_data_has_coordinates(self):
        """Check lat/lon columns exist."""
        if not UNIFIED_DATA_PATH.exists():
            pytest.skip("Unified data not available")
        df = pd.read_csv(UNIFIED_DATA_PATH, nrows=5)
        assert 'latitude' in df.columns, "Missing latitude"
        assert 'longitude' in df.columns, "Missing longitude"

    def test_feature_matrix_exists(self):
        """Check feature matrix was created."""
        assert FEATURE_MATRIX_PATH.exists(), "Feature matrix not found"

    def test_target_exists(self):
        """Check target vector was created."""
        assert TARGET_PATH.exists(), "Target vector not found"

    def test_feature_matrix_no_nan(self):
        """Check feature matrix has no NaN values."""
        if not FEATURE_MATRIX_PATH.exists():
            pytest.skip("Feature matrix not available")
        df = pd.read_csv(FEATURE_MATRIX_PATH)
        assert not df.isnull().any().any(), f"NaN found in columns: {df.columns[df.isnull().any()].tolist()}"

    def test_target_values_valid(self):
        """Check target has valid encoded values."""
        if not TARGET_PATH.exists():
            pytest.skip("Target not available")
        y = pd.read_csv(TARGET_PATH).squeeze()
        assert set(y.unique()).issubset({0, 1, 2}), f"Invalid target values: {y.unique()}"
