"""Tests for preprocessing pipeline."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import UNIFIED_DATA_PATH


class TestPreprocessing:
    """Test preprocessing transformations."""

    @pytest.fixture
    def unified_df(self):
        if not UNIFIED_DATA_PATH.exists():
            pytest.skip("Unified data not available")
        return pd.read_csv(UNIFIED_DATA_PATH)

    def test_severity_mapping(self, unified_df):
        """Severity values should be Minor/Serious/Fatal only."""
        valid = {'Minor', 'Serious', 'Fatal'}
        actual = set(unified_df['severity'].dropna().unique())
        assert actual.issubset(valid)

    def test_temporal_features_present(self, unified_df):
        """Check temporal features were created."""
        for col in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_night']:
            assert col in unified_df.columns, f"Missing temporal feature: {col}"

    def test_hour_range(self, unified_df):
        """Hour should be 0-23."""
        valid = unified_df['hour'].dropna()
        assert valid.min() >= 0 and valid.max() <= 23

    def test_day_of_week_range(self, unified_df):
        """Day of week should be 0-6."""
        valid = unified_df['day_of_week'].dropna()
        assert valid.min() >= 0 and valid.max() <= 6

    def test_month_range(self, unified_df):
        """Month should be 1-12."""
        valid = unified_df['month'].dropna()
        assert valid.min() >= 1 and valid.max() <= 12

    def test_is_weekend_binary(self, unified_df):
        """is_weekend should be 0 or 1."""
        assert set(unified_df['is_weekend'].dropna().unique()).issubset({0, 1})

    def test_country_column(self, unified_df):
        """Country column should exist."""
        assert 'country' in unified_df.columns

    def test_no_empty_severity(self, unified_df):
        """No records should have empty severity after preprocessing."""
        assert unified_df['severity'].isnull().sum() == 0
