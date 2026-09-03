"""Tests for hotspot detection."""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import DATA_PROCESSED, MAPS_DIR, MODELS_DIR


class TestHotspots:
    """Test hotspot detection outputs."""

    def test_hotspot_labels_exist(self):
        """Hotspot labels CSV should exist."""
        assert (DATA_PROCESSED / "hotspot_labels.csv").exists()

    def test_temporal_stats_exist(self):
        """Temporal stats JSON should exist."""
        assert (DATA_PROCESSED / "temporal_stats.json").exists()

    def test_hotspot_map_exists(self):
        """Hotspot map HTML should exist."""
        assert (MAPS_DIR / "hotspot_map.html").exists()

    def test_kde_model_saved(self):
        """KDE model should be saved for risk scoring."""
        assert (MODELS_DIR / "kde_model.joblib").exists()

    def test_kmeans_dbscan_labels(self):
        """Hotspot labels should have both kmeans and dbscan columns."""
        import pandas as pd
        path = DATA_PROCESSED / "hotspot_labels.csv"
        if not path.exists():
            pytest.skip("Hotspot labels not available")
        df = pd.read_csv(path)
        assert 'kmeans_label' in df.columns
        assert 'dbscan_label' in df.columns

    def test_temporal_stats_content(self):
        """Temporal stats should have hourly/daily/monthly data."""
        import json
        path = DATA_PROCESSED / "temporal_stats.json"
        if not path.exists():
            pytest.skip("Temporal stats not available")
        with open(path) as f:
            stats = json.load(f)
        assert 'hourly' in stats or 'top_hours' in stats

    def test_kmeans_function(self):
        """K-Means should produce expected output."""
        from src.hotspots.run import run_kmeans
        coords = np.random.rand(100, 2) * 10
        labels, centers, model = run_kmeans(coords, n_clusters=3)
        assert len(labels) == 100
        assert len(centers) == 3

    def test_dbscan_function(self):
        """DBSCAN should produce labels including possible noise (-1)."""
        from src.hotspots.run import run_dbscan
        coords = np.random.rand(100, 2)
        labels, n_clusters, n_noise = run_dbscan(coords, eps_km=50, min_samples=3)
        assert len(labels) == 100
        assert n_clusters >= 0
