"""Tests for dashboard helper functions."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (MODELS_DIR, DATA_PROCESSED, FIGURES_DIR,
                            REPORTS_DIR, MAPS_DIR)


class TestDashboardArtifacts:
    """Test that dashboard dependencies exist."""

    def test_figures_directory(self):
        """Figures directory should exist and have files."""
        assert FIGURES_DIR.exists()
        figures = list(FIGURES_DIR.glob("*.png"))
        assert len(figures) > 0, "No figures generated"

    def test_confusion_matrices_figure(self):
        """Confusion matrices plot should exist."""
        assert (FIGURES_DIR / "confusion_matrices.png").exists()

    def test_smote_distribution_figure(self):
        """SMOTE distribution plot should exist."""
        assert (FIGURES_DIR / "smote_distribution.png").exists()

    def test_model_comparison_csv(self):
        """Model comparison CSV should exist."""
        assert (REPORTS_DIR / "model_comparison.csv").exists()

    def test_class_distribution_figure(self):
        """Class distribution plot should exist."""
        assert (FIGURES_DIR / "class_distribution_raw.png").exists()

    def test_fatal_recall_comparison_figure(self):
        """Fatal recall comparison plot should exist."""
        assert (FIGURES_DIR / "fatal_recall_comparison.png").exists()

    def test_model_comparison_figure(self):
        """Model comparison plot should exist."""
        assert (FIGURES_DIR / "model_comparison.png").exists()

    def test_streamlit_app_exists(self):
        """Streamlit app file should exist."""
        app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
        assert app_path.exists()
