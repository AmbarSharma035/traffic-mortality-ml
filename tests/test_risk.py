"""Tests for risk scoring."""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk.risk_score import (compute_severity_score, compute_temporal_score,
                                  compute_environmental_score, compute_risk_score,
                                  get_risk_category)


class TestRiskScoring:
    """Test risk score computation."""

    def test_severity_score_range(self):
        """Severity score should be 0-100."""
        assert compute_severity_score(0.0) == 0.0
        assert compute_severity_score(1.0) == 100.0
        assert 0 <= compute_severity_score(0.5) <= 100

    def test_temporal_score_default(self):
        """With no stats, temporal score should be 50 (neutral)."""
        score = compute_temporal_score(12, 3, {})
        assert score == 50.0

    def test_environmental_score_clear_day(self):
        """Clear day should have low environmental score."""
        score = compute_environmental_score(0, 'good', False, None)
        assert score < 30

    def test_environmental_score_severe_night(self):
        """Severe weather + night should have high score."""
        score = compute_environmental_score(3, 'Poor', True, 'Wet')
        assert score >= 50

    def test_risk_score_clipped(self):
        """Final risk score should be clipped to 0-100."""
        score = compute_risk_score(100, 100, 100, 100)
        assert 0 <= score <= 100
        score = compute_risk_score(0, 0, 0, 0)
        assert 0 <= score <= 100

    def test_risk_category_low(self):
        assert get_risk_category(10) == 'Low'

    def test_risk_category_moderate(self):
        assert get_risk_category(45) == 'Moderate'

    def test_risk_category_high(self):
        assert get_risk_category(70) == 'High'

    def test_risk_category_critical(self):
        assert get_risk_category(90) == 'Critical'
