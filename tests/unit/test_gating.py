"""Unit tests for standalone prediction gating utility."""

import pytest
from dendrite.processing.gating import PredictionGating


class TestConfidenceGating:
    """Tests for confidence-based gating."""

    def test_above_threshold_passes(self):
        """Predictions above threshold pass through."""
        gating = PredictionGating(confidence_threshold=0.6, use_confidence=True)
        assert gating.apply(prediction=1, confidence=0.8) == 1

    def test_below_threshold_rejected(self):
        """Predictions below threshold return background_class."""
        gating = PredictionGating(
            confidence_threshold=0.6, background_class=0, use_confidence=True
        )
        assert gating.apply(prediction=1, confidence=0.5) == 0

    def test_at_threshold_passes(self):
        """Predictions at exactly threshold pass through."""
        gating = PredictionGating(confidence_threshold=0.6, use_confidence=True)
        assert gating.apply(prediction=1, confidence=0.6) == 1

    def test_disabled_passes_all(self):
        """When disabled, all predictions pass through."""
        gating = PredictionGating(confidence_threshold=0.9, use_confidence=False)
        assert gating.apply(prediction=1, confidence=0.1) == 1


class TestDwellGating:
    """Tests for dwell time (consistency) gating."""

    def test_consistent_window_passes(self):
        """Consistent predictions within window pass through."""
        gating = PredictionGating(dwell_window_size=3, use_dwell=True, use_confidence=False)
        # Fill buffer with consistent predictions
        assert gating.apply(1, 0.9) == 0  # Buffer not full
        assert gating.apply(1, 0.9) == 0  # Buffer not full
        assert gating.apply(1, 0.9) == 1  # Buffer full, all same

    def test_inconsistent_window_rejected(self):
        """Inconsistent predictions within window return background_class."""
        gating = PredictionGating(
            dwell_window_size=3, background_class=0, use_dwell=True, use_confidence=False
        )
        gating.apply(1, 0.9)
        gating.apply(1, 0.9)
        assert gating.apply(2, 0.9) == 0  # Different prediction, rejected

    def test_buffer_slides(self):
        """Buffer maintains sliding window."""
        gating = PredictionGating(
            dwell_window_size=3, background_class=0, use_dwell=True, use_confidence=False
        )
        # Fill with 1s
        gating.apply(1, 0.9)
        gating.apply(1, 0.9)
        gating.apply(1, 0.9)
        # Add 2s, buffer now has [1, 1, 2] -> inconsistent
        assert gating.apply(2, 0.9) == 0
        # Continue with 2s, buffer becomes [1, 2, 2] -> inconsistent
        assert gating.apply(2, 0.9) == 0
        # Continue with 2s, buffer becomes [2, 2, 2] -> consistent
        assert gating.apply(2, 0.9) == 2

    def test_reset_clears_buffer(self):
        """Reset clears the dwell buffer."""
        gating = PredictionGating(dwell_window_size=3, use_dwell=True, use_confidence=False)
        gating.apply(1, 0.9)
        gating.apply(1, 0.9)
        gating.reset()
        # After reset, buffer is empty again
        assert gating.apply(1, 0.9) == 0  # Buffer not full

    def test_disabled_passes_immediately(self):
        """When disabled, predictions pass through without buffering."""
        gating = PredictionGating(dwell_window_size=5, use_dwell=False, use_confidence=False)
        assert gating.apply(1, 0.9) == 1


class TestCombinedGating:
    """Tests for combined confidence + dwell gating."""

    def test_confidence_applied_before_dwell(self):
        """Confidence gating is applied before dwell gating."""
        gating = PredictionGating(
            confidence_threshold=0.7,
            dwell_window_size=2,
            background_class=0,
            use_confidence=True,
            use_dwell=True,
        )
        # Low confidence -> background, which goes into dwell buffer
        gating.apply(1, 0.5)
        # High confidence, dwell buffer is [0], now adding 1 -> inconsistent
        assert gating.apply(1, 0.9) == 0

    def test_both_disabled(self):
        """When both disabled, all predictions pass through."""
        gating = PredictionGating(
            confidence_threshold=0.99,
            dwell_window_size=10,
            use_confidence=False,
            use_dwell=False,
        )
        assert gating.apply(1, 0.1) == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_window_size_one(self):
        """Window size of 1 passes immediately after first prediction."""
        gating = PredictionGating(dwell_window_size=1, use_dwell=True, use_confidence=False)
        assert gating.apply(1, 0.9) == 1

    def test_custom_background_class(self):
        """Custom background class is used for rejection."""
        gating = PredictionGating(
            confidence_threshold=0.9, background_class=99, use_confidence=True
        )
        assert gating.apply(1, 0.5) == 99

    def test_zero_confidence(self):
        """Zero confidence is rejected."""
        gating = PredictionGating(confidence_threshold=0.1, use_confidence=True)
        assert gating.apply(1, 0.0) == 0
