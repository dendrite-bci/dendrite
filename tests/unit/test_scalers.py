"""
Unit tests for BMI preprocessing scalers.

Tests the ChannelScaler class for per-channel z-score normalization.
"""

import pytest
import numpy as np

from dendrite.processing.preprocessing.scalers import ChannelScaler


class TestChannelScaler:
    """Test suite for ChannelScaler."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample 3D EEG data."""
        np.random.seed(42)
        # 20 samples, 4 channels, 50 timepoints
        # Each channel has different mean/std
        X = np.zeros((20, 4, 50))
        X[:, 0, :] = np.random.randn(20, 50) * 2 + 10   # mean=10, std=2
        X[:, 1, :] = np.random.randn(20, 50) * 5 - 5    # mean=-5, std=5
        X[:, 2, :] = np.random.randn(20, 50) * 0.5 + 0  # mean=0, std=0.5
        X[:, 3, :] = np.random.randn(20, 50) * 1 + 100  # mean=100, std=1
        return X

    def test_shape_preservation(self, sample_data):
        """Output shape matches input shape."""
        X = sample_data
        scaler = ChannelScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape

    def test_per_channel_normalization(self, sample_data):
        """Each channel normalized to mean~0, std~1."""
        X = sample_data
        scaler = ChannelScaler()
        X_scaled = scaler.fit_transform(X)

        n_channels = X.shape[1]
        for ch in range(n_channels):
            ch_data = X_scaled[:, ch, :].flatten()
            mean = np.mean(ch_data)
            std = np.std(ch_data)

            assert abs(mean) < 0.01, f"Channel {ch} mean not ~0: {mean}"
            assert abs(std - 1.0) < 0.01, f"Channel {ch} std not ~1: {std}"

    def test_fit_transform_separation(self, sample_data):
        """Transform uses stats from fit, not from new data."""
        X_train = sample_data
        scaler = ChannelScaler()
        scaler.fit(X_train)

        # Create test data with very different distribution
        X_test = np.ones((5, 4, 50)) * 1000

        X_test_scaled = scaler.transform(X_test)

        # Test data should NOT be normalized to mean=0, std=1
        # because it uses training stats
        for ch in range(4):
            ch_data = X_test_scaled[:, ch, :].flatten()
            mean = np.mean(ch_data)
            # Mean should be far from 0 since test data is 1000s
            # and we're using training stats
            assert abs(mean) > 10, f"Channel {ch} using test stats instead of training stats"

    def test_invalid_input_shape_2d(self):
        """Raises ValueError for 2D input."""
        scaler = ChannelScaler()
        X_2d = np.random.randn(10, 32)

        with pytest.raises(ValueError, match="Expected 3D input"):
            scaler.fit(X_2d)

    def test_invalid_input_shape_4d(self):
        """Raises ValueError for 4D input."""
        scaler = ChannelScaler()
        X_4d = np.random.randn(10, 1, 32, 100)

        with pytest.raises(ValueError, match="Expected 3D input"):
            scaler.fit(X_4d)

    def test_transform_before_fit(self, sample_data):
        """Transform before fit raises sklearn NotFittedError."""
        from sklearn.exceptions import NotFittedError

        scaler = ChannelScaler()
        X = sample_data

        with pytest.raises(NotFittedError):
            scaler.transform(X)
