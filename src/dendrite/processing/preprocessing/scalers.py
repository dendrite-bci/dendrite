"""
Per-channel scaling for EEG data using sklearn StandardScaler.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class ChannelScaler(BaseEstimator, TransformerMixin):
    """
    Per-channel z-score normalization using sklearn StandardScaler.

    For 3D EEG data (n_samples, n_channels, n_times), normalizes each channel
    independently across all samples and time points.

    Example:
        Input: (100, 32, 250) - 100 samples, 32 channels, 250 time points
        Learn: mean/std per channel (32 values each)
        Output: (100, 32, 250) - each channel has mean=0, std=1
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fit scaler on training data."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (n_samples, n_channels, n_times), got {X.shape}")

        n_samples, n_channels, n_times = X.shape
        # Reshape to (n_samples*n_times, n_channels) for per-channel stats
        X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
        self.scaler.fit(X_2d)

        # Pre-cache broadcast arrays as float32 for fast transform
        self._mean_3d = self.scaler.mean_[np.newaxis, :, np.newaxis].astype(np.float32)
        self._scale_3d = self.scaler.scale_[np.newaxis, :, np.newaxis].astype(np.float32)
        return self

    def transform(self, X):
        """Transform data using fitted scaler."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (n_samples, n_channels, n_times), got {X.shape}")

        check_is_fitted(self.scaler)

        # Rebuild cache for legacy scalers deserialized without cached arrays
        if not hasattr(self, "_mean_3d"):
            self._mean_3d = self.scaler.mean_[np.newaxis, :, np.newaxis].astype(np.float32)
            self._scale_3d = self.scaler.scale_[np.newaxis, :, np.newaxis].astype(np.float32)

        return (X - self._mean_3d) / self._scale_3d

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
