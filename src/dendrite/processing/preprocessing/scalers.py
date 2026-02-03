"""
Per-channel scaling for EEG data using sklearn StandardScaler.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


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
        return self

    def transform(self, X):
        """Transform data using fitted scaler."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (n_samples, n_channels, n_times), got {X.shape}")

        n_samples, n_channels, n_times = X.shape
        X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, n_times, n_channels).transpose(0, 2, 1)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
