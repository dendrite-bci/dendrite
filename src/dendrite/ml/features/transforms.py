"""
Band power transform for real-time neurofeedback.

Sklearn-compatible transformer for extracting frequency band powers using Welch's method.
"""

from typing import Any

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

from dendrite.utils.logger_central import get_logger


class BandPowerTransform(BaseEstimator, TransformerMixin):
    """
    Extract frequency band powers from time-series data.

    Computes power in specified frequency bands using Welch's method.
    Output is a feature vector for each sample.
    """

    def __init__(
        self,
        bands: dict[str, list[float]] | None = None,
        fs: float = 500.0,
        nperseg: int | None = None,
        relative: bool = False,
    ):
        """Initialize band power transform.

        Args:
            bands: Frequency bands {'name': [low, high]}. Default: standard EEG bands
            fs: Sampling frequency in Hz
            nperseg: Segment length for Welch's method
            relative: If True, compute relative power (band/total)
        """
        self.logger = get_logger(__name__)
        self.is_fitted = False

        if bands is None:
            self.bands = {
                "delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 100.0],
            }
        else:
            self.bands = bands

        self.fs = fs
        self.nperseg = nperseg
        self.relative = relative

        self.logger.info(
            f"BandPowerTransform initialized: {len(self.bands)} bands, relative={relative}"
        )

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray | None = None) -> "BandPowerTransform":
        """Fit the transform (no fitting needed)."""
        self.is_fitted = True
        self.logger.info("BandPowerTransform fitted (no training required)")
        return self

    def transform(self, X: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract band powers."""
        if not isinstance(X, dict):
            raise ValueError("BandPowerTransform expects dictionary input with modality keys")

        result = {}

        for modality, data in X.items():
            if not isinstance(data, np.ndarray):
                self.logger.debug(f"Skipping non-array data for modality '{modality}'")
                result[modality] = data
                continue

            if data.ndim == 2:
                data = data[np.newaxis, :, :]

            if data.ndim != 3:
                self.logger.warning(f"Unexpected shape for {modality}: {data.shape}")
                result[modality] = data
                continue

            batch_size, n_channels, n_times = data.shape

            # Determine nperseg
            if self.nperseg is None:
                nperseg = min(256, n_times)
            else:
                nperseg = min(self.nperseg, n_times)

            # Number of features per sample
            n_features = len(self.bands) * n_channels
            features_output = np.zeros((batch_size, n_features))

            # Compute band powers for each sample
            for sample_idx in range(batch_size):
                # Compute PSD once for all bands
                freqs, psd = signal.welch(data[sample_idx], fs=self.fs, nperseg=nperseg, axis=-1)

                # Extract power for each band and channel
                feature_idx = 0
                for _band_name, (low, high) in self.bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        # Integrate power in band using trapezoidal rule
                        band_power = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1)
                    else:
                        band_power = np.zeros(n_channels)

                    if self.relative:
                        # Compute relative power
                        total_power = np.trapezoid(psd, freqs, axis=1)
                        band_power = band_power / (total_power + 1e-10)

                    features_output[sample_idx, feature_idx : feature_idx + n_channels] = band_power
                    feature_idx += n_channels

            result[modality] = features_output
            self.logger.debug(f"BandPower: {modality} {data.shape} -> {result[modality].shape}")

        return result

    def get_component_info(self) -> dict[str, Any]:
        """Get information about this component."""
        return {
            "name": "BandPowerTransform",
            "is_fitted": self.is_fitted,
            "bands": self.bands,
            "fs": self.fs,
            "nperseg": self.nperseg,
            "relative": self.relative,
        }
