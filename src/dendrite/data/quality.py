"""
Quality assessment for BMI data.

Provides channel-level and epoch-level quality detection:
- ChannelQualityDetector: One-shot calibration check for bad channels
- EpochQualityChecker: Per-epoch quality check during training
"""

from typing import Any

import numpy as np

from dendrite.utils.logger_central import get_logger


def compute_mad(data: np.ndarray) -> float:
    """Compute Median Absolute Deviation scaled to standard deviation."""
    median = np.median(data)
    return np.median(np.abs(data - median)) * 1.4826


class ChannelQualityDetector:
    """
    One-shot bad channel detection during calibration.

    Detects:
    - Flat channels (no variance - broken/disconnected electrodes)
    - High variance outliers (poor contact, noise)

    Uses robust z-score (MAD-based) for unit-agnostic detection.
    """

    def __init__(self, z_threshold: float = 5.0) -> None:
        """
        Initialize detector.

        Args:
            z_threshold: Z-score threshold for outlier detection (default 5.0 = 5-sigma)
        """
        self.z_threshold = z_threshold
        self.bad_channels: list[int] = []
        self.detection_complete = False
        self.logger = get_logger("ChannelQuality")

    def detect_from_calibration(self, data: np.ndarray) -> list[int]:
        """
        Detect bad channels from calibration data.

        Args:
            data: EEG data (n_channels, n_samples)

        Returns:
            List of bad channel indices
        """
        if data is None or data.size == 0:
            self.logger.warning("No calibration data provided")
            return []

        n_channels, n_samples = data.shape
        self.logger.info(
            f"Running channel quality detection on {n_channels} channels, {n_samples} samples"
        )

        # Compute variance per channel
        variances = np.var(data, axis=1)

        # Detect flat channels (variance near zero)
        median_var = np.median(variances)
        flat_threshold = median_var * 0.01 if median_var > 0 else 1e-10
        flat_channels = np.where(variances < flat_threshold)[0].tolist()

        # Detect high-variance outliers using robust z-score (MAD)
        mad = compute_mad(variances)
        outlier_channels = []
        if mad > 1e-10:
            z_scores = (variances - median_var) / mad
            outlier_channels = np.where(z_scores > self.z_threshold)[0].tolist()

        # Combine and deduplicate
        self.bad_channels = list(set(flat_channels + outlier_channels))
        self.bad_channels.sort()
        self.detection_complete = True

        # Log results
        n_good = n_channels - len(self.bad_channels)
        if self.bad_channels:
            self.logger.warning(
                f"Channel quality: {n_good}/{n_channels} good | "
                f"Bad channels: {self.bad_channels} "
                f"(flat: {flat_channels}, outliers: {outlier_channels})"
            )
        else:
            self.logger.info(f"Channel quality: all {n_channels} channels good")

        return self.bad_channels

    def get_bad_channels(self) -> list[int]:
        """Get the list of bad channel indices."""
        return self.bad_channels.copy()

    def reset(self) -> None:
        """Reset detector state."""
        self.bad_channels = []
        self.detection_complete = False


class EpochQualityChecker:
    """Epoch quality assessment with stats tracking.

    Performs data-driven quality checks on epochs and tracks rejection statistics.
    Checks are modality-agnostic (work for EEG, EMG, etc.).

    Checks performed:
        1. NaN/Inf detection - invalid values
        2. Flat signal - variance below threshold (dead channel/disconnection)
        3. Extreme outlier - MAD-based z-score above threshold (artifacts)
    """

    def __init__(self, variance_threshold: float = 1e-12, outlier_threshold: float = 10.0) -> None:
        """
        Initialize quality checker.

        Args:
            variance_threshold: Minimum variance for non-flat signal
            outlier_threshold: MAD-based z-score threshold for outlier detection
        """
        self.variance_threshold = variance_threshold
        self.outlier_threshold = outlier_threshold

        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "reasons": {},  # {'nan_or_inf': 3, 'flat_signal': 2, ...}
        }

        self.logger = get_logger()

    def check(self, data_dict: dict[str, np.ndarray]) -> tuple[bool, str | None]:
        """
        Check epoch quality.

        Args:
            data_dict: Data by modality, e.g., {'eeg': array, 'emg': array}

        Returns:
            Tuple of (is_bad, reason). reason is None if epoch is good.
        """
        self.stats["total"] += 1

        for modality, data in data_dict.items():
            if not isinstance(data, np.ndarray):
                continue

            # Check 1: NaN or Inf values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self._record_rejection("nan_or_inf")
                return True, f"{modality}:nan_or_inf"

            # Check 2: Flat signal (near-zero variance)
            if np.var(data) < self.variance_threshold:
                self._record_rejection("flat_signal")
                return True, f"{modality}:flat_signal"

            # Check 3: Amplitude outlier (robust z-score using MAD)
            flat_data = data.flatten()
            median = np.median(flat_data)
            mad = compute_mad(flat_data)
            if mad > 1e-10:  # Avoid division by zero
                max_z = np.max(np.abs(flat_data - median)) / mad
                if max_z > self.outlier_threshold:
                    self._record_rejection("extreme_outlier")
                    return True, f"{modality}:extreme_outlier"

        self.stats["accepted"] += 1
        return False, None

    def _record_rejection(self, reason: str) -> None:
        """Record a rejection in stats."""
        self.stats["rejected"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """
        Get rejection statistics summary.

        Returns:
            Dict with total, accepted, rejected counts and breakdown by reason.
        """
        return {
            "total": self.stats["total"],
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "rejection_rate": self.stats["rejected"] / max(1, self.stats["total"]),
            "reasons": dict(self.stats["reasons"]),
        }

    def get_stats_summary(self) -> str:
        """Get human-readable stats summary string."""
        stats = self.get_stats()
        if stats["rejected"] == 0:
            return f"{stats['accepted']}/{stats['total']} epochs"

        reasons_str = ", ".join(f"{k}={v}" for k, v in stats["reasons"].items())
        return f"{stats['accepted']}/{stats['total']} epochs ({stats['rejected']} rejected: {reasons_str})"
