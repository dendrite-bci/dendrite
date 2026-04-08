"""
Quality assessment for BMI data.

- ChannelQualityMonitor: Continuous per-channel quality monitoring + bad channel detection
- EpochQualityChecker: Per-epoch quality check during training
"""

from typing import Any

import numpy as np

from dendrite.utils.logger_central import get_logger


def compute_mad(data: np.ndarray) -> float:
    """Compute Median Absolute Deviation scaled to standard deviation."""
    median = np.median(data)
    return np.median(np.abs(data - median)) * 1.4826


def detect_bad_channels(
    data: np.ndarray,
    flat_threshold: float = 0.01,
    z_threshold: float = 5.0,
) -> dict:
    """Stateless bad channel detection using iterative MAD.

    Works on a batch of data (channels, samples). No hysteresis — suitable for
    offline/one-shot analysis.

    Returns:
        Dict with 'channels' (list of {index, status, variance}) and
        'bad_channels' (sorted list of bad channel indices).
    """
    n_ch = data.shape[0]
    variances = np.var(data, axis=1)

    bad_set: set[int] = set()
    z_scores = np.zeros(n_ch)

    for _ in range(3):
        good_mask = np.ones(n_ch, dtype=bool)
        good_mask[list(bad_set)] = False
        if not np.any(good_mask):
            break
        good_vars = variances[good_mask]
        median_var = float(np.median(good_vars))
        mad = compute_mad(good_vars)
        flat_thresh = median_var * flat_threshold if median_var > 0 else 1e-10

        new_bad: set[int] = set()
        for i in range(n_ch):
            v = variances[i]
            z_scores[i] = (v - median_var) / mad if mad > 1e-10 else 0.0
            if v < flat_thresh or z_scores[i] > z_threshold:
                new_bad.add(i)
        if new_bad == bad_set:
            break
        bad_set = new_bad

    channels = []
    for i in range(n_ch):
        v = float(variances[i])
        if i in bad_set:
            status = "bad"
        elif z_scores[i] > z_threshold * 0.6:
            status = "warning"
        else:
            status = "good"
        channels.append({"index": i, "status": status, "variance": v})

    return {"channels": channels, "bad_channels": sorted(bad_set)}


class ChannelQualityMonitor:
    """Continuous per-channel quality monitoring with bad channel detection.

    Replaces the old one-shot ChannelQualityDetector. Accumulates EEG samples
    in a rolling window and classifies channels as good/warning/bad using
    robust MAD-based statistics.

    Serves two purposes:
    1. Initial calibration: after window fills, `get_bad_channels()` returns
       channels to exclude from CAR (stored in SharedState for modes to read)
    2. Continuous monitoring: `get_quality()` returns per-channel status for
       dashboard display via telemetry
    """

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        window_sec: float = 5.0,
        flat_threshold: float = 0.01,
        z_threshold: float = 5.0,
    ) -> None:
        self.n_channels = n_channels
        self.window_size = int(sample_rate * window_sec)
        self.flat_threshold = flat_threshold
        self.z_threshold = z_threshold
        self.logger = get_logger("ChannelQuality")

        # Ring buffer for per-channel samples
        self._buf = np.zeros((n_channels, self.window_size), dtype=np.float64)
        self._pos = 0
        self._count = 0

        # Hysteresis: latch bad channels after repeated detection
        self._bad_history: list[set[int]] = []
        self._confirmed_bad: set[int] = set()
        self._CONFIRM_WINDOW = 3   # look at last N evaluations
        self._CONFIRM_THRESHOLD = 2  # must be bad in at least this many

    @property
    def is_ready(self) -> bool:
        """True when enough data has been collected for reliable detection."""
        return self._count >= self.window_size // 2

    def update(self, eeg_sample: np.ndarray) -> None:
        """Push a single EEG sample (n_channels, 1) into the rolling window."""
        self._buf[:, self._pos] = eeg_sample.flatten()[: self.n_channels]
        self._pos = (self._pos + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)

    def get_bad_channels(self) -> list[int]:
        """Get list of bad channel indices from current window.

        Replaces the old ChannelQualityDetector.detect_from_calibration().
        Returns empty list if not enough data yet.
        """
        if not self.is_ready:
            return []
        return self.get_quality()["bad_channels"]

    def get_quality(self) -> dict:
        """Compute per-channel quality from current window.

        Uses iterative refinement: bad channels are excluded from the
        median/MAD statistics, then channels are re-evaluated against the
        cleaned statistics. Converges in 1-3 rounds (max 3).

        Returns:
            Dict with 'channels' (list of {index, status, variance}) and
            'bad_channels' (list of bad channel indices).
        """
        if not self.is_ready:
            return {
                "channels": [
                    {"index": i, "status": "unknown", "variance": 0.0}
                    for i in range(self.n_channels)
                ],
                "bad_channels": [],
            }

        data = self._buf[:, : self._count] if self._count < self.window_size else self._buf

        # Core detection (shared with offline path)
        result = detect_bad_channels(data, self.flat_threshold, self.z_threshold)
        bad_set = set(result["bad_channels"])

        # Hysteresis: confirm channels that are repeatedly bad, recover those that aren't
        self._bad_history.append(bad_set)
        if len(self._bad_history) > self._CONFIRM_WINDOW:
            self._bad_history.pop(0)
        for ch in bad_set:
            if sum(ch in h for h in self._bad_history) >= self._CONFIRM_THRESHOLD:
                self._confirmed_bad.add(ch)
        for ch in list(self._confirmed_bad):
            if sum(ch in h for h in self._bad_history) == 0:
                self._confirmed_bad.discard(ch)

        # Re-classify with hysteresis-confirmed bad set
        confirmed = self._confirmed_bad
        channels = []
        for ch_info in result["channels"]:
            if ch_info["index"] in confirmed:
                ch_info["status"] = "bad"
            channels.append(ch_info)

        return {"channels": channels, "bad_channels": sorted(confirmed)}


class EpochQualityChecker:
    """Epoch quality assessment with stats tracking.

    Performs data-driven quality checks on epochs and tracks rejection statistics.
    Checks are modality-agnostic (work for EEG, EMG, etc.).

    Checks performed:
        1. NaN/Inf detection - invalid values
        2. Flat signal - variance below threshold (dead channel/disconnection)
        3. Extreme outlier - MAD-based z-score above threshold (artifacts)
    """

    def __init__(self, variance_threshold: float = 1e-12, outlier_threshold: float = 50.0) -> None:
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
