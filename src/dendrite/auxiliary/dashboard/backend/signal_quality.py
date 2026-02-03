#!/usr/bin/env python
"""
Signal Quality Analyzer

Stateless per-channel signal quality computation from raw EEG ring buffers.
Produces composite quality levels (GOOD / WARNING / BAD / UNKNOWN) based on
relative variance, line noise ratio, and cross-channel variance z-score.

Uses unit-agnostic relative thresholds (matching dendrite.data.quality)
instead of absolute amplitude thresholds.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dendrite.data.quality import compute_mad

# Relative thresholds (unit-agnostic)
FLAT_VARIANCE_RATIO = 0.01  # < 1% of median variance -> flat
OUTLIER_Z_WARN = 3.0  # MAD z-score warning
OUTLIER_Z_BAD = 5.0  # MAD z-score bad
LINE_NOISE_WARN = 0.15
LINE_NOISE_BAD = 0.30


class QualityLevel(Enum):
    """Composite signal quality level for a single channel."""

    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    UNKNOWN = "unknown"


@dataclass
class ChannelQualityResult:
    """Quality assessment for a single EEG channel."""

    channel_index: int
    channel_label: str
    level: QualityLevel
    variance_ratio: float = 0.0
    line_noise_ratio: float = 0.0
    variance_zscore: float = 0.0


class SignalQualityAnalyzer:
    """Computes per-channel signal quality from EEG ring buffers.

    Uses relative, data-driven thresholds matching ChannelQualityDetector
    in dendrite.data.quality, making detection unit-agnostic.
    """

    def __init__(self, sample_rate: float, analysis_seconds: float = 2.0, line_freq: float = 50.0):
        self.sample_rate = sample_rate
        self.analysis_seconds = analysis_seconds
        self.line_freq = line_freq

    def analyze(
        self, eeg_buffers: list[deque], channel_labels: list[str]
    ) -> list[ChannelQualityResult]:
        """Compute quality metrics for each EEG channel.

        Args:
            eeg_buffers: Ring buffers, one per channel.
            channel_labels: Human-readable labels matching buffer indices.

        Returns:
            List of ChannelQualityResult, one per channel.
        """
        n_channels = len(eeg_buffers)
        if n_channels == 0:
            return []

        n_samples = int(self.sample_rate * self.analysis_seconds)
        results: list[ChannelQualityResult] = []
        channel_data: list[np.ndarray | None] = []

        # First pass: extract data
        for i in range(n_channels):
            buf = eeg_buffers[i]
            label = channel_labels[i] if i < len(channel_labels) else f"EEG_{i}"

            if len(buf) < max(n_samples, 64):
                results.append(
                    ChannelQualityResult(
                        channel_index=i,
                        channel_label=label,
                        level=QualityLevel.UNKNOWN,
                    )
                )
                channel_data.append(None)
                continue

            data = np.array(list(buf)[-n_samples:], dtype=np.float64)
            channel_data.append(data)
            results.append(
                ChannelQualityResult(
                    channel_index=i,
                    channel_label=label,
                    level=QualityLevel.UNKNOWN,
                )
            )

        # Compute per-channel variances
        variances: list[float] = []
        valid_indices: list[int] = []
        for i, data in enumerate(channel_data):
            if data is not None:
                variances.append(float(np.var(data)))
                valid_indices.append(i)

        if not valid_indices:
            return results

        var_arr = np.array(variances)
        median_var = float(np.median(var_arr))

        # Variance ratios (channel variance / median variance)
        variance_ratios: dict[int, float] = {}
        if median_var > 0:
            for j, idx in enumerate(valid_indices):
                variance_ratios[idx] = variances[j] / median_var

        # One-sided MAD z-scores for high-variance outlier detection
        variance_zscores: dict[int, float] = {}
        if len(variances) >= 3:
            mad_sigma = compute_mad(var_arr)
            if mad_sigma > 1e-12:
                for j, idx in enumerate(valid_indices):
                    # One-sided: only flag high variance (matching quality.py:74)
                    variance_zscores[idx] = (variances[j] - median_var) / mad_sigma

        # Second pass: compute line noise and composite level
        for i in range(n_channels):
            if channel_data[i] is None:
                continue

            data = channel_data[i]
            result = results[i]

            result.variance_ratio = variance_ratios.get(i, 0.0)
            result.line_noise_ratio = self._compute_line_noise_ratio(data)
            result.variance_zscore = variance_zscores.get(i, 0.0)
            result.level = self._composite_level(result)

        return results

    def _compute_line_noise_ratio(self, data: np.ndarray) -> float:
        """Ratio of power at line frequency +/- 2 Hz vs broadband."""
        n = len(data)
        if n < 32:
            return 0.0

        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        line_mask = (freqs >= self.line_freq - 2.0) & (freqs <= self.line_freq + 2.0)
        total_power = float(np.sum(fft_vals**2))

        if total_power < 1e-20:
            return 0.0

        line_power = float(np.sum(fft_vals[line_mask] ** 2))
        return line_power / total_power

    @staticmethod
    def _composite_level(result: ChannelQualityResult) -> QualityLevel:
        """Derive composite quality from individual metrics."""
        # BAD: flat channel, high-variance outlier, or excessive line noise
        if result.variance_ratio < FLAT_VARIANCE_RATIO:
            return QualityLevel.BAD
        if result.variance_zscore >= OUTLIER_Z_BAD:
            return QualityLevel.BAD
        if result.line_noise_ratio >= LINE_NOISE_BAD:
            return QualityLevel.BAD

        # WARNING: moderate outlier or moderate line noise
        if result.variance_zscore >= OUTLIER_Z_WARN:
            return QualityLevel.WARNING
        if result.line_noise_ratio >= LINE_NOISE_WARN:
            return QualityLevel.WARNING

        return QualityLevel.GOOD
