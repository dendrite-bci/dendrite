"""Precomputed correlation-based interpolation for online bad channel repair.

Computes an interpolation weight matrix W from the pairwise Pearson correlation
of EEG channels observed during a warmup period, then applies W @ good_channel_data
per chunk to reconstruct bad channels.

Unlike spherical spline methods (Perrin et al. 1989) which require 3D electrode
positions from a standard montage, this approach derives channel proximity directly
from the signal — channels that are spatially close produce correlated signals due
to volume conduction.  Works with any channel naming convention.
"""

from dataclasses import dataclass

import numpy as np

from dendrite.utils.logger_central import get_logger

logger = get_logger("interpolation")

_MAX_BAD_FRACTION = 0.2  # reject interpolation if >20% channels are bad


@dataclass(frozen=True)
class InterpolationResult:
    """Precomputed interpolation weights."""

    W: np.ndarray             # (n_bad, n_good) weight matrix
    bad_indices: np.ndarray   # sorted int array of bad channel indices
    good_indices: np.ndarray  # sorted int array of good channel indices
    bad_labels: list[str]
    good_labels: list[str]


class CorrelationInterpolationMatrix:
    """Compute interpolation weights from a channel correlation matrix.

    Given a correlation matrix (from warmup data), bad channel indices, and
    channel labels, builds weight matrix W such that:
        interpolated_bad = W @ data_good

    Weights are derived from squared positive correlations — channels with
    stronger correlation get more influence, negative correlations are clipped.
    """

    @staticmethod
    def compute(
        all_labels: list[str],
        bad_indices: list[int],
        corr_matrix: np.ndarray,
        bad_during_warmup: list[int] | None = None,
    ) -> InterpolationResult | None:
        """Compute interpolation weight matrix from correlation structure.

        Args:
            all_labels: Channel labels for ALL channels.
            bad_indices: Indices of bad channels to interpolate.
            corr_matrix: (n_ch, n_ch) Pearson correlation matrix from warmup.
            bad_during_warmup: Indices of channels that were bad throughout
                warmup (unreliable correlations) — uses equal weights for these.

        Returns:
            InterpolationResult with weight matrix, or None if interpolation
            is not possible (no bad channels, too many bad).
        """
        if not bad_indices:
            return None

        n_total = len(all_labels)
        bad_set = set(bad_indices)

        if len(bad_set) / n_total > _MAX_BAD_FRACTION:
            logger.warning(
                f"Too many bad channels ({len(bad_set)}/{n_total} = "
                f"{len(bad_set) / n_total:.0%}), skipping interpolation"
            )
            return None

        if corr_matrix.shape != (n_total, n_total):
            logger.error(
                f"Correlation matrix shape {corr_matrix.shape} doesn't match "
                f"{n_total} channels, skipping interpolation"
            )
            return None

        good_indices = sorted(i for i in range(n_total) if i not in bad_set)
        bad_indices_sorted = sorted(bad_set)

        bad_labels = [all_labels[i] for i in bad_indices_sorted]
        good_labels = [all_labels[i] for i in good_indices]

        unreliable = set(bad_during_warmup or [])
        n_good = len(good_indices)

        # Build weight matrix: (n_bad, n_good)
        W = np.zeros((len(bad_indices_sorted), n_good), dtype=np.float64)

        for row, bad_idx in enumerate(bad_indices_sorted):
            if bad_idx in unreliable:
                # No reliable correlation data — equal weights
                W[row, :] = 1.0 / n_good
            else:
                # Squared positive correlations as weights
                raw = np.nan_to_num(corr_matrix[bad_idx, good_indices], nan=0.0)
                w = np.clip(raw, 0, None) ** 2
                total = w.sum()
                if total > 0:
                    W[row, :] = w / total
                else:
                    # All correlations <= 0 — fall back to equal weights
                    W[row, :] = 1.0 / n_good

        logger.info(
            f"Interpolation matrix computed: {len(bad_indices_sorted)} bad channels "
            f"({bad_labels}) interpolated from {n_good} good channels"
        )

        return InterpolationResult(
            W=W,
            bad_indices=np.array(bad_indices_sorted, dtype=int),
            good_indices=np.array(good_indices, dtype=int),
            bad_labels=bad_labels,
            good_labels=good_labels,
        )


class InterpolationApplicator:
    """Apply precomputed interpolation weights per chunk.

    Frozen after construction. Thread-safe (read-only numpy operations).
    """

    def __init__(self, result: InterpolationResult) -> None:
        self.W = result.W
        self.bad_indices = result.bad_indices
        self.good_indices = result.good_indices

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Replace bad channels via matrix multiply: data[bad] = W @ data[good].

        Args:
            data: (n_channels, n_samples) float64 array.

        Returns:
            Same array with bad channels replaced in-place.
        """
        data[self.bad_indices] = self.W @ data[self.good_indices]
        return data
