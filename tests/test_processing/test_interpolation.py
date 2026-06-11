"""Tests for bad-channel interpolation (interpolation.py)."""

import numpy as np
import pytest

from dendrite.processing.preprocessing.interpolation import (
    CorrelationInterpolationMatrix,
    InterpolationApplicator,
    SplineInterpolationMatrix,
)

LABELS_10 = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8", "Ch9", "Ch10"]
LABELS_20 = [f"Ch{i}" for i in range(1, 21)]

# Real 10-20 names — all present in MNE's standard_1005 montage.
MONTAGE_24 = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC5", "FC6", "C3", "C4", "Cz",
    "T7", "T8", "CP5", "CP6", "P3", "P4", "Pz", "P7", "P8", "O1", "O2", "Oz",
]


def _montage_field(labels: list[str], montage_name: str = "standard_1005"):
    """Per-channel scalar field equal to a smooth function (z-coordinate) of
    each electrode's 3D position. A spherical spline should reconstruct it well."""
    import mne

    ch_pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    pos = np.array([ch_pos[lbl] for lbl in labels])
    return pos[:, 2], pos  # field, positions


def _make_correlated_data(n_ch: int, n_samples: int = 5000, seed: int = 42):
    """Generate spatially correlated EEG-like data and its correlation matrix."""
    rng = np.random.default_rng(seed)
    # Shared sources + per-channel noise → realistic correlations
    n_sources = max(3, n_ch // 4)
    sources = rng.normal(0, 1, (n_sources, n_samples))
    mixing = rng.normal(0, 1, (n_ch, n_sources))
    data = mixing @ sources + rng.normal(0, 0.3, (n_ch, n_samples))
    corr = np.corrcoef(data)
    return data, corr


# --- CorrelationInterpolationMatrix.compute() tests ---

class TestCorrelationInterpolationMatrix:
    def test_compute_basic(self):
        """Basic interpolation of 1 bad channel from 9 good."""
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [4], corr)
        assert result is not None
        assert result.W.shape == (1, 9)
        assert len(result.bad_indices) == 1
        assert len(result.good_indices) == 9
        assert result.bad_labels == ["Ch5"]
        assert "Ch5" not in result.good_labels

    def test_compute_multiple_bad(self):
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [0, 1], corr)
        assert result is not None
        assert result.W.shape == (2, 8)
        assert list(result.bad_indices) == [0, 1]

    def test_returns_none_for_empty_bad(self):
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [], corr)
        assert result is None

    def test_returns_none_for_too_many_bad(self):
        """More than 20% bad should return None."""
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [0, 1, 2], corr)
        assert result is None

    def test_returns_none_for_wrong_corr_shape(self):
        """Mismatched correlation matrix shape should return None."""
        _, corr = _make_correlated_data(5)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [0], corr)
        assert result is None

    def test_weights_are_finite(self):
        _, corr = _make_correlated_data(20)
        result = CorrelationInterpolationMatrix.compute(LABELS_20, [9], corr)
        assert result is not None
        assert np.all(np.isfinite(result.W))

    def test_weights_sum_to_one(self):
        """Each row of W should sum to 1 (convex combination)."""
        _, corr = _make_correlated_data(20)
        result = CorrelationInterpolationMatrix.compute(LABELS_20, [5, 9], corr)
        assert result is not None
        for row in range(result.W.shape[0]):
            np.testing.assert_allclose(result.W[row].sum(), 1.0, atol=1e-10)

    def test_weights_nonnegative(self):
        """All weights should be >= 0 (negative correlations clipped)."""
        _, corr = _make_correlated_data(20)
        result = CorrelationInterpolationMatrix.compute(LABELS_20, [9], corr)
        assert result is not None
        assert np.all(result.W >= 0)

    def test_higher_corr_gets_higher_weight(self):
        """Channel with higher correlation should get a larger weight."""
        n_ch = 5
        corr = np.eye(n_ch)
        # Ch0 is bad; Ch1 has corr 0.9, Ch2 has corr 0.3, Ch3/Ch4 have 0.1
        corr[0, 1] = corr[1, 0] = 0.9
        corr[0, 2] = corr[2, 0] = 0.3
        corr[0, 3] = corr[3, 0] = 0.1
        corr[0, 4] = corr[4, 0] = 0.1
        labels = ["A", "B", "C", "D", "E"]
        result = CorrelationInterpolationMatrix.compute(labels, [0], corr)
        assert result is not None
        # Ch1 (index 0 in good) should have highest weight
        assert result.W[0, 0] > result.W[0, 1] > result.W[0, 2]

    def test_bad_during_warmup_gets_equal_weights(self):
        """Channels bad during warmup should get uniform weights."""
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(
            LABELS_10, [4], corr, bad_during_warmup=[4],
        )
        assert result is not None
        expected = 1.0 / 9
        np.testing.assert_allclose(result.W[0], expected, atol=1e-10)

    def test_mixed_reliable_and_unreliable(self):
        """Mix of reliable and unreliable bad channels."""
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(
            LABELS_10, [2, 5], corr, bad_during_warmup=[5],
        )
        assert result is not None
        # Ch2 (row 0) uses correlation weights — not uniform
        assert not np.allclose(result.W[0], 1.0 / 8)
        # Ch5 (row 1) was bad during warmup — equal weights
        np.testing.assert_allclose(result.W[1], 1.0 / 8, atol=1e-10)

    def test_sorted_indices(self):
        _, corr = _make_correlated_data(10)
        result = CorrelationInterpolationMatrix.compute(LABELS_10, [5, 2], corr)
        assert result is not None
        assert list(result.bad_indices) == [2, 5]
        assert list(result.good_indices) == sorted(result.good_indices)


# --- SplineInterpolationMatrix.compute() tests ---

class TestSplineInterpolationMatrix:
    def test_reconstructs_smooth_field(self):
        """Hold out a real channel and reconstruct a smooth spatial field."""
        field, _ = _montage_field(MONTAGE_24)
        bad = MONTAGE_24.index("Cz")
        result = SplineInterpolationMatrix.compute(MONTAGE_24, [bad])
        assert result is not None
        recon = (result.W @ field[result.good_indices])[0]
        field_range = float(field.max() - field.min())
        rel_err = abs(recon - field[bad]) / field_range
        assert rel_err < 0.05, f"reconstruction rel err {rel_err:.3f} too high"

    def test_rows_sum_to_one(self):
        result = SplineInterpolationMatrix.compute(MONTAGE_24, [2, 11])
        assert result is not None
        for row in range(result.W.shape[0]):
            np.testing.assert_allclose(result.W[row].sum(), 1.0, atol=1e-6)

    def test_w_shape_matches_positioned_good(self):
        result = SplineInterpolationMatrix.compute(MONTAGE_24, [11])
        assert result is not None
        assert result.W.shape == (1, len(MONTAGE_24) - 1)
        assert list(result.good_indices) == [i for i in range(len(MONTAGE_24)) if i != 11]
        assert result.bad_labels == ["Cz"]

    def test_case_insensitive_labels(self):
        upper = [lbl.upper() for lbl in MONTAGE_24]
        r_canon = SplineInterpolationMatrix.compute(MONTAGE_24, [11])
        r_upper = SplineInterpolationMatrix.compute(upper, [11])
        assert r_canon is not None and r_upper is not None
        np.testing.assert_allclose(r_canon.W, r_upper.W, atol=1e-10)

    def test_returns_none_generic_labels(self):
        """Generic names don't resolve to montage positions → None (fallback)."""
        labels = [f"Ch_{i:02d}" for i in range(24)]
        assert SplineInterpolationMatrix.compute(labels, [5]) is None

    def test_returns_none_when_bad_channel_unpositioned(self):
        """A bad channel with no montage position can't be splined → None."""
        labels = list(MONTAGE_24)
        labels[5] = "Ch_99"
        assert SplineInterpolationMatrix.compute(labels, [5]) is None

    def test_returns_none_too_few_positioned_good(self):
        """Too few positioned good channels for a well-conditioned spline → None."""
        labels = MONTAGE_24[:8]
        assert SplineInterpolationMatrix.compute(labels, [0]) is None

    def test_returns_none_empty_bad(self):
        assert SplineInterpolationMatrix.compute(MONTAGE_24, []) is None

    def test_returns_none_too_many_bad(self):
        """More than 20% bad should return None."""
        bad = list(range(7))  # 7/24 > 20%
        assert SplineInterpolationMatrix.compute(MONTAGE_24, bad) is None


# --- InterpolationApplicator tests ---

class TestInterpolationApplicator:
    @pytest.fixture
    def setup(self):
        data, corr = _make_correlated_data(20)
        result = CorrelationInterpolationMatrix.compute(LABELS_20, [9], corr)
        assert result is not None
        return InterpolationApplicator(result), result

    def test_replaces_bad_channels(self, setup):
        applicator, result = setup
        data = np.random.default_rng(0).normal(0, 1, (20, 100))
        data[9, :] = 999.0
        applicator.apply(data)
        assert not np.any(data[9] == 999.0)
        assert np.all(np.isfinite(data[9]))

    def test_preserves_good_channels(self, setup):
        applicator, result = setup
        data = np.random.default_rng(0).normal(0, 1, (20, 100))
        original_good = data[result.good_indices].copy()
        applicator.apply(data)
        np.testing.assert_array_equal(data[result.good_indices], original_good)

    def test_returns_same_array(self, setup):
        applicator, _ = setup
        data = np.random.default_rng(0).normal(0, 1, (20, 50))
        result = applicator.apply(data)
        assert result is data

    def test_interpolation_reasonable_for_correlated_signal(self):
        """For correlated data, interpolated channel should approximate true value."""
        data, corr = _make_correlated_data(20, n_samples=200, seed=99)
        result = CorrelationInterpolationMatrix.compute(LABELS_20, [9], corr)
        assert result is not None
        applicator = InterpolationApplicator(result)

        true_ch9 = data[9].copy()
        data[9, :] = 0.0
        applicator.apply(data)

        # RMSE should be reasonable — not perfect but capturing the trend
        rmse = np.sqrt(np.mean((data[9] - true_ch9) ** 2))
        true_std = np.std(true_ch9)
        assert rmse < true_std * 1.5, f"RMSE {rmse:.3f} too high (std={true_std:.3f})"


# --- ModalityProcessor integration tests ---

class TestModalityProcessorInterpolation:
    def test_no_interpolation_by_default(self):
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 10,
            "sample_rate": 500.0,
            "lowcut": 0.5,
            "highcut": 50.0,
            "channel_labels": LABELS_10,
        })
        data = np.random.randn(10, 100)
        result = proc.process_chunk(data)
        assert result.shape == (10, 100)

    def test_freeze_and_process(self):
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 20,
            "sample_rate": 500.0,
            "lowcut": 0.5,
            "highcut": 50.0,
            "apply_rereferencing": True,
            "channel_labels": LABELS_20,
        })

        _, corr = _make_correlated_data(20)

        data1 = np.random.randn(20, 100)
        data1[9, :] = 999.0
        result1 = proc.process_chunk(data1)
        assert result1.shape == (20, 100)

        proc.reset_state()
        proc.freeze_interpolation([9], corr_matrix=corr)
        assert proc._interpolator is not None

        data2 = np.random.randn(20, 100)
        data2[9, :] = 999.0
        result2 = proc.process_chunk(data2)
        assert result2.shape == (20, 100)
        assert np.all(np.isfinite(result2))

    def test_freeze_no_labels_is_noop(self):
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 10,
            "sample_rate": 500.0,
        })
        _, corr = _make_correlated_data(10)
        proc.freeze_interpolation([0, 1], corr_matrix=corr)
        assert proc._interpolator is None

    def test_freeze_no_corr_and_generic_labels_is_noop(self):
        """Generic labels (no montage) + no corr matrix → nothing to interpolate."""
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 10,
            "sample_rate": 500.0,
            "channel_labels": LABELS_10,
        })
        proc.freeze_interpolation([0, 1])
        assert proc._interpolator is None

    def test_spline_used_for_real_labels(self):
        """Standard 10-20 labels enable spline even without a correlation matrix."""
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": len(MONTAGE_24),
            "sample_rate": 500.0,
            "lowcut": 0.5,
            "highcut": 50.0,
            "channel_labels": MONTAGE_24,
        })
        proc.freeze_interpolation([MONTAGE_24.index("Cz")], corr_matrix=None)
        assert proc._interpolator is not None

    def test_falls_back_to_correlation_for_generic_labels(self):
        """Generic labels + a correlation matrix → correlation interpolation."""
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 20,
            "sample_rate": 500.0,
            "lowcut": 0.5,
            "highcut": 50.0,
            "channel_labels": LABELS_20,
        })
        _, corr = _make_correlated_data(20)
        proc.freeze_interpolation([9], corr_matrix=corr)
        assert proc._interpolator is not None
        data = np.random.randn(20, 100)
        data[9, :] = 999.0
        result = proc.process_chunk(data)
        assert np.all(np.isfinite(result))

    def test_freeze_empty_bad_is_noop(self):
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor
        proc = ModalityProcessor({
            "num_channels": 10,
            "sample_rate": 500.0,
            "channel_labels": LABELS_10,
        })
        _, corr = _make_correlated_data(10)
        proc.freeze_interpolation([], corr_matrix=corr)
        assert proc._interpolator is None
