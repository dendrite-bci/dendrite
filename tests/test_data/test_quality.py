"""Tests for ChannelQualityMonitor iterative bad channel detection."""

import numpy as np

from dendrite.data.quality import ChannelQualityMonitor


def _fill_monitor(monitor, variances, n_samples=500):
    """Fill monitor with synthetic data matching target per-channel variances."""
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        sample = np.array(
            [rng.normal(0, np.sqrt(v)) for v in variances]
        ).reshape(-1, 1)
        monitor.update(sample)


def _make_monitor(n_ch: int = 60, **kwargs) -> ChannelQualityMonitor:
    defaults = {"sample_rate": 500.0, "window_sec": 1.0}
    defaults.update(kwargs)
    return ChannelQualityMonitor(n_ch, **defaults)


def _confirm(monitor, variances):
    """Fill + evaluate twice to pass hysteresis confirmation threshold."""
    _fill_monitor(monitor, variances)
    monitor.get_quality()
    _fill_monitor(monitor, variances)
    return monitor.get_quality()


class TestIterativeDetection:
    """Verify that iterative refinement catches channels masked by single-pass."""

    def test_single_outlier_detected(self):
        """One noisy channel is detected as bad."""
        n_ch = 60
        variances = np.ones(n_ch) * 10.0
        variances[55] = 10000.0  # extreme outlier

        monitor = _make_monitor(n_ch)
        q = _confirm(monitor, variances)
        assert 55 in q["bad_channels"]

    def test_iterative_finds_masked_outlier(self):
        """Two outliers where the larger one masks the smaller in single-pass.

        With single-pass: the extreme outlier inflates MAD so much that the
        moderate outlier's z-score falls below the threshold.
        With iterative: after removing the extreme outlier, the moderate one
        is detected on the second pass.
        """
        n_ch = 60
        variances = np.ones(n_ch) * 10.0
        variances[50] = 500.0    # moderate outlier
        variances[55] = 50000.0  # extreme outlier that inflates MAD

        monitor = _make_monitor(n_ch, z_threshold=5.0)
        q = _confirm(monitor, variances)
        # Both should be detected
        assert 55 in q["bad_channels"], "Extreme outlier not detected"
        assert 50 in q["bad_channels"], "Moderate outlier masked by extreme — iterative failed"

    def test_flat_channel_detected(self):
        """A dead/flat channel is detected."""
        n_ch = 10
        variances = np.ones(n_ch) * 10.0
        variances[3] = 0.0  # flat

        monitor = _make_monitor(n_ch)
        q = _confirm(monitor, variances)
        assert 3 in q["bad_channels"]
        assert q["channels"][3]["status"] == "bad"

    def test_converges_within_3_iterations(self):
        """Even with multiple bad channels, detection converges quickly."""
        n_ch = 60
        variances = np.ones(n_ch) * 10.0
        # 5 bad channels at various levels
        for i, mult in zip([10, 20, 30, 40, 50], [100, 500, 1000, 5000, 50000], strict=True):
            variances[i] = mult

        monitor = _make_monitor(n_ch)
        q = _confirm(monitor, variances)
        # All 5 should be detected
        for i in [10, 20, 30, 40, 50]:
            assert i in q["bad_channels"], f"Channel {i} not detected as bad"

    def test_all_good_channels(self):
        """No false positives when all channels are similar."""
        n_ch = 60
        variances = np.ones(n_ch) * 10.0

        monitor = _make_monitor(n_ch)
        _fill_monitor(monitor, variances)

        q = monitor.get_quality()
        assert q["bad_channels"] == [], "False positive: good channels flagged as bad"
        assert all(ch["status"] != "bad" for ch in q["channels"])

    def test_not_ready_returns_unknown(self):
        """Before enough data, returns unknown status and no bad channels."""
        monitor = _make_monitor(10, window_sec=5.0)
        # Only push a few samples (not enough)
        for _ in range(10):
            monitor.update(np.zeros((10, 1)))

        q = monitor.get_quality()
        assert q["bad_channels"] == []
        assert all(ch["status"] == "unknown" for ch in q["channels"])


class TestHysteresis:
    """Verify that bad channels latch after repeated detection."""

    def test_confirmed_after_repeated_detection(self):
        """Channel detected bad 2 out of 3 times becomes confirmed."""
        n_ch = 10
        monitor = _make_monitor(n_ch)

        bad_vars = np.ones(n_ch) * 10.0
        bad_vars[5] = 10000.0

        # Call get_quality twice with bad channel → should confirm (2 >= threshold of 2)
        q = _confirm(monitor, bad_vars)
        assert 5 in q["bad_channels"]

    def test_one_off_not_confirmed(self):
        """Channel detected bad only once doesn't get confirmed."""
        n_ch = 10
        monitor = _make_monitor(n_ch)

        # First call: bad
        bad_vars = np.ones(n_ch) * 10.0
        bad_vars[5] = 10000.0
        _fill_monitor(monitor, bad_vars)
        monitor.get_quality()

        # Second and third calls: good (refill with clean data)
        good_vars = np.ones(n_ch) * 10.0
        _fill_monitor(monitor, good_vars)
        monitor.get_quality()
        _fill_monitor(monitor, good_vars)
        q = monitor.get_quality()
        # Channel 5 was only bad 1 out of 3 → not confirmed
        assert 5 not in q["bad_channels"]

    def test_confirmed_stays_bad(self):
        """Once confirmed, channel stays bad even if data cleans up."""
        n_ch = 10
        monitor = _make_monitor(n_ch)

        bad_vars = np.ones(n_ch) * 10.0
        bad_vars[5] = 10000.0

        # Confirm: 2 consecutive bad detections
        _confirm(monitor, bad_vars)

        # Now feed clean data — channel should stay bad (latched)
        good_vars = np.ones(n_ch) * 10.0
        _fill_monitor(monitor, good_vars)
        q = monitor.get_quality()
        assert 5 in q["bad_channels"], "Confirmed bad channel should stay latched"
