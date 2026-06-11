"""Tests for the adaptive streaming EOG-regression estimator."""

import numpy as np

from dendrite.processing.preprocessing.eog_correction import AdaptiveEOGRegression


def _synthetic(n_eeg=8, n_eog=2, n_samples=6000, mix_scale=5.0, seed=0):
    """eeg = neural + A @ ocular, with neural independent of the ocular reference."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1.0, (n_eog, n_samples))
    neural = rng.normal(0, 1.0, (n_eeg, n_samples))
    a = rng.normal(0, mix_scale, (n_eeg, n_eog))
    eeg = neural + a @ ref
    return eeg, ref, neural, a


class TestAdaptiveEOGRegression:
    def _feed(self, est, eeg_low, eeg_high, ref, step=20):
        out = []
        for s in range(0, ref.shape[1], step):
            sl = slice(s, s + step)
            out.append(est.update_and_apply(eeg_low[:, sl], eeg_high[:, sl], ref[:, sl]))
        return np.concatenate(out, axis=1)

    def test_converges_removes_ocular_recombines_high(self):
        # Low band = neural_low + A@ref (ocular); high band = independent neural_high.
        eeg_low, ref, neural_low, a = _synthetic(n_samples=8000)
        eeg_high = np.random.default_rng(9).normal(0, 1.0, eeg_low.shape)
        est = AdaptiveEOGRegression(8, 2, sample_rate=100.0, min_fit_s=1.0, refit_s=0.5)
        out = self._feed(est, eeg_low, eeg_high, ref)
        assert est.B is not None
        assert np.allclose(est.B, a, atol=0.2)
        # Output ≈ (neural_low - 0) + neural_high; ocular gone, both bands preserved.
        tail = slice(4000, 8000)
        target = neural_low + eeg_high
        for ch in range(8):
            # ocular gone, and both the (corrected) low band and the high band kept.
            assert np.corrcoef(out[ch, tail], target[ch, tail])[0, 1] > 0.95
        # High band is untouched: the independent eeg_high is fully present in out.
        for ch in range(8):
            assert np.corrcoef(out[ch, tail], eeg_high[ch, tail])[0, 1] > 0.5

    def test_passthrough_before_min_fit_is_recombination(self):
        eeg_low, ref, _, _ = _synthetic(n_samples=400)
        eeg_high = np.random.default_rng(3).normal(0, 1.0, eeg_low.shape)
        est = AdaptiveEOGRegression(8, 2, sample_rate=100.0, min_fit_s=10.0, refit_s=1.0)
        out = self._feed(est, eeg_low, eeg_high, ref)
        assert est.B is None                          # never reached 10 s of data
        assert np.allclose(out, eeg_low + eeg_high)   # just the recombination

    def test_shape_mismatch_is_noop(self):
        est = AdaptiveEOGRegression(8, 2, sample_rate=100.0, min_fit_s=1.0, refit_s=0.5)
        eeg_low, ref, _, _ = _synthetic(n_samples=50)
        out = est.update_and_apply(eeg_low, eeg_low, ref[:, :-3])  # ref length mismatch
        assert np.allclose(out, eeg_low + eeg_low)
        assert est._n == 0.0                          # mismatched chunk not accumulated

    def test_reset_clears_state(self):
        eeg_low, ref, _, _ = _synthetic(n_samples=4000)
        eeg_high = np.zeros_like(eeg_low)
        est = AdaptiveEOGRegression(8, 2, sample_rate=100.0, min_fit_s=1.0, refit_s=0.5)
        self._feed(est, eeg_low, eeg_high, ref)
        assert est.B is not None
        est.reset()
        assert est.B is None and est._n == 0.0
