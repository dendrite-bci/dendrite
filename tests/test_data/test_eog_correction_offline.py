"""Offline EOG correction: RawData.preprocess_with_eog_correction + load_epochs.

Offline streams the recording through the same OnlinePreprocessor as the live path,
so training sees the same adaptively-corrected EEG inference produces.
"""

import numpy as np

from dendrite.data.loaders._training_data import load_epochs
from dendrite.data.loaders._types import RawData
from tests.eog_synthetic import SR, monopolar, ocular_var


def _raw(n_eeg=16, n_eog=2, secs=90, seed=0, with_eog=True):
    """Monopolar RawData: shared reference + low-freq ocular + neural, with events."""
    eeg, eog = monopolar(n_eeg=n_eeg, n_eog=n_eog, secs=secs, seed=seed)
    n = eeg.shape[1]

    if with_eog:
        data = np.vstack([eeg, eog]).astype(np.float32)
        names = [f"E{i}" for i in range(n_eeg)] + [f"X{i}" for i in range(n_eog)]
        types = ["eeg"] * n_eeg + ["eog"] * n_eog
    else:
        data = eeg.astype(np.float32)
        names = [f"E{i}" for i in range(n_eeg)]
        types = ["eeg"] * n_eeg

    events = [(s, 1 if k % 2 == 0 else 2)
              for k, s in enumerate(range(int(40 * SR), n - int(2 * SR), int(SR)))]
    return RawData(data=data, channel_names=names, channel_types=types,
                   sample_rate=SR, events=events, event_id={"a": 1, "b": 2})


class TestPreprocessWithEOGCorrection:
    def test_collapses_to_eeg_and_reduces_ocular(self):
        raw = _raw()
        applied = raw.preprocess_with_eog_correction(
            {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True})
        assert applied is True
        assert all(t == "eeg" for t in raw.channel_types)  # EOG dropped
        assert raw.data.shape[0] == 16
        # vs the same recording preprocessed WITHOUT correction.
        plain = _raw()
        plain.filter_modality("eeg")
        plain.preprocess({"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True})
        tail = slice(int(40 * SR), raw.data.shape[1])
        assert ocular_var(raw.data[:, tail]) < 0.85 * ocular_var(plain.data[:, tail])

    def test_returns_false_without_eog(self):
        raw = _raw(with_eog=False)
        assert raw.preprocess_with_eog_correction({"lowcut": 0.5, "highcut": 50.0}) is False


class TestLoadEpochsWiring:
    _CFG = {"modality": "eeg", "event_mapping": {1: "a", 2: "b"},
            "epoch_tmin": 0.0, "epoch_tmax": 1.0, "use_epoch_qc": False}

    def test_flag_on_changes_epochs(self):
        cfg_on = {**self._CFG, "mode_preprocessing": {
            "eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True,
                    "apply_eog_correction": True}}}
        cfg_off = {**self._CFG, "mode_preprocessing": {
            "eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True}}}
        ep_off = load_epochs(cfg_off, _raw(seed=1))
        ep_on = load_epochs(cfg_on, _raw(seed=1))
        assert ep_on.X.shape[1] == 16            # EOG dropped, EEG kept
        assert ep_off.X.shape == ep_on.X.shape
        assert not np.allclose(ep_off.X, ep_on.X)  # correction changed the data

    def test_flag_on_noop_without_eog(self):
        cfg_on = {**self._CFG, "mode_preprocessing": {
            "eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_eog_correction": True}}}
        cfg_off = {**self._CFG, "mode_preprocessing": {
            "eeg": {"lowcut": 0.5, "highcut": 50.0}}}
        ep_a = load_epochs(cfg_off, _raw(seed=2, with_eog=False))
        ep_b = load_epochs(cfg_on, _raw(seed=2, with_eog=False))
        assert np.allclose(ep_a.X, ep_b.X)
