"""EOG correction through the SamplePreprocessor (the live mode plumbing).

SamplePreprocessor just builds and streams the OnlinePreprocessor; the regression
self-adapts. This checks the wiring: enabling the flag actually corrects the EEG.
"""

import logging

import numpy as np

from dendrite.processing.modes.mode_utils import SamplePreprocessor
from tests.eog_synthetic import SR, monopolar, ocular_var, stream


def _make(apply_eog: bool) -> SamplePreprocessor:
    eeg = {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True, "filter_order": 4}
    if apply_eog:
        eeg["apply_eog_correction"] = True
    return SamplePreprocessor(
        preproc_config={"eeg": eeg, "eog": {"lowcut": 0.1, "highcut": 10.0}},
        sample_rate=SR, channel_selection={}, modality_labels={},
        shared_state=None, logger=logging.getLogger("test"),
    )


def test_enabled_reduces_ocular():
    eeg, eog = monopolar(seed=2)
    off = stream(_make(False), eeg, eog)
    on = stream(_make(True), eeg, eog)
    n = min(off.shape[1], on.shape[1])
    tail = slice(int(30 * SR), n)
    assert ocular_var(on[:, tail]) < 0.8 * ocular_var(off[:, tail])


def test_disabled_is_passthrough():
    eeg, eog = monopolar(secs=15, seed=2)
    off = stream(_make(False), eeg, eog)
    off2 = stream(_make(False), eeg, eog)
    assert np.allclose(off, off2)


def _make_eeg_only(apply_eog: bool) -> SamplePreprocessor:
    """Single-modality mode config (what the UI actually saves) — no `eog` entry."""
    eeg = {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True, "filter_order": 4}
    if apply_eog:
        eeg["apply_eog_correction"] = True
    return SamplePreprocessor(
        preproc_config={"eeg": eeg},
        sample_rate=SR, channel_selection={}, modality_labels={},
        shared_state=None, logger=logging.getLogger("test"),
    )


def test_eeg_only_config_corrects_with_eog_data():
    """The mode dialog is single-modality, so a saved EEG mode has no `eog` entry — but
    the subprocess still receives the raw EOG channels from the ring buffer. Correction
    is driven by the EEG flag + presence of EOG data alone (no `eog` config needed), so
    enabling it corrects online, matching the offline training path."""
    eeg, eog = monopolar(seed=2)
    off = stream(_make_eeg_only(False), eeg, eog)
    on = stream(_make_eeg_only(True), eeg, eog)
    n = min(off.shape[1], on.shape[1])
    tail = slice(int(30 * SR), n)
    assert ocular_var(on[:, tail]) < 0.8 * ocular_var(off[:, tail])


def test_eeg_only_config_noop_without_eog_channels():
    """No EOG channels in the stream → no estimator is built and the EEG passes
    through identically with the flag on or off."""
    eeg, _ = monopolar(secs=15, seed=2)

    def _run(pre: SamplePreprocessor) -> np.ndarray:
        step, out = 12, []
        for s in range(0, eeg.shape[1] - step, step):
            r = pre.process({"eeg": eeg[:, s:s + step].copy()})
            if r is not None:
                out.append(r["eeg"])
        return np.concatenate(out, axis=1)

    assert np.allclose(_run(_make_eeg_only(True)), _run(_make_eeg_only(False)))
