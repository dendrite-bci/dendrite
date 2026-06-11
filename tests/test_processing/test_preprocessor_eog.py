"""EOG correction through the full OnlinePreprocessor (CAR + bandpass + adaptive).

Uses a realistic *monopolar* synthetic: every channel shares a common reference, and
the ocular artifact is low-frequency — the conditions the re-reference + phase-matched,
band-limited design is built for.
"""

import numpy as np
from scipy import signal

from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from tests.eog_synthetic import SR, band_var, stream


def _monopolar_stream(n_eeg=16, n_eog=2, secs=120, seed=0):
    """EEG/EOG that share a common reference; ocular is low-freq, plus mu/β neural."""
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    ref_sig = rng.normal(0, 1.0, n)                       # shared reference electrode
    b, a = signal.butter(4, 4 / (SR / 2), "low")
    ocular = signal.lfilter(b, a, rng.normal(0, 1.0, (n_eog, n)), axis=1) * 6  # slow
    bm, am = signal.butter(4, [10 / (SR / 2), 14 / (SR / 2)], "band")
    mu = signal.lfilter(bm, am, rng.normal(0, 1.0, (n_eeg, n)), axis=1) * 2     # neural
    a_mix = rng.normal(0, 1.0, (n_eeg, n_eog))
    eeg = mu + a_mix @ ocular + ref_sig                  # + shared reference
    eog = ocular + ref_sig                               # monopolar: also + reference
    return eeg, eog, mu


def _cfg(n_eeg, n_eog, correct):
    eeg = {"num_channels": n_eeg, "sample_rate": SR, "lowcut": 0.5, "highcut": 50.0,
           "apply_rereferencing": True, "filter_order": 4}
    if correct:
        eeg["apply_eog_correction"] = True
    return {"eeg": eeg,
            "eog": {"num_channels": n_eog, "sample_rate": SR,
                    "lowcut": 0.1, "highcut": 10.0, "filter_order": 2}}


def test_converges_removes_ocular_preserves_mubeta():
    eeg, eog, _ = _monopolar_stream()
    off = stream(OnlinePreprocessor(_cfg(16, 2, False)), eeg, eog)
    on = stream(OnlinePreprocessor(_cfg(16, 2, True)), eeg, eog)
    n = min(off.shape[1], on.shape[1])
    tail = slice(int(30 * SR), n)  # past the convergence window
    # Ocular band reduced...
    assert band_var(on[:, tail], 0.5, 8) < 0.75 * band_var(off[:, tail], 0.5, 8)
    # ...mu/β preserved (band-limited correction can't touch it).
    keep = band_var(on[:, tail], 10, 14) / band_var(off[:, tail], 10, 14)
    assert 0.9 < keep < 1.15


def test_disabled_is_passthrough():
    eeg, eog, _ = _monopolar_stream(secs=20)
    off = stream(OnlinePreprocessor(_cfg(16, 2, False)), eeg, eog)
    off2 = stream(OnlinePreprocessor(_cfg(16, 2, False)), eeg, eog)
    assert np.allclose(off, off2)


def test_corrects_without_eog_config_entry():
    """Correction is driven by the EEG flag + EOG data alone — an `eog` config entry
    is not required. Dropping it must give the same ocular reduction as keeping it."""
    eeg, eog, _ = _monopolar_stream()
    cfg_no_eog = _cfg(16, 2, True)
    del cfg_no_eog["eog"]  # only the EEG entry, with apply_eog_correction
    pre = OnlinePreprocessor(cfg_no_eog)
    assert pre.eog_correction_enabled and not pre.eog_active  # built lazily
    on = stream(pre, eeg, eog)
    assert pre.eog_active  # estimator wired once EOG data flowed
    off = stream(OnlinePreprocessor(_cfg(16, 2, False)), eeg, eog)
    n = min(off.shape[1], on.shape[1])
    tail = slice(int(30 * SR), n)
    assert band_var(on[:, tail], 0.5, 8) < 0.75 * band_var(off[:, tail], 0.5, 8)


def test_highpass_above_ocular_band_is_guarded():
    """apply_eog_correction with an 8 Hz high-pass must NOT crash (degenerate
    [8,8] ocular filter) — correction is skipped and EEG passes through."""
    eeg, eog, _ = _monopolar_stream(secs=15)
    cfg = _cfg(16, 2, True)
    cfg["eeg"]["lowcut"] = 8.0  # motor-imagery band: no ocular content below
    pre = OnlinePreprocessor(cfg)  # must construct without raising
    assert not pre.eog_correction_enabled
    out = stream(pre, eeg, eog)
    assert not pre.eog_active  # nothing built even after EOG data flows
    off = stream(OnlinePreprocessor({**_cfg(16, 2, False),
                                     "eeg": {**_cfg(16, 2, False)["eeg"], "lowcut": 8.0}}),
                 eeg, eog)
    assert np.allclose(out, off)  # passthrough — no correction applied


# Standard 10-20 labels → spline interpolation resolves without a corr matrix.
_LABELS_16 = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC5",
              "FC6", "C3", "C4", "Cz", "T7", "T8", "CP5", "CP6"]


def test_frozen_interpolation_survives_correction():
    """Interpolated bad channels must stay interpolated in the corrected output —
    the band-split path replaces the main (interpolating) processor's output."""
    eeg, eog, _ = _monopolar_stream(secs=45)
    eeg[3] = np.random.default_rng(7).normal(0, 300.0, eeg.shape[1])  # garbage channel
    cfg = _cfg(16, 2, True)
    cfg["eeg"]["channel_labels"] = _LABELS_16
    pre = OnlinePreprocessor(cfg)
    pre.processors["eeg"].freeze_interpolation([3])
    assert pre.processors["eeg"]._interpolator is not None
    out = stream(pre, eeg, eog)
    tail = out[:, int(35 * SR):]
    # Garbage replaced by a neighbor combination — variance on par with good channels.
    assert np.var(tail[3]) < 5 * np.median(np.var(tail, axis=1))


def test_notch_survives_correction():
    """line_freq must keep working when correction is on — the notch is carried
    into the high-band processor whose output replaces the main one."""
    eeg, eog, _ = _monopolar_stream(secs=45)
    t = np.arange(eeg.shape[1]) / SR
    amps = np.random.default_rng(11).uniform(3, 10, (16, 1))  # per-channel: survives CAR
    eeg = eeg + amps * np.sin(2 * np.pi * 50.0 * t)
    cfg = _cfg(16, 2, True)
    cfg["eeg"]["line_freq"] = 50.0
    notched = stream(OnlinePreprocessor(cfg), eeg, eog)
    plain = stream(OnlinePreprocessor(_cfg(16, 2, True)), eeg, eog)
    n = min(notched.shape[1], plain.shape[1])
    tail = slice(int(35 * SR), n)
    assert band_var(notched[:, tail], 48, 52) < 0.1 * band_var(plain[:, tail], 48, 52)


def test_no_eog_modality_is_noop():
    eeg, _, _ = _monopolar_stream(secs=20)
    pre = OnlinePreprocessor(_cfg(16, 2, True))
    # Feed only EEG; correction can't activate without an EOG reference.
    step = 12
    out = [pre.process({"eeg": eeg[:, s:s + step].copy()})["eeg"]
           for s in range(0, eeg.shape[1] - step, step)]
    pre0 = OnlinePreprocessor(_cfg(16, 2, False))
    out0 = [pre0.process({"eeg": eeg[:, s:s + step].copy()})["eeg"]
            for s in range(0, eeg.shape[1] - step, step)]
    assert np.allclose(np.concatenate(out, 1), np.concatenate(out0, 1))
