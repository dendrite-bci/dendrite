"""Shared synthetic EEG/EOG fixtures for the EOG-correction tests.

A *monopolar* model: every channel shares a common reference electrode and the
ocular artifact is low-frequency — the conditions the re-reference + phase-matched,
band-limited EOG correction is built for. The RNG draw order (ref → ocular noise →
EEG white noise → mixing matrix) is fixed so results are reproducible across callers.
"""

import numpy as np
from scipy import signal

SR = 250.0


def monopolar(n_eeg=16, n_eog=2, secs=90, seed=0):
    """Monopolar EEG/EOG sharing a common reference; ocular is low-freq.

    Returns ``(eeg, eog)`` where ``eog`` is the ocular signal plus the shared
    reference (i.e. the raw monopolar EOG trace).
    """
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    ref = rng.normal(0, 1.0, n)
    b, a = signal.butter(4, 4 / (SR / 2), "low")
    ocular = signal.lfilter(b, a, rng.normal(0, 1.0, (n_eog, n)), axis=1) * 6
    eeg = rng.normal(0, 1.0, (n_eeg, n)) + rng.normal(0, 1.0, (n_eeg, n_eog)) @ ocular + ref
    return eeg, ocular + ref


def band_var(x, lo, hi):
    """Variance of ``x`` within the ``[lo, hi]`` Hz band (zero-phase)."""
    b, a = signal.butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    return float(np.var(signal.filtfilt(b, a, x, axis=-1)))


def ocular_var(x):
    """Variance in the ocular band [0.5, 8] Hz."""
    return band_var(x, 0.5, 8)


def stream(pre, eeg, eog, step=12):
    """Feed EEG/EOG through a preprocessor in ``step``-sample chunks; cat the EEG out.

    Tolerates preprocessors that return ``None`` for a chunk (the SamplePreprocessor
    plumbing layer does; OnlinePreprocessor never does).
    """
    out = []
    for s in range(0, eeg.shape[1] - step, step):
        r = pre.process({"eeg": eeg[:, s:s + step].copy(), "eog": eog[:, s:s + step].copy()})
        if r is not None:
            out.append(r["eeg"])
    return np.concatenate(out, axis=1)
