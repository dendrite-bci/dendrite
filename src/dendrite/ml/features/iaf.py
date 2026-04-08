"""Individual Alpha Frequency (IAF) detection.

Computes IAF via Center of Gravity (CoG = Σ(f·P)/ΣP), which is more robust
than peak detection (Klimesch 1990, Corcoran 2018). Provides band-shifting
to center alpha-related bands on the detected IAF.
"""

import numpy as np
from scipy import signal


def compute_iaf(
    data: np.ndarray,
    fs: float,
    iaf_range: tuple[float, float] = (7.0, 14.0),
    nperseg: int | None = None,
) -> float:
    """Compute Individual Alpha Frequency via Center of Gravity.

    CoG = Σ(f · P(f)) / Σ(P(f)) within iaf_range, averaged across channels.

    Args:
        data: (n_channels, n_samples) preprocessed EEG data.
        fs: Sampling rate in Hz.
        iaf_range: (low, high) Hz frequency range for CoG search.
        nperseg: Welch segment length. Defaults to min(n_samples, fs/0.5).

    Returns:
        IAF in Hz. Falls back to midpoint of iaf_range if no power is found.
    """
    if nperseg is None:
        nperseg = min(data.shape[-1], int(fs / 0.5))
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    psd_mean = psd.mean(axis=0)
    mask = (freqs >= iaf_range[0]) & (freqs <= iaf_range[1])
    total_power = psd_mean[mask].sum()
    if total_power <= 0:
        return (iaf_range[0] + iaf_range[1]) / 2.0
    return float(np.sum(freqs[mask] * psd_mean[mask]) / total_power)


def shift_bands(
    target_bands: dict[str, list[float]],
    iaf: float,
    iaf_range: tuple[float, float] = (7.0, 14.0),
    canonical_center: float = 10.0,
) -> dict[str, list[float]]:
    """Shift bands overlapping iaf_range by the IAF offset.

    Bands whose frequency interval intersects iaf_range are shifted by
    (iaf - canonical_center). Non-overlapping bands are returned unchanged.
    """
    offset = iaf - canonical_center
    shifted = {}
    for name, (low, high) in target_bands.items():
        if low < iaf_range[1] and high > iaf_range[0]:
            shifted[name] = [max(0.0, low + offset), high + offset]
        else:
            shifted[name] = [low, high]
    return shifted
