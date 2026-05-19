"""Individual Alpha Frequency (IAF) estimation.

Corcoran et al. (2018) Savitzky-Golay peak detection via philistine, plus
the one-shot baseline state machine (`IAFCalibrator`) and the wire payload
(`IAFPayload`) consumed by `NeurofeedbackMode`.

Reference:
    Corcoran, A. W., Alday, P. M., Schlesewsky, M., & Bornkessel-Schlesewsky, I.
    (2018). Toward a reliable, automated method of individual alpha frequency
    quantification. Psychophysiology, 55(7), e13064.
"""

import warnings
from dataclasses import dataclass, field

import mne
import numpy as np
from philistine.mne import savgol_iaf


def compute_iaf(
    data: np.ndarray,
    fs: float,
    iaf_range: tuple[float, float] = (7.0, 14.0),
) -> float:
    """Compute Individual Alpha Frequency via Corcoran 2018 (Sav-Gol peak).

    Wraps philistine's `savgol_iaf` on a synthetic `mne.Raw`. Any philistine
    failure (buffer too short for Sav-Gol window, pink-noise PSD, no peak in
    band, etc.) surfaces as a single `RuntimeError`.

    Args:
        data: (n_channels, n_samples) preprocessed EEG.
        fs: Sampling rate (Hz).
        iaf_range: (low, high) Hz alpha-band bounds passed as fmin/fmax.

    Returns:
        Peak Alpha Frequency in Hz.

    Raises:
        RuntimeError: when philistine cannot detect a peak in the band.
    """
    fmin, fmax = float(iaf_range[0]), float(iaf_range[1])
    n_channels, n_samples = data.shape[0], data.shape[-1]
    info = mne.create_info(
        ch_names=[f"EEG {i + 1}" for i in range(n_channels)],
        sfreq=fs,
        ch_types="eeg",
    )
    resolution = max(0.25, fs / n_samples)
    try:
        with mne.utils.use_log_level("ERROR"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
            result = savgol_iaf(
                raw, picks=None, fmin=fmin, fmax=fmax, resolution=resolution, ax=False
            )
    except Exception as e:
        raise RuntimeError(f"savgol_iaf failed: {e}") from e
    paf = result.PeakAlphaFrequency
    if paf is None or not np.isfinite(paf):
        raise RuntimeError(f"No detectable alpha peak in {fmin}–{fmax} Hz")
    return float(paf)


def shift_bands(
    target_bands: dict[str, list[float]],
    iaf: float,
    canonical_center: float = 10.0,
) -> dict[str, list[float]]:
    """Shift the band named "alpha" by (iaf - canonical_center). Others unchanged.

    IAF anchors the alpha peak only; SMR, theta, beta, etc. have separate
    generators and stay at their canonical ranges.
    """
    offset = iaf - canonical_center
    return {
        name: ([max(0.0, low + offset), high + offset]
               if name.lower() == "alpha" else [low, high])
        for name, (low, high) in target_bands.items()
    }


@dataclass
class IAFPayload:
    """Wire payload for IAF calibration result."""

    iaf_hz: float
    offset_hz: float
    original_bands: dict[str, list[float]] = field(default_factory=dict)
    shifted_bands: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class IAFCalibrator:
    """One-shot IAF baseline collection + estimation.

    State: idle → collecting → done. Caller calls `trigger()` on the event
    marker, `accumulate()` on each subsequent sample, and reads the bool
    return from `accumulate()` to know when to call `finalize()`.
    """

    event_id: int
    baseline_samples: int
    iaf_range: tuple[float, float]
    state: str = "idle"  # "idle" | "collecting" | "done"
    _buf: np.ndarray | None = field(default=None, init=False, repr=False)
    _pos: int = field(default=0, init=False, repr=False)

    def trigger(self, n_channels: int) -> bool:
        """Start collecting. Returns False if not idle (already done/collecting)."""
        if self.state != "idle":
            return False
        self._buf = np.zeros((n_channels, self.baseline_samples), dtype=np.float32)
        self._pos = 0
        self.state = "collecting"
        return True

    def accumulate(self, data: np.ndarray) -> bool:
        """Append a chunk. Returns True iff the buffer just filled.

        Chunks can be wider than 1 when mode preprocessing downsamples, so
        copy all columns and clamp at the buffer boundary.
        """
        if self._buf is None or self.state != "collecting":
            return False
        remaining = self.baseline_samples - self._pos
        take = min(data.shape[1], remaining)
        end = self._pos + take
        self._buf[:, self._pos:end] = data[:, :take]
        self._pos = end
        return self._pos >= self.baseline_samples

    def finalize(
        self, fs: float, target_bands: dict[str, list[float]]
    ) -> IAFPayload | None:
        """Compute IAF and shift bands. Returns None on detection failure."""
        if self._buf is None:
            return None
        try:
            iaf = compute_iaf(self._buf, fs, self.iaf_range)
        except RuntimeError:
            return None
        finally:
            self.state = "done"
            self._buf = None
        return IAFPayload(
            iaf_hz=round(iaf, 3),
            offset_hz=round(iaf - 10.0, 3),
            original_bands={k: list(v) for k, v in target_bands.items()},
            shifted_bands=shift_bands(target_bands, iaf),
        )
