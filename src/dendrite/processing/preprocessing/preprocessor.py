"""Config-driven signal preprocessor for any modality.

Behavior is fully determined by config keys — no per-modality subclasses.
A modality with no lowcut/highcut acts as passthrough (downsample only).
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import signal

from dendrite.processing.preprocessing.eog_correction import AdaptiveEOGRegression
from dendrite.utils.logger_central import get_logger

BACoeffs = tuple[np.ndarray, np.ndarray]

if TYPE_CHECKING:
    from dendrite.processing.preprocessing.interpolation import InterpolationApplicator


class ModalityProcessor:
    """Config-driven processor for any signal modality.

    Behavior determined entirely by config keys:
      lowcut / highcut / filter_order → bandpass filter (omit both = passthrough)
      line_freq / notch_width         → notch (bandstop) filter
      apply_rereferencing             → common average reference
      channel_labels                  → enables freeze_interpolation()
      downsample_factor               → anti-aliased decimation
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = get_logger()
        self.num_channels: int = config["num_channels"]
        self.sample_rate: float = config["sample_rate"]
        self.downsample_factor: int = config.get("downsample_factor", 1)
        self.apply_rereferencing: bool = config.get("apply_rereferencing", False)
        self.channel_labels: list[str] | None = config.get("channel_labels")
        self.montage_name: str = config.get("montage", "standard_1005")
        self._interpolator: InterpolationApplicator | None = None
        # Per-sample common-average subtracted by the last CAR (used to reference
        # the EOG to the EEG average for cross-modality EOG correction). Read via
        # the public ``last_ref_mean`` property.
        self._last_ref_mean: np.ndarray | None = None
        # Prime causal filters to steady-state on the first chunk so a large DC
        # offset doesn't produce a long startup transient (which would otherwise
        # poison the EOG-correction fit on the warmup window).
        self._primed = False
        # When EOG correction is active, the main EEG output is fully replaced by the
        # band-split recombination (see OnlinePreprocessor._apply_eog_correction), so
        # this processor's notch/bandpass are wasted — only its CAR mean (for EOG
        # referencing) and downsample (output rate/shape) are still used. The
        # OnlinePreprocessor toggles this public flag to skip those two filters.
        self.skip_band_filters = False

        nyquist = 0.5 * self.sample_rate

        # --- Bandpass filter ---
        self._has_bandpass = False
        lowcut = config.get("lowcut")
        highcut = config.get("highcut")
        if lowcut is not None and highcut is not None:
            order = config.get("filter_order", 4)
            self._bp_b, self._bp_a = cast(
                BACoeffs,
                signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band"),
            )
            self._bp_zi = self._make_zi(self._bp_b, self._bp_a)
            self._has_bandpass = True

        # --- Notch filter ---
        self._has_notch = False
        line_freq = config.get("line_freq")
        if line_freq is not None:
            notch_width = config.get("notch_width", 4)
            lo = (line_freq - notch_width / 2) / nyquist
            hi = (line_freq + notch_width / 2) / nyquist
            if 0 < lo < hi < 1:
                self._notch_b, self._notch_a = cast(
                    BACoeffs, signal.butter(4, [lo, hi], btype="bandstop")
                )
                self._notch_zi = self._make_zi(self._notch_b, self._notch_a)
                self._has_notch = True
            else:
                self.logger.warning(f"Invalid notch range for {line_freq} Hz — skipped")

        # --- Anti-aliasing filter for downsampling ---
        if self.downsample_factor > 1:
            output_nyquist = (self.sample_rate / self.downsample_factor) / 2
            normalized_cutoff = 0.8 * output_nyquist / (self.sample_rate / 2)
            self._aa_b, self._aa_a = cast(
                BACoeffs, signal.cheby1(8, 0.05, normalized_cutoff)
            )
            self._aa_zi = self._make_zi(self._aa_b, self._aa_a)

        self._ds_buffer = np.zeros((self.num_channels, 0))

        parts = []
        if self._has_notch:
            parts.append(f"notch@{line_freq}Hz")
        if self._has_bandpass:
            parts.append(f"bp {lowcut}-{highcut}Hz (order {config.get('filter_order', 4)})")
        if self.apply_rereferencing:
            parts.append("CAR")
        if self.downsample_factor > 1:
            parts.append(f"ds×{self.downsample_factor}")
        label = ", ".join(parts) or "passthrough"
        self.logger.info(f"Processor: {self.num_channels}ch — {label}")

    @property
    def last_ref_mean(self) -> np.ndarray | None:
        """Common-average subtracted by the most recent CAR (None if no CAR ran)."""
        return self._last_ref_mean

    @property
    def interpolator(self) -> "InterpolationApplicator | None":
        """The frozen bad-channel interpolator, if one has been computed."""
        return self._interpolator

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def freeze_interpolation(
        self,
        bad_indices: list[int],
        corr_matrix: np.ndarray | None = None,
        bad_during_warmup: list[int] | None = None,
    ) -> None:
        """Precompute interpolation matrix for bad channels.

        Prefers geometry-based spherical-spline interpolation (Perrin et al.
        1989) when channel labels resolve to standard montage positions; falls
        back to correlation-based weights from warmup data otherwise.  Safe to
        call on any modality — no-op when channel_labels is absent.
        """
        if not bad_indices or not self.channel_labels:
            return

        from dendrite.processing.preprocessing.interpolation import (
            CorrelationInterpolationMatrix,
            InterpolationApplicator,
            SplineInterpolationMatrix,
        )

        result = SplineInterpolationMatrix.compute(
            self.channel_labels, bad_indices, self.montage_name,
        )
        method = "spline"
        if result is None and corr_matrix is not None:
            result = CorrelationInterpolationMatrix.compute(
                self.channel_labels, bad_indices, corr_matrix, bad_during_warmup,
            )
            method = "correlation"

        if result is not None:
            self._interpolator = InterpolationApplicator(result)
            self.logger.info(
                f"Interpolation frozen ({method}): {len(bad_indices)} bad channels "
                f"({result.bad_labels}) from {len(result.good_indices)} good"
            )
        else:
            self.logger.warning("Interpolation matrix computation failed")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_chunk(
        self, data: np.ndarray, bad_channels: list[int] | None = None,
    ) -> np.ndarray:
        """Process a chunk: interpolate → CAR → notch → bandpass → downsample."""
        data = data.astype(np.float64)

        # 1. Interpolate bad channels (precomputed spline weights)
        if self._interpolator is not None:
            self._interpolator.apply(data)

        # 2. Common average reference (record the subtracted mean for EOG referencing)
        self._last_ref_mean = None
        if self.apply_rereferencing and data.shape[0] > 1:
            if bad_channels and self._interpolator is None:
                good_mask = np.ones(data.shape[0], dtype=bool)
                good_mask[bad_channels] = False
                if np.any(good_mask):
                    ref = np.mean(data[good_mask, :], axis=0, keepdims=True)
                    data -= ref
                    self._last_ref_mean = ref
            else:
                ref = np.mean(data, axis=0, keepdims=True)
                data -= ref
                self._last_ref_mean = ref

        prime = not self._primed and data.shape[1] > 0

        # 3. Notch filter (power-line removal)
        if self._has_notch and not self.skip_band_filters:
            if prime:
                self._notch_zi = self._prime_zi(self._notch_b, self._notch_a, data[:, 0])
            data, self._notch_zi = signal.lfilter(
                self._notch_b, self._notch_a, data, axis=1, zi=self._notch_zi,
            )

        # 4. Bandpass filter
        if self._has_bandpass and not self.skip_band_filters:
            if prime:
                self._bp_zi = self._prime_zi(self._bp_b, self._bp_a, data[:, 0])
            data, self._bp_zi = signal.lfilter(
                self._bp_b, self._bp_a, data, axis=1, zi=self._bp_zi,
            )

        # 5. Anti-aliased downsampling
        if self.downsample_factor > 1:
            data = self._downsample(data, prime=prime)

        # Priming is done once the filters have actually been primed on a non-empty
        # input — keyed off ``prime`` (input length), NOT the post-downsample output
        # length, which can be empty for a short first chunk and would otherwise leave
        # the filters un-primed and re-prime (clobber their state) on the next chunk.
        if prime:
            self._primed = True
        return data

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all filter states and interpolator."""
        if self._has_bandpass:
            self._bp_zi = self._make_zi(self._bp_b, self._bp_a)
        if self._has_notch:
            self._notch_zi = self._make_zi(self._notch_b, self._notch_a)
        if self.downsample_factor > 1:
            self._aa_zi = self._make_zi(self._aa_b, self._aa_a)
        self._ds_buffer = np.zeros((self.num_channels, 0))
        self._interpolator = None
        self._last_ref_mean = None
        self._primed = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_zi(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.zeros((self.num_channels, max(len(a), len(b)) - 1))

    def _prime_zi(self, b: np.ndarray, a: np.ndarray, first_col: np.ndarray) -> np.ndarray:
        """Steady-state filter state for each channel held at its first value."""
        return np.outer(first_col, signal.lfilter_zi(b, a))

    def _downsample(self, data: np.ndarray, prime: bool = False) -> np.ndarray:
        """Stateful anti-aliased decimation (Chebyshev Type I + stride)."""
        if prime and data.shape[1] > 0:
            self._aa_zi = self._prime_zi(self._aa_b, self._aa_a, data[:, 0])
        data, self._aa_zi = signal.lfilter(
            self._aa_b, self._aa_a, data, axis=1, zi=self._aa_zi,
        )
        self._ds_buffer = np.concatenate([self._ds_buffer, data], axis=1)

        n_out = self._ds_buffer.shape[1] // self.downsample_factor
        if n_out == 0:
            return np.zeros((self.num_channels, 0))

        output = self._ds_buffer[:, :: self.downsample_factor][:, :n_out]
        consumed = n_out * self.downsample_factor
        self._ds_buffer = self._ds_buffer[:, consumed:]
        return output


# Crossover between the corrected ocular band [lowcut, this] and the untouched high
# band [this, highcut]. Set below the alpha/mu band (8-12 Hz) so the band-split's
# crossover seam doesn't distort it — blink power is concentrated <6 Hz anyway.
_EOG_REF_HIGHCUT = 6.0


class OnlinePreprocessor:
    """Multi-modality preprocessor — routes data to per-modality processors.

    When EEG enables ``apply_eog_correction`` and EOG data is present in the streamed
    chunks, regresses ocular artifact out of the EEG as the final step. Correction is
    configured by the EEG flag alone — no ``eog`` config entry is required; the
    reference is taken from whatever EOG channels the stream supplies, and the
    estimator is wired lazily on the first EOG chunk (channel count from the data).
    The regression runs on a CAR-referenced, EEG-phase-matched, ocular-band
    **dedicated** EOG reference (not the EOG modality's own output), and is fit
    **adaptively** — covariance accumulators update per chunk and B refits
    periodically — so it converges to a stable fit over the first minutes. Online and
    offline (streamed) runs share this code path, so the adaptive B(t) follows the
    same trajectory (refit timing can differ by at most one chunk boundary).
    """

    def __init__(self, modality_preprocessing: dict[str, dict]) -> None:
        self.logger = get_logger()
        cfg = {m.lower(): c for m, c in modality_preprocessing.items()}
        self.processors: dict[str, ModalityProcessor] = {
            modality: ModalityProcessor(config) for modality, config in cfg.items()
        }

        # --- EOG correction setup (cross-modality) ---
        # Correction is configured by the EEG `apply_eog_correction` flag alone; the
        # reference is taken from whatever EOG channels the stream supplies at runtime
        # (no `eog` config entry needed — its filter params are irrelevant here, the
        # reference band is derived from the EEG ocular band). The two EEG band-split
        # processors are built now (they only need the EEG channel count); the reference
        # processor and estimator are built lazily on the first EOG chunk, once the EOG
        # channel count is known — see _ensure_eog_estimator.
        self._eog_ref_proc: ModalityProcessor | None = None
        self._eog_eeg_low_proc: ModalityProcessor | None = None
        self._eog_eeg_high_proc: ModalityProcessor | None = None
        self._eog_estimator: AdaptiveEOGRegression | None = None
        self._eog_ref_cfg: dict | None = None   # ref-proc config minus num_channels
        self._eog_n_eeg = 0
        self._eog_eff_rate = 0.0
        eeg_cfg = cfg.get("eeg")
        eog_low = (eeg_cfg or {}).get("lowcut", 0.5)
        # Crossover between the corrected ocular band and the untouched high band;
        # per-study override via `eog_crossover_hz`, else the _EOG_REF_HIGHCUT default.
        crossover = (eeg_cfg or {}).get("eog_crossover_hz") or _EOG_REF_HIGHCUT
        if eeg_cfg and eeg_cfg.get("apply_eog_correction") and (
            eog_low is None or eog_low < crossover
        ):
            # Split EEG into a phase-matched ocular low band [lowcut, crossover]
            # and a high band [crossover, highcut]. The regression is fit AND
            # applied entirely in the low band (EEG + reference filtered identically →
            # same causal phase, so the subtraction cancels the ocular rather than
            # time-shifting it), then recombined with the untouched high band.
            low = eog_low if eog_low is not None else 0.5
            high = eeg_cfg.get("highcut", 50.0) or 50.0
            order = eeg_cfg.get("filter_order", 4)
            sr = eeg_cfg["sample_rate"]
            ds = eeg_cfg.get("downsample_factor", 1)
            n_eeg = eeg_cfg["num_channels"]
            # Low/fit-domain EEG + EOG reference: CAR + [lowcut, crossover].
            low_cfg = {"lowcut": low, "highcut": crossover, "filter_order": order,
                       "sample_rate": sr, "downsample_factor": ds}
            self._eog_eeg_low_proc = ModalityProcessor(
                {**low_cfg, "apply_rereferencing": True, "num_channels": n_eeg}
            )
            # High band: CAR + [crossover, highcut], passed through untouched.
            # The notch (if any) lives here — line frequency is always above the
            # ocular band, and this recombined output replaces the main EEG output.
            self._eog_eeg_high_proc = ModalityProcessor(
                {"lowcut": crossover, "highcut": high, "filter_order": order,
                 "sample_rate": sr, "downsample_factor": ds,
                 "line_freq": eeg_cfg.get("line_freq"),
                 "notch_width": eeg_cfg.get("notch_width", 4),
                 "apply_rereferencing": True, "num_channels": n_eeg}
            )
            # Deferred to first EOG chunk: the EOG reference processor and estimator.
            self._eog_ref_cfg = {**low_cfg, "apply_rereferencing": False}
            self._eog_n_eeg = n_eeg
            eeg_proc = self.processors["eeg"]
            self._eog_eff_rate = eeg_proc.sample_rate / max(eeg_proc.downsample_factor, 1)
        elif eeg_cfg and eeg_cfg.get("apply_eog_correction"):
            self.logger.info(
                f"EOG correction skipped: EEG high-pass {eog_low}Hz is at/above the "
                f"{crossover}Hz ocular band — no ocular content to regress out"
            )

    @property
    def eog_correction_enabled(self) -> bool:
        """True when EOG correction is configured (EEG flag + valid ocular band),
        even before the estimator is lazily built on the first EOG chunk."""
        return self._eog_eeg_low_proc is not None

    @property
    def eog_active(self) -> bool:
        """True once the adaptive estimator has been built (first EOG chunk seen)."""
        return self._eog_estimator is not None

    def _ensure_eog_estimator(self, n_eog: int) -> None:
        """Build the EOG reference processor and adaptive estimator on first EOG data.

        Correction is configured by the EEG ``apply_eog_correction`` flag; the EOG
        channel count is only known once data flows, so the reference processor and
        estimator are built here, once, on the first chunk that carries EOG. No-op if
        already built, not configured, or ``n_eog <= 0``.
        """
        if self._eog_estimator is not None or self._eog_ref_cfg is None or n_eog <= 0:
            return
        self._eog_ref_proc = ModalityProcessor({**self._eog_ref_cfg, "num_channels": n_eog})
        self._eog_estimator = AdaptiveEOGRegression(
            n_eeg=self._eog_n_eeg, n_eog=n_eog, sample_rate=self._eog_eff_rate,
        )
        self.logger.info(
            f"EOG correction engaged: {self._eog_n_eeg} EEG ch, {n_eog} EOG reference ch"
        )

    def process(
        self,
        data_dict: dict[str, np.ndarray],
        bad_channels: dict[str, list[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Process each modality, then regress EOG out of EEG as the final step."""
        result: dict[str, np.ndarray] = {}
        # When EOG correction is active AND EOG data is present this chunk, Phase 3
        # fully replaces the main EEG output with the band-split recombination, so the
        # main processor's notch/bandpass are dead compute (only its CAR mean and
        # downsample are still used) — skip them. If EOG is absent, Phase 3 bails and
        # the Phase 1 EEG *is* the output, so it must be fully filtered.
        eeg_proc = self.processors.get("eeg")
        eog_raw = data_dict.get("eog")
        # Lazily wire the estimator on the first EOG chunk (channel count from data).
        if self._eog_eeg_low_proc is not None and isinstance(eog_raw, np.ndarray) \
                and eog_raw.shape[0] > 0:
            self._ensure_eog_estimator(eog_raw.shape[0])
        eog_will_correct = self._eog_estimator is not None and isinstance(eog_raw, np.ndarray)
        if eeg_proc is not None:
            eeg_proc.skip_band_filters = eog_will_correct

        # Phase 1: process EEG (and any non-EOG modality) normally.
        for modality, data in data_dict.items():
            if modality == "eog":
                continue
            proc = self.processors.get(modality)
            if proc is None:
                result[modality] = (
                    data.astype(np.float64) if isinstance(data, np.ndarray) else data
                )
                continue
            mod_bad = (bad_channels or {}).get(modality)
            result[modality] = proc.process_chunk(data, bad_channels=mod_bad)

        # Phase 2: process the EOG modality's own output (for other consumers).
        car_mean = eeg_proc.last_ref_mean if eeg_proc is not None else None
        if isinstance(eog_raw, np.ndarray):
            proc = self.processors.get("eog")
            result["eog"] = (
                proc.process_chunk(eog_raw.astype(np.float64),
                                   bad_channels=(bad_channels or {}).get("eog"))
                if proc is not None else eog_raw.astype(np.float64)
            )

        # Phase 3: phase-matched ocular-band regression, recombined with the high band.
        self._apply_eog_correction(result, data_dict.get("eeg"), eog_raw, car_mean,
                                   (bad_channels or {}).get("eeg"))
        return result

    def _apply_eog_correction(
        self,
        result: dict[str, np.ndarray],
        eeg_raw: np.ndarray | None,
        eog_raw: np.ndarray | None,
        car_mean: np.ndarray | None,
        eeg_bad: list[int] | None,
    ) -> None:
        if (self._eog_ref_proc is None or self._eog_eeg_low_proc is None
                or self._eog_eeg_high_proc is None or self._eog_estimator is None):
            return
        if not isinstance(eog_raw, np.ndarray) or not isinstance(eeg_raw, np.ndarray):
            return
        eeg_full = result.get("eeg")
        if not isinstance(eeg_full, np.ndarray) or eeg_full.shape[1] == 0:
            return

        # Complementary bands of the CAR'd EEG: low = [lowcut, _EOG_REF_HIGHCUT] (fit +
        # corrected), high = [_EOG_REF_HIGHCUT, highcut] (untouched). The reference is the
        # EEG-referenced EOG in the SAME low band, so regression and subtraction are
        # phase-matched.
        eeg_in = eeg_raw.astype(np.float64)
        # Mirror the main EEG processor's bad-channel handling: when interpolation is
        # frozen there, interpolate the band-split input too (this recombined output
        # replaces the main output) and CAR over all channels; otherwise exclude the
        # bad channels from the band CARs.
        eeg_proc = self.processors.get("eeg")
        interpolator = eeg_proc.interpolator if eeg_proc is not None else None
        if interpolator is not None:
            interpolator.apply(eeg_in)
            eeg_bad = None
        eeg_low = self._eog_eeg_low_proc.process_chunk(eeg_in.copy(), bad_channels=eeg_bad)
        eeg_high = self._eog_eeg_high_proc.process_chunk(eeg_in, bad_channels=eeg_bad)
        eog_in = eog_raw.astype(np.float64)
        if car_mean is not None and car_mean.shape[1] == eog_in.shape[1]:
            eog_in = eog_in - car_mean  # reference EOG to the EEG common-average
        ref = self._eog_ref_proc.process_chunk(eog_in)
        if not (ref.shape[1] == eeg_low.shape[1] == eeg_high.shape[1] == eeg_full.shape[1]):
            # Can't regress this chunk (EEG/EOG length mismatch). The main processor
            # skipped its notch/bandpass (skip_band_filters), so result["eeg"] holds an
            # un-band-passed signal — recover the proper full-band output from the split
            # (low+high IS the full band) instead of returning the unfiltered one.
            if eeg_low.shape == eeg_high.shape == eeg_full.shape:
                result["eeg"] = eeg_low + eeg_high
            return

        result["eeg"] = self._eog_estimator.update_and_apply(eeg_low, eeg_high, ref)

    def reset_all_states(self) -> None:
        for proc in self.processors.values():
            proc.reset_state()
        for proc in (self._eog_ref_proc, self._eog_eeg_low_proc, self._eog_eeg_high_proc):
            if proc is not None:
                proc.reset_state()
        if self._eog_estimator is not None:
            self._eog_estimator.reset()
