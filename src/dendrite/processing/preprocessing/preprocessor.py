"""Config-driven signal preprocessor for any modality.

Behavior is fully determined by config keys — no per-modality subclasses.
A modality with no lowcut/highcut acts as passthrough (downsample only).
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import signal

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
        self._interpolator: InterpolationApplicator | None = None

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

        Uses correlation-based weights derived from warmup data.
        Safe to call on any modality — no-op when channel_labels or corr_matrix
        is absent.
        """
        if not bad_indices or not self.channel_labels or corr_matrix is None:
            return

        from dendrite.processing.preprocessing.interpolation import (
            CorrelationInterpolationMatrix,
            InterpolationApplicator,
        )

        result = CorrelationInterpolationMatrix.compute(
            self.channel_labels, bad_indices, corr_matrix, bad_during_warmup,
        )
        if result is not None:
            self._interpolator = InterpolationApplicator(result)
            self.logger.info(
                f"Interpolation frozen: {len(bad_indices)} bad channels "
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

        # 2. Common average reference
        if self.apply_rereferencing and data.shape[0] > 1:
            if bad_channels and self._interpolator is None:
                good_mask = np.ones(data.shape[0], dtype=bool)
                good_mask[bad_channels] = False
                if np.any(good_mask):
                    data -= np.mean(data[good_mask, :], axis=0, keepdims=True)
            else:
                data -= np.mean(data, axis=0, keepdims=True)

        # 3. Notch filter (power-line removal)
        if self._has_notch:
            data, self._notch_zi = signal.lfilter(
                self._notch_b, self._notch_a, data, axis=1, zi=self._notch_zi,
            )

        # 4. Bandpass filter
        if self._has_bandpass:
            data, self._bp_zi = signal.lfilter(
                self._bp_b, self._bp_a, data, axis=1, zi=self._bp_zi,
            )

        # 5. Anti-aliased downsampling
        if self.downsample_factor > 1:
            data = self._downsample(data)

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_zi(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.zeros((self.num_channels, max(len(a), len(b)) - 1))

    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """Stateful anti-aliased decimation (Chebyshev Type I + stride)."""
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


class OnlinePreprocessor:
    """Multi-modality preprocessor — routes data to per-modality processors."""

    def __init__(self, modality_preprocessing: dict[str, dict]) -> None:
        self.processors: dict[str, ModalityProcessor] = {}
        self.logger = get_logger()
        for modality, config in modality_preprocessing.items():
            self.processors[modality.lower()] = ModalityProcessor(config)

    def process(
        self,
        data_dict: dict[str, np.ndarray],
        bad_channels: dict[str, list[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Process each modality through its configured processor."""
        result: dict[str, np.ndarray] = {}
        for modality, data in data_dict.items():
            proc = self.processors.get(modality)
            if proc is None:
                result[modality] = (
                    data.astype(np.float64) if isinstance(data, np.ndarray) else data
                )
                continue
            mod_bad = (bad_channels or {}).get(modality)
            result[modality] = proc.process_chunk(data, bad_channels=mod_bad)
        return result

    def reset_all_states(self) -> None:
        for proc in self.processors.values():
            proc.reset_state()
