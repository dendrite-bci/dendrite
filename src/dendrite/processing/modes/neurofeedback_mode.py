from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dendrite.ml.features.iaf import compute_iaf, shift_bands
from dendrite.ml.features.transforms import BandPowerTransform
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import Buffer, extract_event_code


@dataclass
class BandPowerPayload:
    """Band power features per channel."""

    channel_powers: dict[str, dict[str, float]] = field(default_factory=dict)
    target_bands: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class IAFPayload:
    """Result of IAF baseline calibration."""

    iaf_hz: float
    offset_hz: float
    original_bands: dict[str, list[float]] = field(default_factory=dict)
    shifted_bands: dict[str, list[float]] = field(default_factory=dict)


class NeurofeedbackMode(BaseMode):
    """
    Neurofeedback Mode: Extracts band power from sliding windows.

    Features:
    - Welch's method-based band power calculation using BandPowerTransform
    - Single or multi-band extraction support
    - Configurable relative power normalization (default: enabled)
    - Fast processing for real-time applications
    """

    MODE_TYPE = "neurofeedback"

    buffer: Buffer | None

    def __init__(
        self,
        output_queue,
        stop_event,
        instance_config: dict[str, Any],
        prediction_queue=None,
        shared_state=None,
        training_queue=None,
    ):
        """Initialize NeurofeedbackMode with validated instance configuration."""
        super().__init__(
            output_queue=output_queue,
            stop_event=stop_event,
            instance_config=instance_config,
            prediction_queue=prediction_queue,
            shared_state=shared_state,
            training_queue=training_queue,
        )

        # Extract neurofeedback-specific configuration
        self.feature_config = instance_config.get("feature_config", {})

        # Configure cluster mode (average all selected channels into one output)
        self.use_cluster_mode = self.feature_config.get("use_cluster_mode", False)

        self.modality_name = self._get_primary_modality()
        self.channel_labels = self._resolve_selected_labels()

        # Configure target bands (multi-band or single band)
        self.target_bands = self.feature_config.get(
            "target_bands", {"default": self.feature_config.get("target_band", [8.0, 12.0])}
        )

        # Extract timing parameters from instance config
        self.window_length_sec = instance_config.get("window_length_sec", 1.0)
        self.step_size_ms = instance_config.get("step_size_ms", 250)
        self.window_step_sec = self.step_size_ms / 1000.0

        self.use_relative_power = self.feature_config.get("use_relative_power", True)

        self.window_length_samples = int(self.window_length_sec * self.sample_rate)
        self.window_step_samples = int(self.window_step_sec * self.sample_rate)

        # Transform will be initialized in _initialize_mode()
        self.band_power_transform: BandPowerTransform | None = None

        # Event handlers: map event_code → callback(sample)
        # Subclasses or init logic can register handlers here.
        self._event_handlers: dict[int, Any] = {}

        # IAF calibration (optional, gated on iaf_event_id)
        iaf_event = self.feature_config.get("iaf_event_id")
        self.iaf_event_id: int | None = int(iaf_event) if iaf_event is not None else None
        self.iaf_baseline_sec: float = float(self.feature_config.get("iaf_baseline_sec", 5.0))
        self.iaf_range: tuple[float, float] = self._resolve_iaf_range()
        self.iaf_state: str = "idle"  # "idle" | "collecting" | "done"
        self.iaf_baseline_buf: np.ndarray | None = None
        self.iaf_baseline_pos: int = 0
        self.iaf_baseline_samples: int = 0
        self.iaf_value: float | None = None

        if self.iaf_event_id is not None:
            self._event_handlers[self.iaf_event_id] = self._on_iaf_trigger

    def _resolve_selected_labels(self) -> list[str]:
        """Per-channel labels in buffered-data order.

        Buffered data has channel_selection applied upstream
        (mode_utils.py ModalityProcessor), so labels must be resolved
        through the same indices to stay aligned.
        """
        full = self.modality_labels.get(self.modality_name, [])
        sel = (self.channel_selection or {}).get(self.modality_name)
        if not sel:
            return list(full)
        if not full:
            return []
        try:
            return [full[i] for i in sel]
        except IndexError:
            self.logger.warning(
                f"channel_selection out of range for '{self.modality_name}' "
                f"(have {len(full)} labels, indices {sel})"
            )
            return []

    def _resolve_iaf_range(self) -> tuple[float, float]:
        """Resolve IAF search range: explicit config or derived from target_bands."""
        explicit = self.feature_config.get("iaf_range")
        if explicit:
            return (float(explicit[0]), float(explicit[1]))
        # Derive from bands overlapping the broad alpha region [6, 15] Hz
        ALPHA_LO, ALPHA_HI, PAD = 6.0, 15.0, 2.0
        lows, highs = [], []
        for low, high in self.target_bands.values():
            if low < ALPHA_HI and high > ALPHA_LO:
                lows.append(low)
                highs.append(high)
        if lows:
            return (max(1.0, min(lows) - PAD), min(30.0, max(highs) + PAD))
        return (7.0, 14.0)

    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        if self.window_length_samples <= 0:
            self.logger.error(f"Invalid window length: {self.window_length_samples}")
            return False

        if self.window_step_samples <= 0:
            self.window_step_samples = 1

        for band_name, band_range in self.target_bands.items():
            if len(band_range) != 2 or band_range[0] >= band_range[1]:
                self.logger.error(f"Invalid band '{band_name}': {band_range}")
                return False
            if band_range[1] > self.sample_rate / 2:
                self.logger.warning(f"Band '{band_name}' exceeds Nyquist frequency")

        return True

    def _initialize_mode(self) -> bool:
        """Initialize neurofeedback mode."""
        try:
            # Setup per-mode preprocessing (sets self.effective_sample_rate)
            self._setup_preprocessor()

            # Recalculate window sizes with effective sample rate
            self.window_length_samples = int(self.window_length_sec * self.effective_sample_rate)
            self.window_step_samples = int(self.window_step_sec * self.effective_sample_rate)

            self._setup_buffer(self.window_length_samples)

            # Calculate nperseg for adequate frequency resolution
            # Need at least 0.5 Hz resolution for narrow bands like SMR (13-15 Hz)
            min_nperseg = int(self.effective_sample_rate / 0.5)
            nperseg = min(min_nperseg, self.window_length_samples)
            freq_resolution = self.effective_sample_rate / nperseg

            # Initialize band power transform
            self.band_power_transform = BandPowerTransform(
                bands=self.target_bands,
                fs=self.effective_sample_rate,
                nperseg=nperseg,
                relative=self.use_relative_power,
            )
            self.band_power_transform.fit({})

            self.logger.info("NeurofeedbackMode initialized")
            if len(self.target_bands) > 1:
                self.logger.info(f"Multi-band mode: {len(self.target_bands)} bands")
                for name, band in self.target_bands.items():
                    self.logger.info(f"  Band {name}: {band[0]}-{band[1]} Hz")
            else:
                band_name, band_range = next(iter(self.target_bands.items()))
                self.logger.info(f"Single band '{band_name}': {band_range[0]}-{band_range[1]} Hz")

            self.logger.info(f"Window: {self.window_length_sec}s, Step: {self.step_size_ms}ms")
            self.logger.info(f"Power: {'Relative' if self.use_relative_power else 'Absolute'}")
            self.logger.info(f"Frequency resolution: {freq_resolution:.2f} Hz (nperseg={nperseg})")
            self.logger.info(f"Cluster mode: {'Enabled' if self.use_cluster_mode else 'Disabled'}")

            # IAF calibration setup
            if self.iaf_event_id is not None:
                self.iaf_baseline_samples = int(
                    self.iaf_baseline_sec * self.effective_sample_rate
                )
                self.logger.info(
                    f"IAF calibration armed: event={self.iaf_event_id}, "
                    f"baseline={self.iaf_baseline_sec}s ({self.iaf_baseline_samples} samples), "
                    f"range={self.iaf_range[0]}-{self.iaf_range[1]} Hz"
                )

            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _run_main_loop(self):
        """Main processing loop."""
        self.logger.info("Starting neurofeedback processing")

        while not self.stop_event.is_set():
            try:
                sample = self._get_next_sample()
                if sample is None:
                    continue

                # Apply per-mode preprocessing (CAR, bandpass, downsample)
                processed = self._preprocess_sample(sample)
                if processed is None:
                    continue  # Accumulating for downsample
                sample = processed

                # Track LSL timestamp for payloads
                self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

                # Generic event dispatch + per-sample hooks
                if self._event_handlers:
                    event_code = extract_event_code(sample)
                    if event_code != -1 and event_code in self._event_handlers:
                        self._event_handlers[event_code](sample)
                if self.iaf_state == "collecting":
                    self._accumulate_iaf_sample(sample)

                if self.buffer is None:
                    continue
                self.buffer.add_sample(sample)

                if self.buffer.is_ready_for_step(self.window_step_samples):
                    self._extract_and_send_features()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

    def _extract_and_send_features(self):
        """Extract band power features and send payload."""
        # Compute internal latency BEFORE feature extraction
        self._compute_and_store_internal_latency()

        if self.buffer is None:
            return
        X_input = self.buffer.extract_window()
        if not X_input:
            return

        signal = X_input.get(self.modality_name)
        if signal is None:
            return

        channel_powers = self._calculate_band_powers(signal)
        payload = BandPowerPayload(
            channel_powers=channel_powers,
            target_bands=self.target_bands,
        )

        self._send_output(payload, "neurofeedback", queue="prediction")
        self._send_output(payload, "neurofeedback_features", queue="main")

    def _calculate_band_powers(self, data: np.ndarray) -> dict[str, dict[str, float]]:
        """Calculate band power using BandPowerTransform with Welch's method."""
        if self.band_power_transform is None:
            return {}
        # Prepare data: (channels, times) → (batch=1, channels, times)
        # Channel selection is already applied by _preprocess_sample before buffering
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 1:
            data = data.reshape(1, 1, -1)
        n_channels = data.shape[1]

        X_input = {self.modality_name: data}
        band_features = self.band_power_transform.transform(X_input)
        feature_array = band_features[self.modality_name]

        band_names = list(self.target_bands.keys())
        channel_powers = {}

        for ch_idx in range(n_channels):
            channel_label = (
                self.channel_labels[ch_idx]
                if ch_idx < len(self.channel_labels)
                else f"ch{ch_idx}"
            )
            band_powers = {}

            for band_idx, band_name in enumerate(band_names):
                feature_idx = band_idx * n_channels + ch_idx
                band_powers[band_name] = float(feature_array[0, feature_idx])

            channel_powers[channel_label] = band_powers

        # If cluster mode enabled, average all channels into single output
        if self.use_cluster_mode and len(channel_powers) > 0:
            cluster_name = f"cluster_{self.modality_name}"
            cluster_powers = {}

            for band_name in band_names:
                # Average power across all channels for this band
                powers = [ch_data[band_name] for ch_data in channel_powers.values()]
                cluster_powers[band_name] = float(np.mean(powers))

            # Return only the cluster (single output)
            return {cluster_name: cluster_powers}

        # Return individual channel powers (default)
        return channel_powers

    # ---- IAF calibration ----

    def _on_iaf_trigger(self, sample: Mapping[str, Any]) -> None:
        """Event handler: begin collecting baseline data for IAF.

        Calibration is one-shot per session — re-triggers after completion
        are ignored so reward bands remain stable during a run.
        """
        if self.iaf_state != "idle":
            self.logger.debug(
                f"IAF trigger ignored (state={self.iaf_state})"
            )
            return
        data = sample.get(self.modality_name)
        n_channels = data.shape[0] if data is not None else 1
        self.iaf_baseline_buf = np.zeros(
            (n_channels, self.iaf_baseline_samples), dtype=np.float32
        )
        self.iaf_baseline_pos = 0
        self.iaf_state = "collecting"
        self.logger.info("IAF baseline collection started")

    def _accumulate_iaf_sample(self, sample: Mapping[str, Any]) -> None:
        """Append a (possibly multi-sample) chunk to the IAF baseline accumulator.

        Chunks can be wider than 1 when mode preprocessing downsamples, so
        copy all columns and clamp at the buffer boundary.
        """
        data = sample.get(self.modality_name)
        if data is None or self.iaf_baseline_buf is None:
            return
        remaining = self.iaf_baseline_samples - self.iaf_baseline_pos
        take = min(data.shape[1], remaining)
        end = self.iaf_baseline_pos + take
        self.iaf_baseline_buf[:, self.iaf_baseline_pos:end] = data[:, :take]
        self.iaf_baseline_pos = end
        if self.iaf_baseline_pos >= self.iaf_baseline_samples:
            self._finalize_iaf()

    def _finalize_iaf(self) -> None:
        """Compute IAF, shift bands, rebuild transform."""
        if self.iaf_baseline_buf is None:
            return
        iaf = compute_iaf(
            self.iaf_baseline_buf, self.effective_sample_rate, self.iaf_range
        )
        offset = iaf - 10.0
        shifted = shift_bands(self.target_bands, iaf, self.iaf_range)
        self.logger.info(
            f"IAF detected: {iaf:.2f} Hz (offset {offset:+.2f} Hz from canonical 10 Hz)"
        )
        for name, bands in shifted.items():
            if bands != self.target_bands[name]:
                self.logger.info(
                    f"  {name}: {self.target_bands[name]} → {bands}"
                )

        nyquist = self.effective_sample_rate / 2.0
        for name, bands in shifted.items():
            if bands[1] > nyquist:
                self.logger.warning(
                    f"Shifted band '{name}' high {bands[1]:.2f} Hz exceeds Nyquist {nyquist:.2f} Hz"
                )

        # Send result before updating bands
        self._send_output(
            IAFPayload(
                iaf_hz=round(iaf, 3),
                offset_hz=round(offset, 3),
                original_bands={k: list(v) for k, v in self.target_bands.items()},
                shifted_bands=shifted,
            ),
            "iaf_result",
            queue="main",
        )

        # Update bands and rebuild transform
        self.target_bands = shifted
        nperseg = min(
            int(self.effective_sample_rate / 0.5), self.window_length_samples
        )
        self.band_power_transform = BandPowerTransform(
            bands=self.target_bands,
            fs=self.effective_sample_rate,
            nperseg=nperseg,
            relative=self.use_relative_power,
        )
        self.band_power_transform.fit({})

        self.iaf_value = iaf
        self.iaf_state = "done"
        self.iaf_baseline_buf = None

    def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up NeurofeedbackMode")
        super()._cleanup()
