from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dendrite.ml.features.iaf import IAFCalibrator, IAFPayload
from dendrite.ml.features.transforms import BandPowerTransform
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import Buffer, extract_event_code


@dataclass
class BandPowerPayload:
    """Band power features per channel."""

    channel_powers: dict[str, dict[str, float]] = field(default_factory=dict)
    target_bands: dict[str, list[float]] = field(default_factory=dict)


class NeurofeedbackMode(BaseMode):
    """Neurofeedback Mode: extracts per-channel band power from sliding windows."""

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
        super().__init__(
            output_queue=output_queue,
            stop_event=stop_event,
            instance_config=instance_config,
            prediction_queue=prediction_queue,
            shared_state=shared_state,
            training_queue=training_queue,
        )

        self.feature_config = instance_config.get("feature_config", {})
        self.use_cluster_mode = self.feature_config.get("use_cluster_mode", False)

        self.modality_name = self._get_primary_modality()
        self.channel_labels = self._resolve_selected_labels()

        self.target_bands = self.feature_config.get(
            "target_bands", {"default": [8.0, 12.0]}
        )

        self.window_length_sec = instance_config.get("window_length_sec", 1.0)
        self.step_size_ms = instance_config.get("step_size_ms", 250)
        self.window_step_sec = self.step_size_ms / 1000.0

        self.use_relative_power = self.feature_config.get("use_relative_power", True)

        self.window_length_samples = int(self.window_length_sec * self.sample_rate)
        self.window_step_samples = int(self.window_step_sec * self.sample_rate)

        self.band_power_transform: BandPowerTransform | None = None

        self.iaf_baseline_sec = float(self.feature_config.get("iaf_baseline_sec", 5.0))
        self.iaf: IAFCalibrator | None = None  # built in _initialize_mode

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

    def _validate_configuration(self) -> bool:
        if self.window_length_samples <= 0:
            self.logger.error(f"Invalid window length: {self.window_length_samples}")
            return False

        for band_name, band_range in self.target_bands.items():
            if len(band_range) != 2 or band_range[0] >= band_range[1]:
                self.logger.error(f"Invalid band '{band_name}': {band_range}")
                return False
            if band_range[1] > self.sample_rate / 2:
                self.logger.warning(f"Band '{band_name}' exceeds Nyquist frequency")

        return True

    def _build_band_power_transform(self) -> tuple[BandPowerTransform, int]:
        # Need at least 0.5 Hz resolution for narrow bands like SMR (13-15 Hz)
        nperseg = min(int(self.effective_sample_rate / 0.5), self.window_length_samples)
        transform = BandPowerTransform(
            bands=self.target_bands,
            fs=self.effective_sample_rate,
            nperseg=nperseg,
            relative=self.use_relative_power,
        )
        transform.fit({})
        return transform, nperseg

    def _initialize_mode(self) -> bool:
        try:
            self._setup_preprocessor()

            self.window_length_samples = int(self.window_length_sec * self.effective_sample_rate)
            self.window_step_samples = int(self.window_step_sec * self.effective_sample_rate)

            self._setup_buffer(self.window_length_samples)

            self.band_power_transform, nperseg = self._build_band_power_transform()
            freq_resolution = self.effective_sample_rate / nperseg

            self.logger.info(
                f"NeurofeedbackMode: {len(self.target_bands)} band(s) {dict(self.target_bands)}"
            )
            self.logger.info(
                f"Window {self.window_length_sec}s, step {self.step_size_ms}ms, "
                f"power={'relative' if self.use_relative_power else 'absolute'}, "
                f"cluster={self.use_cluster_mode}, "
                f"freq_res {freq_resolution:.2f}Hz (nperseg={nperseg})"
            )

            iaf_event = self.feature_config.get("iaf_event_id")
            if iaf_event is not None:
                self.iaf = IAFCalibrator(
                    event_id=int(iaf_event),
                    baseline_samples=int(self.iaf_baseline_sec * self.effective_sample_rate),
                    iaf_range=tuple(self.feature_config.get("iaf_range", (7.0, 14.0))),
                )
                self.logger.info(
                    f"IAF calibration armed: event={self.iaf.event_id}, "
                    f"baseline={self.iaf_baseline_sec}s ({self.iaf.baseline_samples} samples), "
                    f"range={self.iaf.iaf_range[0]}-{self.iaf.iaf_range[1]} Hz"
                )

            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _run_main_loop(self):
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

                self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

                if self.iaf is not None:
                    data = sample.get(self.modality_name)
                    if data is not None:
                        if extract_event_code(sample) == self.iaf.event_id and self.iaf.trigger(
                            data.shape[0]
                        ):
                            self.logger.info("IAF baseline collection started")
                        if self.iaf.state == "collecting" and self.iaf.accumulate(data):
                            self._on_iaf_complete()

                if self.buffer is None:
                    continue
                self.buffer.add_sample(sample)

                if self.buffer.is_ready_for_step(self.window_step_samples):
                    self._extract_and_send_features()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

    def _extract_and_send_features(self):
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
        if self.band_power_transform is None:
            return {}
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

        if self.use_cluster_mode and len(channel_powers) > 0:
            cluster_name = f"cluster_{self.modality_name}"
            cluster_powers = {}
            for band_name in band_names:
                powers = [ch_data[band_name] for ch_data in channel_powers.values()]
                cluster_powers[band_name] = float(np.mean(powers))
            return {cluster_name: cluster_powers}

        return channel_powers

    def _on_iaf_complete(self) -> None:
        """Consume the calibrator's result: log, emit payload, rebuild transform."""
        assert self.iaf is not None
        result: IAFPayload | None = self.iaf.finalize(
            self.effective_sample_rate, self.target_bands
        )
        if result is None:
            self.logger.warning("IAF calibration failed — keeping canonical bands")
            return

        self.logger.info(
            f"IAF detected: {result.iaf_hz:.2f} Hz (CoG: {result.cog_hz:.2f}, "
            f"offset {result.offset_hz:+.2f} Hz from canonical 10 Hz)"
        )
        nyquist = self.effective_sample_rate / 2.0
        for name, bands in result.shifted_bands.items():
            if bands != self.target_bands[name]:
                self.logger.info(f"  {name}: {self.target_bands[name]} → {bands}")
            if bands[1] > nyquist:
                self.logger.warning(
                    f"Shifted band '{name}' high {bands[1]:.2f} Hz exceeds Nyquist {nyquist:.2f} Hz"
                )

        self._send_output(result, "iaf_result", queue="main")

        self.target_bands = result.shifted_bands
        self.band_power_transform, _ = self._build_band_power_transform()
