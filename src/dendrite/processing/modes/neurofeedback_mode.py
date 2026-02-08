from dataclasses import dataclass, field
from queue import Empty
from typing import Any

import numpy as np

from dendrite.ml.features.transforms import BandPowerTransform
from dendrite.processing.modes.base_mode import BaseMode


@dataclass
class BandPowerPayload:
    """Band power features per channel."""

    channel_powers: dict[str, dict[str, float]] = field(default_factory=dict)
    target_bands: dict[str, list[float]] = field(default_factory=dict)


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

    def __init__(
        self,
        data_queue,
        output_queue,
        stop_event,
        instance_config: dict[str, Any],
        sample_rate,
        prediction_queue=None,
        shared_state=None,
    ):
        """
        Initialize NeurofeedbackMode with validated instance configuration.

        Parameters:
        -----------
        instance_config : dict
            Pre-validated configuration dictionary containing all mode parameters
        """
        super().__init__(
            data_queue=data_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            instance_config=instance_config,
            sample_rate=sample_rate,
            prediction_queue=prediction_queue,
            shared_state=shared_state,
        )

        # Extract neurofeedback-specific configuration
        self.feature_config = instance_config.get("feature_config", {})

        # Configure cluster mode (average all selected channels into one output)
        self.use_cluster_mode = self.feature_config.get("use_cluster_mode", False)

        # Store modality name (already lowercase from base_mode normalization)
        self.modality_name = self.modalities[0] if self.modalities else "eeg"

        # Get channel labels from unified modality_labels (set by BaseMode)
        self.channel_labels = self.modality_labels.get(self.modality_name, [])

        # Extract selected channel indices for proper labeling after filtering
        # If channel_selection specifies indices, store for label mapping
        self.selected_channel_indices = None
        if self.channel_selection and self.modality_name in self.channel_selection:
            self.selected_channel_indices = self.channel_selection[self.modality_name]

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
        self.band_power_transform = None

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
            self._setup_buffer(self.window_length_samples)

            # Calculate nperseg for adequate frequency resolution
            # Need at least 0.5 Hz resolution for narrow bands like SMR (13-15 Hz)
            min_nperseg = int(self.sample_rate / 0.5)
            nperseg = min(min_nperseg, self.window_length_samples)
            freq_resolution = self.sample_rate / nperseg

            # Initialize band power transform
            self.band_power_transform = BandPowerTransform(
                bands=self.target_bands,
                fs=self.sample_rate,
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

            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _run_main_loop(self):
        """Main processing loop."""
        self.logger.info("Starting neurofeedback processing")

        while not self.stop_event.is_set():
            try:
                sample = self.data_queue.get(timeout=0.1)

                # Track LSL timestamp for payloads
                self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

                self.buffer.add_sample(sample)

                if self.buffer.is_ready_for_step(self.window_step_samples):
                    self._extract_and_send_features()

            except Empty:
                pass  # Queue timeout, continue loop
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

    def _extract_and_send_features(self):
        """Extract band power features and send payload."""
        # Compute internal latency BEFORE feature extraction
        self._compute_and_store_internal_latency()

        X_input = self.buffer.extract_window()
        if not X_input:
            return

        eeg_data = X_input.get(self.modality_name)
        if eeg_data is None:
            return

        channel_powers = self._calculate_band_powers(eeg_data)
        payload = BandPowerPayload(
            channel_powers=channel_powers,
            target_bands=self.target_bands,
        )

        self._send_output(payload, "neurofeedback", queue="prediction")
        self._send_output(payload, "neurofeedback_features", queue="main")

    def _calculate_band_powers(self, eeg_data: np.ndarray) -> dict[str, dict[str, float]]:
        """Calculate band power using BandPowerTransform with Welch's method."""
        # Prepare data: (channels, times) → (batch=1, channels, times)
        if eeg_data.ndim == 2:
            eeg_data = eeg_data[np.newaxis, :, :]
        elif eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, 1, -1)
        n_channels = eeg_data.shape[1]

        X_input = {self.modality_name: eeg_data}
        band_features = self.band_power_transform.transform(X_input)
        feature_array = band_features[self.modality_name]

        band_names = list(self.target_bands.keys())
        channel_powers = {}

        # Calculate per-channel band powers
        for ch_idx in range(n_channels):
            original_idx = (
                self.selected_channel_indices[ch_idx] if self.selected_channel_indices else ch_idx
            )
            channel_label = (
                self.channel_labels[original_idx]
                if self.channel_labels and original_idx < len(self.channel_labels)
                else f"ch{original_idx}"
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

    def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up NeurofeedbackMode")
        super()._cleanup()
