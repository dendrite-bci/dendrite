from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from dendrite.processing._types import Sample
from dendrite.processing.modes._metrics import SynchronousMetrics
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import Buffer, extract_event_code
from dendrite.utils.state_keys import mode_metric_key

DEFAULT_TRAINING_INTERVAL = 10


@dataclass
class MetricsPayload:
    """Performance metrics for GUI."""

    accuracy: float = 0.0
    confidence: float = 0.0
    chance_level: float | None = None
    adaptive_chance_level: float | None = None
    cohens_kappa: float | None = None


@dataclass
class ERPPayload:
    """Epoch data for ERP visualization."""

    event_type: str | int = 0
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    start_offset_ms: float = 0.0  # Epoch start relative to stimulus (negative = pre-stim)
    sample_rate: float = 500.0  # For time-to-sample conversion


@dataclass
class SyncPrediction:
    """Trial prediction with ground truth."""

    prediction: int = 0  # Original event code from paradigm configuration
    event_name: str = ""
    true_event: str = ""
    confidence: float = 0.0


class SynchronousMode(BaseMode):
    """
    Synchronous Mode for Dendrite Systems.

    This mode focuses on dataset-based training:
    - Collects data in epochs triggered by events
    - Stores all epochs in a dataset
    - Trains decoders on the full dataset every N epochs
    - Provides real-time performance metrics
    """

    MODE_TYPE = "synchronous"

    metrics_manager: SynchronousMetrics | None  # narrowed from BaseMode
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
        """Initialize SynchronousMode with validated instance configuration."""
        # Call parent constructor - extracts name, channel_selection automatically
        super().__init__(
            output_queue=output_queue,
            stop_event=stop_event,
            prediction_queue=prediction_queue,
            instance_config=instance_config,
            shared_state=shared_state,
            training_queue=training_queue,
        )

        self.decoder_config = instance_config.get("decoder_config", {})

        self.file_identifier = instance_config.get("file_identifier")

        self.training_interval = instance_config.get("training_interval", DEFAULT_TRAINING_INTERVAL)

        # Calculate epoch timing (recalculated in _initialize_mode with effective rate)
        self.epoch_tmin = instance_config.get("epoch_tmin", 0.0)
        self.epoch_tmax = instance_config.get("epoch_tmax", 2.0)
        self.tmin_samples = int(self.epoch_tmin * self.sample_rate)
        self.tmax_samples = int(self.epoch_tmax * self.sample_rate)
        self.epoch_length_samples = self.tmax_samples - self.tmin_samples

        # Epoch tracking
        self.epoch_count = 0

        # Simple epoch tracking
        self.pending_epochs = []
        self.current_sample_index = 0

        self._training_pending = False

    def _validate_configuration(self) -> bool:
        """Validate synchronous mode configuration."""
        if self.epoch_length_samples <= 0:
            self.logger.error(
                f"Epoch length must be positive. "
                f"Calculated: {self.epoch_length_samples} samples from "
                f"epoch_tmin={self.epoch_tmin}, epoch_tmax={self.epoch_tmax}, "
                f"sample_rate={self.sample_rate}"
            )
            return False

        if not self.event_mapping:
            self.logger.error("Event mapping is required for synchronous mode")
            return False

        if not self.label_mapping:
            self.logger.error("Label mapping is required for synchronous mode")
            return False

        return True

    def _calculate_sync_buffer_size(self, safety_factor: float = 2.0) -> int:
        """Calculate appropriate buffer size for synchronous mode epochs."""
        pre_event = abs(self.tmin_samples) if self.tmin_samples < 0 else 0
        min_buffer = pre_event + self.epoch_length_samples
        recommended = int(min_buffer * safety_factor)

        self.logger.info(
            f"Buffer size: {recommended} samples (epoch={self.epoch_length_samples}, safety={safety_factor}x)"
        )
        return recommended

    def _initialize_mode(self) -> bool:
        """Initialize synchronous mode components."""
        try:
            # Setup per-mode preprocessing (sets self.effective_sample_rate)
            self._setup_preprocessor()

            # Recalculate epoch timing with effective sample rate
            self.tmin_samples = int(self.epoch_tmin * self.effective_sample_rate)
            self.tmax_samples = int(self.epoch_tmax * self.effective_sample_rate)
            self.epoch_length_samples = self.tmax_samples - self.tmin_samples
            self.logger.info(f"Epoch: {self.epoch_length_samples} samples at {self.effective_sample_rate}Hz")

            # Setup unified buffer with appropriate size for epochs + pre-event data
            buffer_size = self._calculate_sync_buffer_size(safety_factor=2.0)
            self._setup_buffer(buffer_size)

            num_classes = len(self.label_mapping) if self.label_mapping else 2
            self._setup_metrics_manager(num_classes=num_classes, mode_type="synchronous")

            # Create decoder
            if self.modalities:
                decoder = self._create_decoder(self.decoder_config)
                if decoder:
                    decoder.event_mapping = self.event_mapping
                    decoder.label_mapping = self.label_mapping
            if not self.is_decoder_ready:
                self.logger.info("Decoder creation deferred until modalities are detected")

            if not self.training_queue:
                self.logger.warning("No training_queue — training will be skipped")
            else:
                self.logger.info("Training via web service (training_queue connected)")

            self.logger.info("SynchronousMode initialized successfully")
            self.logger.info(f"Training interval: {self.training_interval} epochs")
            self.logger.info(f"Label mapping: {self.label_mapping}")

            # Log optional features
            model_config = self.decoder_config.get("model_config", {})
            if model_config.get("use_augmentation", False):
                self.logger.info(
                    f"Data augmentation: {model_config.get('aug_strategy', 'moderate')} strategy"
                )

            # Auto-train on existing session data (late-join scenario)
            if self.training_queue and self.event_mapping and not self.is_decoder_ready:
                self.logger.info("Auto-training on existing session data (late start)")
                self._trigger_training()

            return True

        except Exception as e:
            self.logger.error(f"Error initializing mode: {e}", exc_info=True)
            return False

    def _run_main_loop(self):
        """Run the main synchronous processing loop."""
        self.logger.info("Entering main event-driven loop")
        trained_decoder_key = f"{self.mode_name}:trained_decoder"

        while not self.stop_event.is_set():
            try:
                sample = self._get_next_sample()
                if sample is None:
                    continue
                self._process_data(sample)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

            # Poll for trained model ~1Hz (SharedState IPC is expensive at 500Hz)
            if self._training_pending and self.shared_state:
                if self.current_sample_index % max(1, int(self.effective_sample_rate)) == 0:
                    result = self.shared_state.get(trained_decoder_key)
                    if result:
                        self._start_model_load(result)

            # Check if background model load completed (non-blocking swap)
            if self._pending_decoder_load is not None:
                info = self._pending_decoder_load
                self._pending_decoder_load = None
                decoder = info.pop("_decoder", None)
                if decoder:
                    # Validate time-sample dimension matches inference epoch_length
                    _, _, expected_t = self._validate_decoder_channels(decoder)
                    if expected_t and expected_t != self.epoch_length_samples:
                        self.logger.error(
                            f"Rejecting model: trained n_times={expected_t} vs "
                            f"inference epoch_length={self.epoch_length_samples}"
                        )
                    else:
                        self.decoder = decoder
                        elapsed = info.get("elapsed", 0.0)
                        self._send_output({"training_s_point": float(elapsed)}, "training_point")
                        self._update_gpu_metrics()
                        self.logger.info(f"Model swap complete ({elapsed:.1f}s training)")

    def _start_model_load(self, result: dict):
        """Start background decoder load. Activation happens in main loop."""
        path = result.get("path", "")
        self._training_pending = False
        self._start_background_decoder_load(path, result)

    def _cleanup(self):
        """Cleanup synchronous mode resources."""

        if self.decoder:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                parts = [self.mode_name, self.file_identifier, timestamp]
                file_identifier = "_".join(p for p in parts if p)
                self.decoder.save(file_identifier)
                self.logger.info(f"Decoder saved: {file_identifier}")
            except Exception as e:
                self.logger.error(f"Error saving decoder: {e}")

        # Call parent cleanup
        super()._cleanup()

    def _process_data(self, sample: Sample):
        """Process a single data sample using unified buffer."""
        # Apply per-mode preprocessing (CAR, bandpass, downsample)
        processed = self._preprocess_sample(sample)
        if processed is None:
            return  # Accumulating for downsample
        sample = processed

        # Track LSL timestamp for payloads
        self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

        assert self.buffer is not None, "_setup_buffer must run before _process_data"
        self.buffer.add_sample(sample)
        self.current_sample_index += 1

        # Update internal latency metric ~1Hz (telemetry polls at 1Hz)
        if self.current_sample_index % max(1, int(self.effective_sample_rate)) == 0:
            self._compute_and_store_internal_latency()

        # Extract event marker and check if it's a trigger
        event_code = extract_event_code(sample)
        if event_code != -1 and event_code in self.event_mapping:
            self._handle_event_trigger(event_code)

        # Check for ready pending epochs
        self._check_pending_epochs()

    def _handle_event_trigger(self, event_code: int):
        """Handle an event trigger by scheduling epoch extraction."""
        event_type = self.event_mapping[event_code]

        if event_type not in self.label_mapping:
            return

        class_index = self.label_mapping[event_type]
        self.logger.info(f"Event: {event_type} (class_index={class_index})")

        event_sample_index = self.current_sample_index - 1  # Event was at previous sample

        # Schedule epoch extraction after post-event data is collected
        if self.tmax_samples > 0:
            self.pending_epochs.append(
                (
                    event_sample_index,  # event position
                    event_code,
                    event_type,
                    class_index,
                    self.tmax_samples,  # samples to wait
                )
            )
        else:
            # Extract immediately if no post-event data needed
            self._extract_epoch(event_code, event_type, class_index, 0)

    def _check_pending_epochs(self):
        """Check if any pending epochs are ready for extraction."""
        still_pending = []
        for event_pos, event_code, event_type, class_index, samples_needed in self.pending_epochs:
            samples_elapsed = self.current_sample_index - event_pos
            if samples_elapsed >= samples_needed:
                self._extract_epoch(event_code, event_type, class_index, samples_elapsed)
            else:
                still_pending.append(
                    (event_pos, event_code, event_type, class_index, samples_needed)
                )
        self.pending_epochs = still_pending

    def _extract_epoch(
        self, event_code: int, event_type: str, class_index: int, delay_samples: int
    ):
        """Extract and process epoch from buffer."""
        if self.buffer is None:
            return
        X_input = self.buffer.extract_epoch_at_event(
            start_offset_samples=self.tmin_samples,
            epoch_length_samples=self.epoch_length_samples,
            event_position_from_end=delay_samples,
        )

        if X_input:
            self.epoch_count += 1
            self._process_extracted_epoch(X_input, event_code, event_type, class_index)
        else:
            self.logger.warning(f"Failed to extract epoch for {event_type}")

    def _process_extracted_epoch(
        self, X_input: dict, event_code: int, event_type: str, class_index: int
    ):
        """Process an extracted epoch."""
        self.logger.info(f"Epoch {self.epoch_count}: {event_type} (class_index={class_index})")

        try:
            # Send ERP data for visualization (downsampled to ~125 Hz, all modalities)
            viz_rate = 125.0
            factor = max(1, int(self.effective_sample_rate / viz_rate))
            for _, data in X_input.items():
                if data.ndim != 2:
                    continue
                data_ds = data[:, ::factor] if factor > 1 else data
                erp_payload = ERPPayload(
                    event_type=event_type,
                    data=data_ds,
                    start_offset_ms=self.epoch_tmin * 1000,
                    sample_rate=float(self.effective_sample_rate / factor),
                )
                self._send_output(erp_payload, "erp", queue="main")

            # Check if training is needed
            if self.epoch_count % self.training_interval == 0:
                self._trigger_training()

            # Make prediction if model is ready
            if self.is_decoder_ready:
                prediction, confidence, _ = self._predict(X_input)
                self._update_gpu_metrics()
                self._update_metrics_and_send(prediction, confidence, class_index, event_type)
            elif not self.decoder:
                self.logger.warning("No decoder available for prediction")

        except Exception as e:
            self.logger.error(f"Error processing epoch: {e}", exc_info=True)

    def _trigger_training(self):
        """Trigger model training via web service.

        Sends lightweight config to training_queue — MLService pulls epoch
        data from the recording file, no large arrays through the queue.
        """
        self.logger.info(
            f"Training triggered: epoch {self.epoch_count}, interval {self.training_interval}"
        )

        if not self.training_queue:
            self.logger.warning("No training_queue available")
            return

        # Build channel indices from channel_selection
        channel_indices = None
        modality = self._get_primary_modality()
        if self.channel_selection and modality in self.channel_selection:
            channel_indices = self.channel_selection[modality]

        # Get effective bad channels + labels for training interpolation
        effective_bad = {}
        channel_labels = {}
        if self.shared_state:
            quality = self.shared_state.get("channel_quality") or {}
            effective_bad = quality.get("effective_bad", {})
        rb_labels = (self._rb_config or {}).get("modality_labels", {})
        if rb_labels:
            channel_labels = dict(rb_labels)

        try:
            self.training_queue.put_nowait({
                "mode_name": self.mode_name,
                "decoder_config": self.decoder_config,
                "modalities": self.modalities,
                "event_mapping": dict(self.event_mapping),
                "label_mapping": dict(self.label_mapping),
                "include_background": self.instance_config.get("include_background", False),
                "epoch_tmin": self.epoch_tmin,
                "epoch_tmax": self.epoch_tmax,
                "channel_indices": channel_indices,
                "mode_preprocessing": self.instance_config.get("mode_preprocessing", {}),
                "sample_rate": self.sample_rate,
                "effective_sample_rate": self.effective_sample_rate,
                "epoch_length_samples": self.epoch_length_samples,
                "effective_bad": effective_bad,
                "channel_labels": channel_labels,
                "use_study_history": self.instance_config.get("use_study_history", False),
                "study_history_recording_ids": self.instance_config.get("study_history_recording_ids"),
                "study_name": self.instance_config.get("study_name"),
                "file_identifier": self.file_identifier,
            })
            self._training_pending = True
            self.logger.info("Training request sent to service")
        except Exception:
            self.logger.warning("Training queue full, skipping")

    def _update_metrics_and_send(
        self, prediction: int, confidence: float, class_index: int, event_type: str
    ):
        """Update metrics and send performance updates."""
        current_metrics = {}
        if self.metrics_manager:
            self.metrics_manager.add_prediction(prediction=prediction, true_label=class_index)
            current_metrics = self.metrics_manager.get_all_metrics()

        # Prepare payloads
        predicted_event_name = self.reverse_label_mapping.get(prediction, f"Class_{prediction}")
        accuracy = current_metrics.get("prequential_accuracy", 0.0)

        # Push accuracy to SharedState for telemetry widget
        if self.shared_state:
            self.shared_state.set(mode_metric_key(self.mode_name, "accuracy"), accuracy)

        # Send outputs
        metrics_payload = MetricsPayload(
            accuracy=accuracy,
            confidence=confidence,
            chance_level=current_metrics.get("chance_level"),
            adaptive_chance_level=current_metrics.get("adaptive_chance_level"),
            cohens_kappa=current_metrics.get("cohens_kappa"),
        )

        event_code = self.index_to_event_code.get(prediction, prediction)
        prediction_payload = SyncPrediction(
            prediction=event_code,
            event_name=predicted_event_name,
            true_event=event_type,
            confidence=confidence,
        )

        self._send_output(metrics_payload, "performance", queue="main")
        self._send_output(prediction_payload, "prediction", queue="prediction")

        # Log results
        is_correct = prediction == class_index
        self.logger.info(
            f"{predicted_event_name} vs {event_type}: "
            f"{'CORRECT' if is_correct else 'WRONG'} (conf={confidence:.3f}, acc={accuracy:.3f})"
        )

