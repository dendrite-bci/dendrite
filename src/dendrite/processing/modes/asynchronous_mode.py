import os
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any

from dendrite.constants import LOG_INTERVAL_SAMPLES, get_study_paths
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import extract_event_code, get_shared_model_path
from dendrite.utils.state_keys import mode_metric_key

# Async mode constants
MODEL_CHECK_INTERVAL_SEC = 5
LOG_PREDICTION_INTERVAL = 500
LOG_METRICS_INTERVAL = 50


@dataclass
class AsyncPrediction:
    """Continuous prediction output."""

    prediction: int = 0  # Original event code from paradigm configuration
    event_name: str = ""
    confidence: float = 0.0


@dataclass
class AsyncMetrics:
    """Continuous mode metrics for GUI."""

    prediction: int = 0  # Original event code from paradigm configuration
    confidence: float = 0.0
    event_name: str = ""
    true_label: int | None = None
    balanced_accuracy: float = 0.0


class AsynchronousMode(BaseMode):
    """
    Asynchronous (Continuous) Mode for Dendrite Systems.

    Inference-only mode that processes data in continuous sliding windows:
    - Loads pre-trained model for inference
    - Maintains sliding window buffers for each modality
    - Triggers predictions at regular intervals
    - Auto-detects events for evaluation when available
    """

    MODE_TYPE = "asynchronous"

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
        """Initialize AsynchronousMode with validated instance configuration."""
        super().__init__(
            data_queue=data_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            sample_rate=sample_rate,
            prediction_queue=prediction_queue,
            instance_config=instance_config,
            shared_state=shared_state,
        )

        self.decoder_config = instance_config.get("decoder_config", {})

        # Decoder source configuration
        self.decoder_source = instance_config.get("decoder_source", "pretrained")
        self.source_sync_mode = instance_config.get("source_sync_mode", "")
        self.study_name = instance_config.get("study_name", "default_study")
        self.file_identifier = instance_config.get("file_identifier", "")
        self.last_model_check_time = 0
        self.model_check_interval = MODEL_CHECK_INTERVAL_SEC
        self.last_model_modification_time = 0

        # Prediction timing configuration
        self.step_size_ms = instance_config.get("step_size_ms", 100)
        self.samples_per_prediction_step = max(
            1, int(self.sample_rate * (self.step_size_ms / 1000.0))
        )

        # Sliding window configuration
        self.window_length_sec = instance_config.get("window_length_sec", 1.0)
        self.epoch_length_samples = int(self.window_length_sec * self.sample_rate)

        # Evaluation configuration
        eval_config = instance_config.get("evaluation_config", {})
        self.background_class = eval_config.get("background_class", None)

        self.prediction_count = 0
        self.current_sample_index = 0
        self.decoder = None
        self._current_label = -1
        self._active_label = -1
        self._labeling_samples_remaining = 0

    def _validate_configuration(self) -> bool:
        """Validate asynchronous mode configuration."""
        if self.epoch_length_samples <= 0:
            self.logger.error(f"Epoch length must be positive. Got: {self.epoch_length_samples}")
            return False

        if self.samples_per_prediction_step <= 0:
            self.logger.error(
                f"Prediction step must be positive. Got: {self.samples_per_prediction_step}"
            )
            return False

        if not self.decoder_config:
            self.logger.error("Decoder config is required")
            return False

        if not self.channel_selection:
            self.logger.error("channel_selection is required for AsynchronousMode")
            return False

        return True

    def _initialize_mode(self) -> bool:
        """Initialize asynchronous mode components."""
        try:
            self._model_lock = threading.Lock()

            self._setup_buffer(self.epoch_length_samples)

            num_classes = len(self.label_mapping) if self.label_mapping else 2
            self.num_classes = num_classes
            self._setup_metrics_manager(
                num_classes=num_classes,
                mode_type="asynchronous",
                label_mapping=self.reverse_label_mapping,
                background_class=self.background_class,
            )

            # Create decoder and handle initial model loading
            if self.decoder_source == "external":
                self.logger.info("Decoder will be injected externally")
            else:
                self.decoder = self._create_decoder(self.decoder_config)
                if self.decoder:
                    if self.decoder_source == "sync_mode":
                        self.logger.info(
                            f"Will receive models from sync mode '{self.source_sync_mode}'"
                        )
                        self._check_for_model_updates()
                    else:
                        model_path = self.decoder_config.get("decoder_path", "")
                        if model_path:
                            self._load_model_from_path(model_path, self.decoder_source)
                        else:
                            self.logger.warning(
                                f"No model path specified for {self.decoder_source} source"
                            )

            self.logger.info("AsynchronousMode initialized successfully (inference-only)")
            self.logger.info(
                f"Configuration: epoch_length={self.epoch_length_samples} samples, "
                f"prediction_step={self.samples_per_prediction_step} samples ({self.step_size_ms}ms), "
                f"decoder_source={self.decoder_source}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error initializing mode: {e}", exc_info=True)
            return False

    def _run_main_loop(self):
        """Run the main asynchronous processing loop."""
        self.logger.info("Entering main continuous processing loop")

        data_received_count = 0

        while not self.stop_event.is_set():
            try:
                sample = self.data_queue.get(timeout=0.1)
                data_received_count += 1
                self._process_data(sample)

                if data_received_count % LOG_INTERVAL_SAMPLES == 0:
                    self.logger.info(f"Processed {data_received_count} data samples")

                if self.buffer.is_ready_for_step(self.samples_per_prediction_step):
                    self._trigger_prediction()

                self._check_for_model_updates()

            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

        self.logger.info(f"Exiting main loop. Processed {data_received_count} samples total")

    def _cleanup(self):
        """Cleanup asynchronous mode resources."""
        self.logger.info("Cleaning up AsynchronousMode")
        self.logger.info(f"Total predictions made: {self.prediction_count}")
        super()._cleanup()

    def _process_data(self, sample: dict):
        """Process incoming data sample using unified buffer and existing event labeling."""
        if self.stop_event.is_set():
            return

        if not isinstance(sample, dict):
            return

        self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

        if not self.decoder and self.decoder_source == "sync_mode":
            self._check_for_model_updates()

        event_code = extract_event_code(sample)

        # Handle event labeling: track ground truth labels for evaluation
        if event_code != -1 and event_code in self.event_mapping:
            event_type = self.event_mapping[event_code]
            if event_type in self.label_mapping:
                self._active_label = self.label_mapping[event_type]
                self._labeling_samples_remaining = self.epoch_length_samples

        if self._labeling_samples_remaining > 0:
            self._labeling_samples_remaining -= 1
            current_label = self._active_label
        else:
            current_label = -1

        if self._current_label == -1 and current_label >= 0:
            if self.metrics_manager:
                self.metrics_manager.register_event(self.current_sample_index, current_label)
        self._current_label = current_label

        self.buffer.add_sample(sample)
        self.current_sample_index += 1

    def _trigger_prediction(self):
        """Trigger a prediction using the current sliding window data."""
        if not self.decoder or not getattr(self.decoder, "is_fitted", False):
            return

        try:
            self._compute_and_store_internal_latency()

            X_input = self.buffer.extract_window()
            if not X_input:
                if self.prediction_count == 0:
                    self.logger.debug("No input data extracted from buffer")
                return

            if self.prediction_count == 0:
                self.logger.info("Model is now ready - starting predictions!")

            prediction, confidence, _ = self._predict(
                model=self.decoder, X_input=X_input, lock=self._model_lock, blocking=True
            )
            self._update_gpu_metrics()

            self._update_metrics_and_send(prediction, confidence)

            self.prediction_count += 1

            if self.prediction_count % LOG_PREDICTION_INTERVAL == 1:
                self.logger.debug(
                    f"Made prediction #{self.prediction_count}: {prediction} (confidence={confidence:.3f})"
                )

        except Exception as e:
            self.logger.error(f"Error in _trigger_prediction: {e}", exc_info=True)

    def _update_metrics_and_send(self, prediction, confidence):
        """Update metrics with trial-level evaluation."""
        if self.metrics_manager:
            self.metrics_manager.update_metrics(
                prediction=prediction,
                true_label=self._current_label,
                current_sample_idx=self.current_sample_index,
            )

        current_metrics = self.metrics_manager.get_current_metrics() if self.metrics_manager else {}

        self._send_prediction_output(prediction, confidence, current_metrics)

        if self.prediction_count % LOG_METRICS_INTERVAL == 1 and self._current_label >= 0:
            n_trials = current_metrics.get("n_trials", 0)
            balanced_acc = current_metrics.get("balanced_accuracy", 0.0)
            self.logger.info(
                f"EVENT - Pred: {int(prediction)} Label: {int(self._current_label)} "
                f"(conf={confidence:.3f}, acc={balanced_acc:.2f}, trials={n_trials})"
            )

    def _send_prediction_output(self, prediction, confidence, current_metrics):
        """Send prediction data to output queues."""
        event_code = self.index_to_event_code.get(prediction, prediction)

        prediction_payload = AsyncPrediction(
            prediction=event_code,
            event_name=self.reverse_label_mapping.get(prediction, str(int(prediction))),
            confidence=confidence,
        )
        self._send_output(prediction_payload, "prediction", queue="prediction")

        balanced_accuracy = current_metrics.get("balanced_accuracy", 0.0)
        if self.shared_state:
            self.shared_state.set(mode_metric_key(self.mode_name, "balanced_accuracy"), balanced_accuracy)

        true_label_code = (
            self.index_to_event_code.get(self._current_label, self._current_label)
            if self._current_label >= 0
            else -1
        )

        metrics_payload = AsyncMetrics(
            prediction=event_code,
            confidence=float(confidence),
            event_name=self.reverse_label_mapping.get(prediction, str(event_code)),
            true_label=true_label_code,
            balanced_accuracy=balanced_accuracy,
        )
        self._send_output(metrics_payload, "performance", queue="main")

    def _create_decoder(self, decoder_config: dict[str, Any]) -> Any | None:
        """Create decoder for AsynchronousMode with simplified num_classes logic."""
        if self.decoder_source == "pretrained":
            original_mapping = self.label_mapping
            self.label_mapping = {}
            decoder = super()._create_decoder(decoder_config)
            self.label_mapping = original_mapping
            return decoder

        return super()._create_decoder(decoder_config)

    def _load_model_from_path(self, model_path: str, model_type: str = "decoder") -> bool:
        """Load a decoder from the specified path with async-specific handling."""
        success = super()._load_model_from_path(model_path, model_type)

        if success:
            self._update_epoch_length_from_model()

            if not self._validate_channel_count():
                return False

        return success

    def _validate_channel_count(self) -> bool:
        """Validate that mode's channel count matches decoder's expected input."""
        if not hasattr(self.decoder, "input_shapes") or not self.decoder.input_shapes:
            return True

        for modality, channel_indices in self.channel_selection.items():
            if modality not in self.decoder.input_shapes:
                continue

            expected_shape = self.decoder.input_shapes[modality]
            expected_channels = expected_shape[0]
            actual_channels = len(channel_indices)

            if actual_channels != expected_channels:
                self.logger.error(
                    f"Channel count mismatch for {modality}: "
                    f"decoder expects {expected_channels} channels, mode has {actual_channels}. "
                    f"Configure mode to use the same {expected_channels} channels the decoder was trained on."
                )
                return False

        return True

    def _update_epoch_length_from_model(self):
        """Update epoch length based on model's input shapes if needed."""
        if not hasattr(self.decoder, "input_shapes") or not self.decoder.input_shapes:
            return

        if not self.channel_selection:
            return

        primary_modality = next(
            (mod for mod in self.channel_selection.keys() if mod in self.decoder.input_shapes), None
        )

        if not primary_modality:
            return

        model_shape = self.decoder.input_shapes[primary_modality]
        model_epoch_length = model_shape[1] if len(model_shape) > 1 else model_shape[0]

        if model_epoch_length != self.epoch_length_samples:
            self.logger.info(
                f"Updating epoch length: {self.epoch_length_samples} -> {model_epoch_length}"
            )
            self.epoch_length_samples = model_epoch_length
            self.window_length_sec = model_epoch_length / self.sample_rate
            self._setup_buffer(self.epoch_length_samples)
            self.samples_per_prediction_step = max(
                1, int(self.sample_rate * (self.step_size_ms / 1000.0))
            )

    def _check_for_model_updates(self):
        """Check for updated models from linked sync mode."""
        if self.decoder_source != "sync_mode" or not self.source_sync_mode:
            return

        current_time = time.time()
        if current_time - self.last_model_check_time < self.model_check_interval:
            return

        try:
            shared_model_path = get_shared_model_path(self.source_sync_mode, self.file_identifier)

            decoders_dir = get_study_paths(self.study_name)["decoders"]
            full_path = decoders_dir / shared_model_path
            json_path = f"{full_path}.json"
            if not os.path.exists(json_path):
                self.last_model_check_time = current_time
                return

            model_mtime = os.path.getmtime(json_path)

            if model_mtime > self.last_model_modification_time:
                self.logger.info(f"Loading updated model from {self.source_sync_mode}: {full_path}")
                self._load_updated_model(str(full_path))
                self.last_model_modification_time = model_mtime

            self.last_model_check_time = current_time

        except Exception as e:
            self.logger.error(f"Error checking for model updates: {e}", exc_info=True)

    def _load_updated_model(self, model_path):
        """Load an updated model from the specified path."""
        was_ready = getattr(self.decoder, "is_fitted", False) if self.decoder else False
        success = self._load_model_from_path(model_path, "updated")

        is_ready_now = getattr(self.decoder, "is_fitted", False) if self.decoder else False
        if success and not was_ready and is_ready_now:
            self.logger.info("Model is now ready for predictions!")
