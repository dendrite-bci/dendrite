import time
from dataclasses import dataclass
from typing import Any

from dendrite.processing._types import Sample
from dendrite.processing.modes._metrics import AsynchronousMetrics
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import Buffer, extract_event_code
from dendrite.utils.state_keys import mode_metric_key

# Async mode constants
LOG_INTERVAL_SAMPLES = 30000  # ~60s @ 500Hz between log messages
LOG_PREDICTION_INTERVAL = 500
LOG_METRICS_INTERVAL = 50


@dataclass
class AsyncPrediction:
    """Continuous prediction output."""

    prediction: int = 0  # Original event code from paradigm configuration
    event_name: str = ""
    confidence: float = 0.0
    detected: bool = False  # True when dwell threshold was met


@dataclass
class AsyncMetrics:
    """Continuous mode metrics for GUI."""

    prediction: int = 0  # Original event code from paradigm configuration
    confidence: float = 0.0
    event_name: str = ""
    true_label: int | None = None
    balanced_accuracy: float = 0.0
    detected: bool = False


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

    metrics_manager: AsynchronousMetrics | None  # narrowed from BaseMode
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
        """Initialize AsynchronousMode with validated instance configuration."""
        super().__init__(
            output_queue=output_queue,
            stop_event=stop_event,
            prediction_queue=prediction_queue,
            instance_config=instance_config,
            shared_state=shared_state,
            training_queue=training_queue,
        )

        self.decoder_config = instance_config.get("decoder_config", {})

        # Decoder source configuration
        self.decoder_source = instance_config.get("decoder_source", "database")

        # Prediction timing configuration (recalculated in _initialize_mode with effective rate)
        self.step_size_ms = instance_config.get("step_size_ms", 100)
        self.samples_per_prediction_step = max(
            1, int(self.sample_rate * (self.step_size_ms / 1000.0))
        )

        # Sliding window configuration
        self.window_length_sec = instance_config.get("window_length_sec", 1.0)
        self.epoch_length_samples = int(self.window_length_sec * self.sample_rate)

        self.prediction_count = 0
        self.current_sample_index = 0
        self._current_label = -1
        self._active_label = -1
        self._labeling_samples_remaining = 0
        self._cached_metrics: dict = {}

        # Online decoder reload state
        self._last_decoder_check_ts: float = 0.0
        self._source_mode: str | None = instance_config.get("source_mode")

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

        # Online mode starts without decoder — it will be trained during the session
        if self.decoder_source != "online" and not self.decoder_config:
            self.logger.error("Decoder config is required")
            return False

        if not self.channel_selection:
            self.logger.error("channel_selection is required for AsynchronousMode")
            return False

        return True

    def _initialize_mode(self) -> bool:
        """Initialize asynchronous mode components."""
        try:
            # Setup per-mode preprocessing (sets self.effective_sample_rate)
            self._setup_preprocessor()

            # Recalculate timing with effective sample rate
            self.epoch_length_samples = int(self.window_length_sec * self.effective_sample_rate)
            self.samples_per_prediction_step = max(
                1, int(self.effective_sample_rate * (self.step_size_ms / 1000.0))
            )
            self.logger.info(f"Window: {self.epoch_length_samples} samples at {self.effective_sample_rate}Hz")

            self._setup_buffer(self.epoch_length_samples)

            from dendrite.ml.decision_gate import DecisionGate

            num_classes = len(self.label_mapping) if self.label_mapping else 2
            gate = DecisionGate(
                strategy=self.instance_config.get("detection_strategy", "dwell"),
                dwell_n=self.instance_config.get("dwell_n", 3),
                confidence_threshold=float(
                    self.instance_config.get("confidence_threshold", 0.0),
                ),
            )
            self._setup_metrics_manager(
                num_classes=num_classes,
                mode_type="asynchronous",
                label_mapping=self.reverse_label_mapping,
                gate=gate,
                step_size_ms=self.step_size_ms,
            )

            # Create decoder and handle initial model loading
            if self.decoder_source == "online":
                self.logger.info(
                    "Decoder source: online — waiting for trained decoder via SharedState"
                )
            elif self.decoder_source == "external":
                self.logger.info("Decoder will be injected externally")
            else:
                # database: load from decoder path
                self._create_decoder(self.decoder_config)

                if self.decoder:
                    model_path = self.decoder_config.get("decoder_path", "")
                    if model_path:
                        if self._load_decoder(model_path, self.decoder_source):
                            self._activate_decoder(self.decoder)
                    else:
                        self.logger.warning(
                            f"No model path specified for {self.decoder_source} source"
                        )

            self.logger.info("AsynchronousMode initialized successfully (inference-only)")
            if self._source_mode:
                self.logger.info(f"Linked to sync mode: {self._source_mode}")
            self.logger.info(
                f"Configuration: epoch_length={self.epoch_length_samples} samples, "
                f"prediction_step={self.samples_per_prediction_step} samples ({self.step_size_ms}ms)"
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
                sample = self._get_next_sample()
                if sample is None:
                    continue
                data_received_count += 1
                self._process_data(sample)

                if data_received_count % LOG_INTERVAL_SAMPLES == 0:
                    self.logger.info(f"Processed {data_received_count} data samples")

                if (
                    self.buffer is not None
                    and self.buffer.is_ready_for_step(self.samples_per_prediction_step)
                ):
                    self._trigger_prediction()

                # Poll for online decoder ~1Hz
                if self.decoder_source == "online" and self.shared_state:
                    if self.current_sample_index % max(1, int(self.effective_sample_rate)) == 0:
                        self._check_for_trained_decoder()

                # Activate decoder loaded by background thread (thread-safe: main loop only)
                if self._pending_decoder_load is not None:
                    info = self._pending_decoder_load
                    self._pending_decoder_load = None
                    decoder = info.pop("_decoder", None)
                    if decoder and self._activate_decoder(decoder):
                        source = info.get("source_mode", "unknown")
                        self.logger.info(
                            f"Online decoder activated from {source} "
                            f"(trained in {info.get('elapsed', 0):.1f}s, "
                            f"{info.get('n_epochs', '?')} epochs)"
                        )

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

        self.logger.info(f"Exiting main loop. Processed {data_received_count} samples total")

    def _cleanup(self):
        """Cleanup asynchronous mode resources."""
        self.logger.info("Cleaning up AsynchronousMode")
        self.logger.info(f"Total predictions made: {self.prediction_count}")
        super()._cleanup()

    def _process_data(self, sample: Sample):
        """Process incoming data sample using unified buffer and existing event labeling."""
        if self.stop_event.is_set():
            return

        # Apply per-mode preprocessing (CAR, bandpass, downsample)
        processed = self._preprocess_sample(sample)
        if processed is None:
            return  # Accumulating for downsample
        sample = processed

        assert self.buffer is not None, "_setup_buffer must run before _process_data"

        self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

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
            if self.metrics_manager and self.is_decoder_ready:
                self.metrics_manager.register_event(self.current_sample_index, current_label)
        self._current_label = current_label

        self.buffer.add_sample(sample)
        self.current_sample_index += 1

    def _trigger_prediction(self):
        """Trigger a prediction using the current sliding window data."""
        if not self.is_decoder_ready or self.buffer is None:
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

            prediction, confidence, _ = self._predict(X_input)
            self._update_gpu_metrics()

            self._update_metrics_and_send(prediction, confidence)

            self.prediction_count += 1

            if self.prediction_count % LOG_PREDICTION_INTERVAL == 1:
                self.logger.debug(
                    f"Made prediction #{self.prediction_count}: {prediction} (confidence={confidence:.3f})"
                )

        except Exception as e:
            self.logger.error(f"Error in _trigger_prediction: {e}", exc_info=True)

    def _update_metrics_and_send(self, prediction: int, confidence: float):
        """Update metrics and send prediction output."""
        detected = False
        if self.metrics_manager:
            _in_trial, detected = self.metrics_manager.add_prediction(
                prediction=prediction,
                current_sample_idx=self.current_sample_index,
                confidence=confidence,
            )

        if self.metrics_manager and self.prediction_count % LOG_METRICS_INTERVAL == 0:
            self._cached_metrics = self.metrics_manager.get_all_metrics()

            if self._current_label >= 0:
                n_trials = self._cached_metrics.get("n_trials", 0)
                balanced_acc = self._cached_metrics.get("balanced_accuracy", 0.0)
                self.logger.info(
                    f"EVENT - Pred: {int(prediction)} Label: {int(self._current_label)} "
                    f"(conf={confidence:.3f}, acc={balanced_acc:.2f}, trials={n_trials})"
                )

        self._send_prediction_output(prediction, confidence, self._cached_metrics, detected)

    def _send_prediction_output(
        self,
        prediction: int,
        confidence: float,
        current_metrics: dict[str, Any],
        detected: bool = False,
    ):
        """Send prediction data to output queues."""
        event_code = self.index_to_event_code.get(prediction, prediction)

        prediction_payload = AsyncPrediction(
            prediction=event_code,
            event_name=self.reverse_label_mapping.get(prediction, str(int(prediction))),
            confidence=confidence,
            detected=detected,
        )
        self._send_output(prediction_payload, "prediction", queue="prediction")

        balanced_accuracy = current_metrics.get("balanced_accuracy", 0.0)
        if self.shared_state:
            self.shared_state.set(mode_metric_key(self.mode_name, "accuracy"), balanced_accuracy)

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
            detected=detected,
        )
        self._send_output(metrics_payload, "performance", queue="main")

    def _check_for_trained_decoder(self):
        """Poll SharedState for a newly trained decoder. Load in background thread."""
        if self.shared_state is None:
            return
        # Prefer mode-specific key (immune to overwrite by other modes)
        result = None
        if self._source_mode:
            result = self.shared_state.get(f"{self._source_mode}:trained_decoder")

        if not result:
            result = self.shared_state.get("latest_trained_decoder")
            if not result:
                return
            # Filter by source mode if linked
            if self._source_mode and result.get("source_mode") != self._source_mode:
                self.logger.debug(
                    f"Ignoring decoder from '{result.get('source_mode')}' "
                    f"(linked to '{self._source_mode}')"
                )
                return

        ts = result.get("timestamp", 0)
        if ts <= self._last_decoder_check_ts:
            return  # Already processed this decoder

        self._last_decoder_check_ts = ts
        path = result.get("path", "")
        if not path:
            return

        delay = time.time() - ts
        self.logger.info(
            f"New trained decoder detected: {path} "
            f"(source={result.get('source_mode')}, delay={delay:.1f}s)"
        )
        self._start_background_decoder_load(path, result)

    def _activate_decoder(self, decoder) -> bool:
        """Configure mode for a new decoder: shapes, preprocessing, metrics.

        Called from the main loop (via _pending_decoder_load) or during init.
        Returns True if decoder is ready to use, False on validation failure.
        """
        try:
            primary, _ch, expected_t = self._validate_decoder_channels(decoder)
        except ValueError as e:
            self.logger.error(f"{e} — decoder disabled")
            return False

        self.decoder = decoder

        if not primary:
            return True

        if expected_t:
            self.logger.info(
                f"Decoder: {decoder.config.model_type} "
                f"{_ch}ch x {expected_t} samples ({primary})"
            )

        # --- Update epoch length / buffer / metrics ---
        if expected_t and expected_t != self.epoch_length_samples:
            self.logger.info(
                f"Epoch length: {self.epoch_length_samples}"
                f" -> {expected_t}"
            )
            self.epoch_length_samples = expected_t
            self.window_length_sec = expected_t / self.effective_sample_rate
            self._setup_buffer(self.epoch_length_samples)
            self.samples_per_prediction_step = max(
                1,
                int(self.effective_sample_rate * (self.step_size_ms / 1000.0)),
            )

        # Update detection window on existing metrics (preserve trial history)
        if self.metrics_manager:
            self.metrics_manager.detection_window_samples = self.epoch_length_samples

        # --- Apply decoder's preprocessing config if it differs ---
        preproc_cfg = getattr(decoder.config, "preprocessing_config", None)
        if preproc_cfg and preproc_cfg.modality_preprocessing and self._sample_preprocessor:
            dp = preproc_cfg.modality_preprocessing.get(primary)
            if dp:
                new_config = dp.model_dump(exclude_none=True)
                cur = self._sample_preprocessor._config.get(primary, {})

                runtime_keys = {"num_channels", "sample_rate", "channel_labels"}
                cur_cmp = {k: v for k, v in cur.items() if k not in runtime_keys}
                new_cmp = {k: v for k, v in new_config.items() if k not in runtime_keys}

                if cur_cmp != new_cmp:
                    self.logger.warning(
                        f"Overriding preprocessing: "
                        f"{cur.get('lowcut')}-{cur.get('highcut')}Hz"
                        f" -> {new_config.get('lowcut')}-{new_config.get('highcut')}Hz"
                        f" (from decoder)"
                    )
                    self._sample_preprocessor.reset_config({primary: new_config})
                    self.effective_sample_rate = self._sample_preprocessor.effective_sample_rate

        return True

