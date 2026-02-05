import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from dendrite.constants import MODE_THREAD_JOIN_TIMEOUT
from dendrite.data.dataset import Dataset
from dendrite.ml.search import create_optuna_search_configs
from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.decoder_pool import DecoderPool
from dendrite.processing.modes.mode_utils import extract_event_code, get_shared_model_path
from dendrite.utils.state_keys import mode_metric_key

# Sync mode constants
MODEL_SAVE_INTERVAL_SEC = 30
TRAINING_QUEUE_SIZE = 5
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
    """ERP epoch data for visualization."""

    event_type: str | int = 0
    eeg_data: np.ndarray = field(default_factory=lambda: np.array([]))
    start_offset_ms: float = 0.0  # Epoch start relative to stimulus (negative = pre-stim)
    sample_rate: float = 500.0  # For time-to-sample conversion


@dataclass
class DecoderSearchPayload:
    """Multi-decoder search results."""

    performances: dict[str, float] = field(default_factory=dict)
    best_id: str = ""
    epoch: int = 0


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
        """Initialize SynchronousMode with validated instance configuration."""
        # Call parent constructor - extracts name, channel_selection automatically
        super().__init__(
            data_queue=data_queue,
            output_queue=output_queue,
            stop_event=stop_event,
            sample_rate=sample_rate,
            prediction_queue=prediction_queue,
            instance_config=instance_config,
            shared_state=shared_state,
        )

        # Extract synchronous-specific parameters (event_mapping, label_mapping from BaseMode)
        self.decoder_config = instance_config.get("decoder_config", {})

        self.file_identifier = instance_config.get("file_identifier")

        # Model sharing configuration
        self.study_name = instance_config.get("study_name", "default_study")
        self.last_model_save_time = 0
        self.model_save_interval = MODEL_SAVE_INTERVAL_SEC

        # Dataset storage (enabled by default)
        self.dataset = Dataset(name=f"SyncMode_{self.mode_name}")
        self.training_interval = instance_config.get("training_interval", DEFAULT_TRAINING_INTERVAL)

        # Decoder search configuration (pool created lazily on first training)
        self.enable_decoder_search = instance_config.get("enable_decoder_search", False)
        self.decoder_pool = None

        # Calculate epoch timing (recalculated in _initialize_mode with effective rate)
        self.start_offset = instance_config.get("start_offset", 0.0)
        self.end_offset = instance_config.get("end_offset", 2.0)
        self.start_offset_samples = int(self.start_offset * self.sample_rate)
        self.end_offset_samples = int(self.end_offset * self.sample_rate)
        self.epoch_length_samples = self.end_offset_samples - self.start_offset_samples

        # Epoch tracking
        self.epoch_count = 0

        # Simple epoch tracking
        self.pending_epochs = []
        self.current_sample_index = 0

        # Decoder/model management
        self.decoder = None

        # Simplified concurrent training
        self._training_queue = None
        self._training_thread = None
        self._stop_training = None
        self.effective_sample_rate: float | None = None

    def _validate_configuration(self) -> bool:
        """Validate synchronous mode configuration."""
        if self.epoch_length_samples <= 0:
            self.logger.error(
                f"Epoch length must be positive. "
                f"Calculated: {self.epoch_length_samples} samples from "
                f"start_offset={self.start_offset}, end_offset={self.end_offset}, "
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
        pre_event = abs(self.start_offset_samples) if self.start_offset_samples < 0 else 0
        min_buffer = pre_event + self.epoch_length_samples
        recommended = int(min_buffer * safety_factor)

        self.logger.info(
            f"Buffer size: {recommended} samples (epoch={self.epoch_length_samples}, safety={safety_factor}x)"
        )
        return recommended

    def _initialize_mode(self) -> bool:
        """Initialize synchronous mode components."""
        try:
            # Calculate epoch timing using effective sample rate (accounts for preprocessing)
            primary_modality = self.modalities[0] if self.modalities else "eeg"
            self.effective_sample_rate = self._get_modality_sample_rate(primary_modality)
            self.start_offset_samples = int(self.start_offset * self.effective_sample_rate)
            self.end_offset_samples = int(self.end_offset * self.effective_sample_rate)
            self.epoch_length_samples = self.end_offset_samples - self.start_offset_samples
            self.logger.info(f"Epoch: {self.epoch_length_samples} samples at {self.effective_sample_rate}Hz")

            # Setup unified buffer with appropriate size for epochs + pre-event data
            buffer_size = self._calculate_sync_buffer_size(safety_factor=2.0)
            self._setup_buffer(buffer_size)

            num_classes = len(self.label_mapping) if self.label_mapping else 2
            self._setup_metrics_manager(num_classes=num_classes, mode_type="synchronous")

            # Create decoder (decoder pool created lazily on first training if enabled)
            self.decoder = self._create_decoder(self.decoder_config)
            if not self.decoder:
                self.logger.info("Decoder creation deferred until modalities are detected")

            self._setup_concurrent_training()

            self.logger.info("SynchronousMode initialized successfully")
            self.logger.info(f"Training interval: {self.training_interval} epochs")
            self.logger.info(f"Label mapping: {self.label_mapping}")

            # Log optional features
            model_config = self.decoder_config.get("model_config", {})
            if model_config.get("use_augmentation", False):
                self.logger.info(
                    f"Data augmentation: {model_config.get('aug_strategy', 'moderate')} strategy"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error initializing mode: {e}", exc_info=True)
            return False

    def _setup_concurrent_training(self):
        """Setup simplified concurrent training."""
        self._model_lock = threading.Lock()
        self._training_queue = Queue(maxsize=TRAINING_QUEUE_SIZE)
        self._stop_training = threading.Event()

        self._training_thread = threading.Thread(
            target=self._training_worker, daemon=True, name=f"Training-{self.mode_name}"
        )
        self._training_thread.start()
        self.logger.info("Concurrent training thread started")

    def _create_decoder(self, decoder_config: dict[str, Any]) -> Any | None:
        """Create decoder for SynchronousMode with modality check."""
        # Check if modalities detected (sync-specific requirement)
        if not self.modalities_detected or not self.modalities:
            self.logger.info("Modalities not detected - deferring decoder creation")
            return None

        # Use base implementation for core creation
        decoder = super()._create_decoder(decoder_config)

        if decoder is None:
            return None

        # Set GUI mappings
        if hasattr(decoder, "event_mapping"):
            decoder.event_mapping = self.event_mapping
        if hasattr(decoder, "label_mapping"):
            decoder.label_mapping = self.label_mapping

        return decoder

    def _run_main_loop(self):
        """Run the main synchronous processing loop."""
        self.logger.info("Entering main event-driven loop")

        while not self.stop_event.is_set():
            try:
                sample = self.data_queue.get(timeout=0.1)
                self._process_data(sample)
            except Empty:
                pass  # Queue timeout, continue loop
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

    def _cleanup(self):
        """Cleanup synchronous mode resources."""
        if self._stop_training:
            self._stop_training.set()

        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=MODE_THREAD_JOIN_TIMEOUT)

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

    def _training_worker(self):
        """Simple background training worker."""
        while not self._stop_training.is_set():
            try:
                # Get training data with timeout
                X_data, y_data = self._training_queue.get(timeout=0.5)

                if not self.decoder:
                    continue

                # Set input shapes from training data if not already set
                if not self.decoder.input_shapes:
                    modality = self.modalities[0] if self.modalities else "eeg"
                    self.decoder.input_shapes = {modality: list(X_data.shape[1:])}
                    self.decoder.config.input_shapes = self.decoder.input_shapes

                # Train with lock to prevent prediction conflicts
                with self._model_lock:
                    start_time = time.time()
                    self.decoder.fit(X_data, y_data)
                    elapsed = time.time() - start_time

                self.logger.info(f"Training completed: {len(y_data)} samples in {elapsed:.3f}s")
                try:
                    self._send_output({"training_s_point": float(elapsed)}, "training_point")
                except Exception as e:
                    self.logger.warning(f"Failed to emit training_point: {e}")
                self._save_model_for_sharing()

                # Update GPU metrics after training (has internal throttling)
                self._update_gpu_metrics()

            except Empty:
                pass  # Timeout, check stop event and continue
            except Exception as e:
                self.logger.error(f"Training error: {e}")

    def _process_data(self, sample):
        """Process a single data sample using unified buffer."""
        if not isinstance(sample, dict):
            return

        # Track LSL timestamp for payloads
        self.last_lsl_timestamp = sample.get("lsl_timestamp", 0.0)

        # Add sample to unified buffer
        self.buffer.add_sample(sample)
        self.current_sample_index += 1

        # Update internal latency metric per-sample (keeps GUI fresh)
        self._compute_and_store_internal_latency()

        # Extract event marker and check if it's a trigger
        event_code = extract_event_code(sample)
        if event_code != -1 and event_code in self.event_mapping:
            self._handle_event_trigger(event_code)

        # Check for ready pending epochs
        self._check_pending_epochs()

    def _handle_event_trigger(self, event_code: int):
        """Handle an event trigger by scheduling epoch extraction."""
        if event_code not in self.event_mapping:
            return

        event_type = self.event_mapping[event_code]

        if event_type not in self.label_mapping:
            return

        class_index = self.label_mapping[event_type]
        self.logger.info(f"Event: {event_type} (class_index={class_index})")

        event_sample_index = self.current_sample_index - 1  # Event was at previous sample

        # Schedule epoch extraction after post-event data is collected
        if self.end_offset_samples > 0:
            self.pending_epochs.append(
                (
                    event_sample_index,  # event position
                    event_code,
                    event_type,
                    class_index,
                    self.end_offset_samples,  # samples to wait
                )
            )
        else:
            # Extract immediately if no post-event data needed
            self._extract_epoch(event_code, event_type, class_index, 0)

    def _check_pending_epochs(self):
        """Check if any pending epochs are ready for extraction."""
        ready_epochs = []
        remaining_epochs = []

        for event_pos, event_code, event_type, class_index, samples_needed in self.pending_epochs:
            samples_elapsed = self.current_sample_index - event_pos
            if samples_elapsed >= samples_needed:
                ready_epochs.append((event_code, event_type, class_index, samples_elapsed))
            else:
                remaining_epochs.append(
                    (event_pos, event_code, event_type, class_index, samples_needed)
                )

        # Process ready epochs and update pending list
        for event_code, event_type, class_index, delay in ready_epochs:
            self._extract_epoch(event_code, event_type, class_index, delay)

        self.pending_epochs = remaining_epochs

    def _extract_epoch(
        self, event_code: int, event_type: str, class_index: int, delay_samples: int
    ):
        """Extract and process epoch from buffer."""
        X_input = self.buffer.extract_epoch_at_event(
            start_offset_samples=self.start_offset_samples,
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
            # Send ERP data for visualization
            if "eeg" in X_input:
                erp_payload = ERPPayload(
                    event_type=event_type,
                    eeg_data=X_input["eeg"],
                    start_offset_ms=self.start_offset * 1000,
                    sample_rate=float(self.effective_sample_rate),
                )
                self._send_output(erp_payload, "erp", queue="main")

            # Store epoch in dataset
            self._store_epoch_in_dataset(X_input, class_index)

            # Check if training is needed
            if self.epoch_count % self.training_interval == 0:
                self._trigger_training()

            # Make prediction if model is ready
            if self.decoder and getattr(self.decoder, "is_fitted", False):
                prediction, confidence, _ = self._predict(
                    model=self.decoder,
                    X_input=X_input,
                    lock=self._model_lock,
                    blocking=False,  # Don't wait if model is being trained
                )
                self._update_gpu_metrics()
                self._update_metrics_and_send(prediction, confidence, class_index, event_type)
            elif not self.decoder:
                self.logger.warning("No decoder available for prediction")

        except Exception as e:
            self.logger.error(f"Error processing epoch: {e}", exc_info=True)

    def _store_epoch_in_dataset(self, X_input: dict, class_index: int):
        """Store epoch data in the dataset."""
        # Create informative source identifier
        source_id = (
            f"{self.mode_name}_{self.file_identifier}"
            if self.file_identifier
            else f"{self.mode_name}_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        # Add relevant metadata to source info
        source_info = {
            "sampling_freq": self.sample_rate,
            "event_mapping": self.event_mapping,
            "label_mapping": self.label_mapping,
            "reverse_label_mapping": self.reverse_label_mapping,
            "channel_selection": self.channel_selection,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "epoch_length_samples": self.epoch_length_samples,
            "mode_name": self.mode_name,
        }
        self.dataset.add_sample(
            X_input, class_index, source_id, source_info, input_shape="(n_channels, n_times)"
        )

    def _trigger_training(self):
        """Trigger model training when interval is reached."""
        self.logger.info(
            f"Training triggered: epoch {self.epoch_count}, interval {self.training_interval}"
        )

        # Get training data (bad epochs auto-excluded via is_bad flag)
        modality = self.modalities[0] if self.modalities else "eeg"
        training_data = self.dataset.get_training_data(modality=modality)

        # Schedule training if we have sufficient data
        if not training_data or len(training_data.get("y", [])) < 2:
            self.logger.warning(
                f"Insufficient training data: dataset_samples={len(self.dataset.samples)}, "
                f"training_data_y={len(training_data.get('y', [])) if training_data else 0}, "
                f"modality={modality}"
            )
            return

        self.logger.info(f"Training with {len(training_data['y'])} samples")

        if self.enable_decoder_search:
            # Lazy create pool and decoders on first training
            if self.decoder_pool is None:
                # Generate search configs using Optuna for intelligent hyperparameter sampling
                search_configs = create_optuna_search_configs(
                    n_trials=5,
                    model_types=["LinearEEG", "EEGNet"],
                    sampler_type="random",  # Fast random sampling for real-time use
                    seed=42,  # Reproducibility
                )
                self.decoder_pool = DecoderPool(
                    base_config=self.decoder_config,
                    search_configs=search_configs,
                    logger=self.logger,
                )

                # Initialize decoders with training data shapes
                num_classes = len(self.label_mapping) if self.label_mapping else 2
                input_shapes = {modality: training_data["input_shape"]}
                self.decoder_pool.initialize_decoders(
                    num_classes=num_classes,
                    input_shapes=input_shapes,
                    event_mapping=self.event_mapping,
                    label_mapping=self.label_mapping,
                )
                self.logger.info(f"Decoder pool created with {len(search_configs)} decoders")

            # Train in background thread (non-blocking)
            threading.Thread(
                target=self._train_pool_background,
                args=(training_data["X"], training_data["y"]),
                daemon=True,
                name=f"PoolTraining-{self.mode_name}",
            ).start()
        else:
            # Schedule training on background thread (non-blocking)
            if self._training_queue:
                try:
                    self._training_queue.put((training_data["X"], training_data["y"]), block=False)
                    self.logger.info(f"Training scheduled: {len(training_data['X'])} samples")
                except Full:
                    self.logger.warning("Training queue full, skipping training")

    def _train_pool_background(self, X_data, y_data):
        """Background training for decoder pool."""
        try:
            # Train all decoders
            performances = self.decoder_pool.train_all(X_data, y_data, self._model_lock)

            # Get and set best decoder (thread-safe)
            best_id, best_decoder = self.decoder_pool.get_best_decoder()
            if best_decoder:
                with self._model_lock:
                    self.decoder = best_decoder
                self.logger.info(f"Best decoder: {best_id}")

            # Send metrics for visualization
            if performances:
                search_payload = DecoderSearchPayload(
                    performances=performances, best_id=best_id or "", epoch=self.epoch_count
                )
                self._send_output(search_payload, "decoder_search", queue="main")

            # Save best model for sharing
            self._save_model_for_sharing()

        except Exception as e:
            self.logger.error(f"Pool training failed: {e}", exc_info=True)

    def _update_metrics_and_send(self, prediction, confidence, class_index, event_type):
        """Update metrics and send performance updates."""
        current_metrics = {}
        if self.metrics_manager:
            self.metrics_manager.update_metrics(prediction, class_index)
            current_metrics = self.metrics_manager.get_current_metrics()

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

    def _save_model_for_sharing(self):
        """Save model to shared folder for other modes to consume."""
        if not self.decoder or not getattr(self.decoder, "is_fitted", False):
            return

        current_time = time.time()
        if current_time - self.last_model_save_time < self.model_save_interval:
            return

        try:
            identifier = get_shared_model_path(self.mode_name, self.file_identifier)

            with self._model_lock:
                self.decoder.save(identifier, study_name=self.study_name)

            self.last_model_save_time = current_time
            self.logger.info(f"Decoder shared to: {identifier}")

        except Exception as e:
            self.logger.error(f"Model sharing failed: {e}")
