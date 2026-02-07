"""
Base Mode Class for Dendrite System

Abstract base class for all Dendrite processing modes. Modes handle workflow orchestration
and data processing patterns while decoders focus on algorithm implementation.

SAMPLE STRUCTURE FROM DATAPROCESSOR:
Each sample is a dictionary with:
- Modality data: 'eeg', 'emg', 'eog', etc. (numpy arrays, shape: (n_channels, 1))
- 'markers': np.ndarray, shape: (1, 1) (-1 = background, >0 = event codes)
- 'lsl_timestamp': float (LSL-provided timestamp)
- '_daq_receive_ns': int (time.time_ns() when DAQ receives from LSL)
- '_stream_name': str (name of the stream this sample came from)
- '_eeg_latency_ms': float (EEG stream external latency in ms, computed by DAQ)

CHANNEL SELECTION:
Optional channel filtering: {'eeg': [0,1,2,3], 'emg': [0]} selects specific channels.
If None, all available channels are used.
"""

import logging
import multiprocessing
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from dendrite.processing.queue_utils import FanOutQueue
from dendrite.utils.shared_state import SharedState


@dataclass
class ModeOutputPacket:
    """Packet structure for mode output to queues."""

    type: str
    mode_name: str
    mode_type: str
    data: dict[str, Any]
    data_timestamp: float | None = None


from dendrite.constants import (
    MODE_GPU_EMIT_INTERVAL,
)
from dendrite.ml.metrics.metrics_manager import MetricsManager
from dendrite.processing.modes.mode_utils import (
    Buffer,
    extract_event_mapping,
)
from dendrite.utils.logger_central import setup_logger
from dendrite.utils.modality import normalize_modality, normalize_modality_dict
from dendrite.utils.state_keys import mode_metric_key


class BaseMode(multiprocessing.Process, ABC):
    """
    Abstract base class for all Dendrite processing modes.

    Handles common infrastructure: process lifecycle, output queues,
    decoder management, metrics collection, and unified buffering.
    """

    def __init__(
        self,
        data_queue: "multiprocessing.Queue",
        output_queue: "FanOutQueue",
        stop_event: "multiprocessing.Event",
        instance_config: dict[str, Any],
        sample_rate: float,
        prediction_queue: "multiprocessing.Queue | None" = None,
        shared_state: "SharedState | None" = None,
    ):
        """Initialize the base mode with core infrastructure."""
        super().__init__()

        self.data_queue = data_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.prediction_queue = prediction_queue
        self.shared_state = shared_state

        self.instance_config = instance_config
        self.mode_name = instance_config["name"]
        raw_channel_selection = instance_config.get("channel_selection") or {}
        self.channel_selection = normalize_modality_dict(raw_channel_selection)
        raw_modality_labels = instance_config.get("modality_labels") or {}
        self.modality_labels = normalize_modality_dict(raw_modality_labels)
        self.preprocessing_config = instance_config.get("preprocessing_config")
        self.sample_rate = float(sample_rate)

        raw_required = instance_config.get("required_modalities", ["eeg"])
        self.required_modalities = [normalize_modality(m) for m in raw_required]

        self.logger = None

        if self.channel_selection:
            self.modalities = list(self.channel_selection.keys())
            self.modality_num_channels = {
                mod: len(indices) for mod, indices in self.channel_selection.items()
            }
            self.modalities_detected = True
        else:
            self.modalities = []
            self.modality_num_channels = {}
            self.modalities_detected = False

        self.metrics_manager = None
        self._start_time = None
        self._is_running = False
        self.buffer = None
        self._model_lock = None
        self.last_lsl_timestamp = 0.0

        self._gpu_emit_interval_sec = MODE_GPU_EMIT_INTERVAL
        self._gpu_last_emit_time = 0.0

        self._mode_type = getattr(self, "MODE_TYPE", "unknown")

        self.event_mapping = extract_event_mapping(instance_config)
        self.label_mapping: dict[str, int] = {}
        self.reverse_label_mapping: dict[int, str] = {}
        self.index_to_event_code: dict[int, int] = {}
        self._generate_label_mapping(self.event_mapping)

    @abstractmethod
    def _validate_configuration(self) -> bool:
        """Validate mode-specific configuration. Returns True if valid."""

    @abstractmethod
    def _initialize_mode(self) -> bool:
        """Initialize mode-specific components. Returns True if successful."""

    @abstractmethod
    def _run_main_loop(self):
        """Run the main processing loop with mode-specific logic."""

    def run(self):
        """Main process entry point with common setup and mode-specific delegation."""
        self._setup_logger()
        self._start_time = time.time()
        self._is_running = True

        self.logger.info(f"{self.__class__.__name__} process starting")

        if self.channel_selection:
            self.logger.info(f"Using configured modalities: {list(self.channel_selection.keys())}")
        else:
            self.logger.info("Auto-detecting modalities from data")

        try:
            if not self._validate_configuration():
                self.logger.error("Configuration validation failed")
                return

            if not self._initialize_mode():
                self.logger.error("Mode initialization failed")
                return

            self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Fatal error in run(): {e}", exc_info=True)
        finally:
            self._cleanup()
            self._is_running = False

            if self._start_time:
                runtime = time.time() - self._start_time
                self.logger.info(f"Process ran for {runtime:.2f} seconds")

            self.logger.info(f"{self.__class__.__name__} process stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status information for this mode."""
        status = {
            "mode_name": self.mode_name,
            "mode_type": self.__class__.__name__,
            "is_running": self._is_running,
            "modalities": self.modalities,
            "sample_rate": self.sample_rate,
        }

        if self._start_time:
            status["uptime_seconds"] = time.time() - self._start_time

        if self.metrics_manager:
            status["metrics"] = self.metrics_manager.get_current_metrics()

        if self.buffer:
            status["buffer"] = self.buffer.get_status()

        return status

    def get_mode_type(self) -> str:
        """Get the mode type string for this mode instance."""
        return self._mode_type

    def _cleanup(self):
        """Cleanup resources before process termination."""
        pass

    def _setup_logger(self):
        """Initialize the logger for this mode."""
        process_name = f"{self.__class__.__name__}-{self.mode_name}"
        multiprocessing.current_process().name = process_name

        self.logger = setup_logger(process_name, level=logging.INFO)
        self.logger.info(f"{self.__class__.__name__} initializing...")

    def _setup_buffer(self, buffer_size: int):
        """Setup unified buffer for all modes."""
        self.buffer = Buffer(self.modalities, buffer_size, self.logger)

    def _setup_metrics_manager(
        self,
        num_classes: int = 2,
        mode_type: str | None = None,
        label_mapping: dict[int, str] | None = None,
        background_class: int | None = None,
    ):
        """Initialize the metrics manager."""
        if mode_type is None:
            mode_type = self._mode_type

        detection_window_samples = None
        if mode_type == "asynchronous" and hasattr(self, "epoch_length_samples"):
            detection_window_samples = self.epoch_length_samples

        self.metrics_manager = MetricsManager(
            mode_type=mode_type,
            sample_rate=int(self.sample_rate),
            num_classes=num_classes,
            detection_window_samples=detection_window_samples,
            label_mapping=label_mapping,
            background_class=background_class,
        )
        self.logger.info(f"MetricsManager initialized for {mode_type} mode")

    def _send_output(
        self,
        payload: Any,
        output_type: str,
        queue: str = "both",
        data_timestamp: float | None = None,
    ) -> None:
        """Send output to specified queue(s).

        Args:
            payload: Data to send (dataclass or dict)
            output_type: Type identifier (e.g., 'performance', 'erp', 'prediction')
            queue: Target queue ('main', 'prediction', or 'both')
            data_timestamp: LSL timestamp for the data (auto-filled from last_lsl_timestamp if None)
        """
        processed = asdict(payload) if is_dataclass(payload) else payload

        if data_timestamp is None:
            data_timestamp = self.last_lsl_timestamp or None

        packet = ModeOutputPacket(
            type=output_type,
            mode_name=self.mode_name,
            mode_type=self.get_mode_type(),
            data=processed,
            data_timestamp=data_timestamp,
        )

        if queue in ("main", "both") and self.output_queue:
            self.output_queue.put(asdict(packet))
        if queue in ("prediction", "both") and self.prediction_queue:
            self.prediction_queue.put(asdict(packet))

    def _compute_and_store_internal_latency(self):
        """Compute internal pipeline latency (window ready -> now). Call BEFORE inference."""
        if not self.buffer or not self.shared_state:
            return
        newest_ts = self.buffer.get_newest_timestamp()
        if newest_ts:
            now_ns = time.time_ns()
            internal_ms = (now_ns - newest_ts) / 1_000_000.0
            self.shared_state.set(mode_metric_key(self.mode_name, "internal_ms"), internal_ms)

    def _update_gpu_metrics(self):
        """Track GPU memory if using CUDA."""
        if not self.decoder or not self.shared_state:
            return
        if (time.time() - self._gpu_last_emit_time) < self._gpu_emit_interval_sec:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            self.shared_state.set(mode_metric_key(self.mode_name, "gpu_mb"), float(allocated_mb))
            self._gpu_last_emit_time = time.time()
        except Exception:
            pass  # GPU metrics are optional

    def _generate_label_mapping(self, event_mapping: dict):
        """Generate label mapping from event mapping.

        Creates sequential class indices (0, 1, 2...) for ML training from
        event codes that may be non-sequential (e.g., event_id 7, 8).

        Args:
            event_mapping: Dictionary mapping event codes to event names
                           e.g., {7: 'left_hand', 8: 'right_hand'}
        """
        if not event_mapping:
            self.label_mapping = {}
            self.reverse_label_mapping = {}
            self.index_to_event_code = {}
            return

        sorted_events = sorted(event_mapping.items())
        self.label_mapping = {name: i for i, (code, name) in enumerate(sorted_events)}
        self.reverse_label_mapping = {i: name for name, i in self.label_mapping.items()}
        self.index_to_event_code = {i: code for i, (code, name) in enumerate(sorted_events)}

    def _predict(self, model, X_input, lock=None, blocking=True):
        """
        Unified prediction method with integrated locking and timing.

        Args:
            model: The model/decoder object with predict_sample method
            X_input: Input data - dict with modality keys or array directly
            lock: Optional threading lock for thread-safe prediction
            blocking: Whether lock acquisition should block (default True)

        Returns:
            Tuple of (prediction, confidence, inference_ms)
            Returns (0, 0.5, 0.0) as default if model not ready or error occurs
        """
        if (
            not model
            or not hasattr(model, "predict_sample")
            or not getattr(model, "is_fitted", False)
        ):
            return 0, 0.5, 0.0

        if isinstance(X_input, dict):
            primary_key = next(iter(X_input))
            X_array = X_input[primary_key]
        else:
            X_array = X_input

        acquired = False
        if lock:
            acquired = lock.acquire(blocking=blocking)
            if not acquired:
                return 0, 0.5, 0.0

        try:
            start_ns = time.perf_counter_ns()
            prediction, confidence = model.predict_sample(X_array)
            end_ns = time.perf_counter_ns()
            inference_ms = (end_ns - start_ns) / 1_000_000.0

            if self.shared_state and inference_ms > 0:
                self.shared_state.set(mode_metric_key(self.mode_name, "inference_ms"), inference_ms)

            return prediction, confidence, inference_ms

        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction error: {e}")
            return 0, 0.5, 0.0

        finally:
            if lock and acquired:
                lock.release()

    def _create_decoder(self, decoder_config: dict[str, Any]) -> Any | None:
        """
        Create and initialize a decoder with unified logic for all modes.

        This base implementation handles common decoder creation patterns:
        - Extracts model config and parameters
        - Calculates num_classes from label mapping
        - Calls create_decoder factory
        - Provides consistent logging and error handling

        Subclasses can override this method for mode-specific behavior.

        Args:
            decoder_config: Full decoder configuration dictionary

        Returns:
            Decoder instance or None if creation fails
        """
        from dendrite.ml.decoders import create_decoder

        try:
            model_config = decoder_config.get("model_config", {})
            model_type = model_config.get("model_type", "EEGNet")

            decoder_params = model_config.copy()

            num_classes = (
                len(self.label_mapping)
                if self.label_mapping
                else model_config.get("num_classes", 2)
            )
            decoder_params["num_classes"] = num_classes

            if self.logger:
                self.logger.info(f"Creating {model_type} decoder with {num_classes} classes")

            decoder = create_decoder(**decoder_params)

            if self.modality_labels:
                decoder.channel_labels = self.modality_labels
            decoder.sample_rate = self.sample_rate
            if self.preprocessing_config:
                decoder.config.preprocessing_config = self.preprocessing_config

            if self.logger:
                self.logger.info("Decoder created successfully")

            return decoder

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating decoder: {e}", exc_info=True)
            return None

    def _load_model_from_path(self, model_path: str, model_type: str = "decoder") -> bool:
        """
        Load a decoder from the specified path and assign to self.decoder.

        Args:
            model_path: Path to the model file (without .json extension)
            model_type: Description of model type for logging (e.g., "pretrained", "updated")

        Returns:
            True if loading succeeded, False otherwise
        """
        import os

        from dendrite.ml.decoders import load_decoder

        if not model_path.endswith(".json") and os.path.exists(model_path + ".json"):
            model_path += ".json"

        if self.logger:
            self.logger.info(f"Loading {model_type} from {model_path}")

        try:
            if self._model_lock:
                with self._model_lock:
                    self.decoder = load_decoder(model_path)
            else:
                self.decoder = load_decoder(model_path)

            if self.logger:
                self.logger.info(f"Successfully loaded {model_type}")
            self._validate_loaded_decoder_channels(model_path)
            self._validate_loaded_decoder_sample_rate(self.decoder)
            return True

        except (FileNotFoundError, RuntimeError) as e:
            if self.logger:
                self.logger.error(f"Error loading {model_type} model: {e}")
            return False

    def _validate_loaded_decoder_channels(self, decoder_path: str) -> None:
        """Validate loaded decoder's channels match mode's selected channels."""
        if not self.modality_labels:
            return

        selected_labels = {}
        for modality, all_labels in self.modality_labels.items():
            indices = self.channel_selection.get(modality, [])
            if indices and all_labels:
                selected_labels[modality] = [all_labels[i] for i in indices if i < len(all_labels)]
            else:
                selected_labels[modality] = all_labels

        if not selected_labels:
            return

        try:
            from dendrite.ml.decoders import validate_decoder_file

            _, issues = validate_decoder_file(decoder_path, expected_labels=selected_labels)
            if issues and self.logger:
                for issue in issues:
                    self.logger.warning(f"Channel validation: {issue}")
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not validate decoder channels: {e}")

    def _validate_loaded_decoder_sample_rate(self, model) -> None:
        """Warn if decoder's expected sample rate doesn't match mode's rate."""
        if not self.logger:
            return

        try:
            mode_rate = getattr(self, "sample_rate", None)
            if not mode_rate:
                return

            if hasattr(model, "get_expected_sample_rate"):
                expected_rate = model.get_expected_sample_rate()
            elif hasattr(model, "config") and hasattr(model.config, "effective_sample_rate"):
                expected_rate = model.config.effective_sample_rate
            else:
                return

            if abs(expected_rate - mode_rate) > 0.1:
                self.logger.warning(
                    f"Sample rate mismatch: decoder trained at {expected_rate:.0f} Hz, "
                    f"mode outputs {mode_rate:.0f} Hz. Performance may degrade."
                )
        except Exception as e:
            self.logger.debug(f"Could not validate sample rate: {e}")

    def _get_modality_sample_rate(self, modality: str) -> int:
        """Get effective sample rate for a modality after preprocessing.

        Checks preprocessing config for target_sample_rate or downsample_factor.
        Falls back to native stream sample_rate if no preprocessing is configured.

        Args:
            modality: The modality name (e.g., 'eeg', 'emg')

        Returns:
            Effective sample rate in Hz after any resampling/downsampling
        """
        if not self.preprocessing_config:
            return int(self.sample_rate)

        mod_config = self.preprocessing_config.get("modality_preprocessing", {}).get(modality, {})

        # Check for explicit target sample rate first
        target = mod_config.get("target_sample_rate")
        if target:
            return int(target)

        # Fall back to downsample_factor calculation
        native = mod_config.get("sample_rate") or self.sample_rate
        downsample = mod_config.get("downsample_factor", 1)
        return int(native) // downsample

