"""
Base Mode Class — reads from SharedRingBuffer, runs processing pipeline.

SAMPLE STRUCTURE (from ring buffer):
Each sample is a dict with:
- Modality data: 'eeg', 'emg', etc. (numpy arrays, shape: (n_channels, 1))
- 'markers': np.ndarray, shape: (1, 1) (0 = no marker, >0 = event codes)
- 'lsl_timestamp': float
- '_stream_name': str
"""

import logging
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from dendrite.ml.decoders import create_decoder, load_decoder
from dendrite.processing._types import Sample
from dendrite.processing.modes._metrics import AsynchronousMetrics, SynchronousMetrics
from dendrite.processing.modes.mode_utils import (
    Buffer,
    FanOutQueue,
    SamplePreprocessor,
    SampleReader,
    extract_event_mapping,
    generate_label_mapping,
)
from dendrite.utils.component_state import ComponentState, ComponentStateMachine
from dendrite.utils.logger_central import setup_logger
from dendrite.utils.modality import normalize_modality_dict
from dendrite.utils.shared_state import SharedState
from dendrite.utils.state_keys import mode_metric_key

MODE_GPU_EMIT_INTERVAL = 2.0


@dataclass
class ModeOutputPacket:
    """Packet structure for mode output to queues."""

    type: str
    mode_name: str
    mode_type: str
    data: dict[str, Any]
    data_timestamp: float | None = None


class BaseMode(multiprocessing.Process, ABC):
    """
    Abstract base class for all Dendrite processing modes.

    Handles common infrastructure: process lifecycle, output queues,
    decoder management, metrics collection, and unified buffering.
    """

    def __init__(
        self,
        output_queue: "FanOutQueue",
        stop_event: "multiprocessing.Event",
        instance_config: dict[str, Any],
        prediction_queue: "multiprocessing.Queue | None" = None,
        shared_state: "SharedState | None" = None,
        training_queue: "multiprocessing.Queue | None" = None,
    ):
        """Initialize the base mode with core infrastructure."""
        super().__init__()

        self.output_queue = output_queue
        self.stop_event = stop_event
        self.prediction_queue = prediction_queue
        self.shared_state = shared_state
        self.training_queue = training_queue

        # Ring buffer config — kept for modality_labels / sample_rate extraction
        self._rb_config = instance_config.get("ring_buffer")
        self._reader: SampleReader | None = None  # created in run()

        self.instance_config = instance_config
        self.mode_name = instance_config["name"]
        raw_channel_selection = instance_config.get("channel_selection") or {}
        self.channel_selection = normalize_modality_dict(raw_channel_selection)
        rb_labels = (self._rb_config or {}).get("modality_labels", {})
        self.modality_labels = normalize_modality_dict(rb_labels)
        rb_rate = self._rb_config.get("sample_rate") if self._rb_config else None
        self.sample_rate = float(rb_rate) if rb_rate else 500.0

        self.logger = None

        self.modalities = list(self.channel_selection.keys()) if self.channel_selection else []

        self.metrics_manager = None
        self._start_time = None
        self._is_running = False
        self.buffer = None
        self.last_lsl_timestamp = 0.0

        self.decoder = None
        self._gpu_last_emit_time = 0.0
        self._state_machine: ComponentStateMachine | None = None

        self._sample_preprocessor: SamplePreprocessor | None = None
        self.effective_sample_rate: float = self.sample_rate

        # Background decoder load result (set by _start_background_decoder_load)
        self._pending_decoder_load: dict | None = None

        self._mode_type = self.MODE_TYPE

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
        if self._rb_config:
            self._reader = SampleReader(self._rb_config, self.logger)
            self._reader.connect()
        self._state_machine = ComponentStateMachine(
            f"mode:{self.mode_name}", self.shared_state
        )
        self._start_time = time.time()
        self._is_running = True

        self.logger.info(f"{self.__class__.__name__} process starting")
        self._state_machine.transition(ComponentState.STARTING)

        if self.channel_selection:
            self.logger.info(f"Using configured modalities: {list(self.channel_selection.keys())}")
        else:
            self.logger.info("Auto-detecting modalities from data")

        try:
            if not self._validate_configuration():
                self.logger.error("Configuration validation failed")
                self._state_machine.set_error("Configuration validation failed")
                return

            if not self._initialize_mode():
                self.logger.error("Mode initialization failed")
                self._state_machine.set_error("Mode initialization failed")
                return

            self._state_machine.transition(ComponentState.RUNNING)
            self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Fatal error in run(): {e}", exc_info=True)
            if self._state_machine.state not in (ComponentState.STOPPING, ComponentState.STOPPED):
                self._state_machine.set_error(str(e))
        finally:
            self._cleanup()
            self._is_running = False

            self._state_machine.finalize()

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
            status["metrics"] = self.metrics_manager.get_all_metrics()

        if self.buffer:
            status["buffer"] = self.buffer.get_status()

        return status

    def get_mode_type(self) -> str:
        """Get the mode type string for this mode instance."""
        return self._mode_type

    def _cleanup(self):
        """Cleanup resources before process termination."""
        if self._reader:
            self._reader.close()

    def _get_next_sample(self) -> Sample | None:
        """Get next sample from ring buffer. Returns Sample dict or None."""
        if self._reader is not None:
            return self._reader.read_sample()
        return None

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
        gate: Any | None = None,
        step_size_ms: float = 100.0,
    ):
        """Initialize the metrics tracker (sync or async)."""
        if mode_type is None:
            mode_type = self._mode_type

        if mode_type == "asynchronous":
            kwargs: dict[str, Any] = {
                "detection_window_samples": self.epoch_length_samples,
                "sample_rate": int(self.effective_sample_rate),
                "step_size_ms": step_size_ms,
                "label_mapping": label_mapping,
            }
            if gate is not None:
                kwargs["gate"] = gate
            self.metrics_manager = AsynchronousMetrics(**kwargs)
        else:
            self.metrics_manager = SynchronousMetrics(num_classes=num_classes)
        self.logger.info(f"Metrics initialized for {mode_type} mode")

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

    def _get_primary_modality(self) -> str:
        """Return the first configured modality."""
        if not self.modalities:
            raise RuntimeError("No modalities configured — check channel_selection")
        return self.modalities[0]

    @property
    def is_decoder_ready(self) -> bool:
        """True if decoder exists and is fitted."""
        return self.decoder is not None and getattr(self.decoder, "is_fitted", False)

    def _create_decoder(
        self, decoder_config: dict[str, Any], *, override_num_classes: int | None = None
    ) -> Any | None:
        """Create decoder from config. Stores as self.decoder."""
        try:
            model_config = decoder_config.get("model_config", {})
            model_type = model_config.get("model_type", "EEGNet")
            decoder_params = model_config.copy()

            if override_num_classes is not None:
                num_classes = override_num_classes
            else:
                num_classes = (
                    len(self.label_mapping) if self.label_mapping
                    else model_config.get("num_classes", 2)
                )
            decoder_params["num_classes"] = num_classes
            self.logger.info(f"Creating {model_type} decoder with {num_classes} classes")

            decoder = create_decoder(**decoder_params)
            if self.modality_labels:
                decoder.channel_labels = self.modality_labels
            decoder.sample_rate = self.sample_rate

            preproc = self.instance_config.get("mode_preprocessing")
            if preproc:
                from dendrite.processing.preprocessing.preprocessing_schemas import (
                    PreprocessingConfig,
                )
                decoder.config.preprocessing_config = PreprocessingConfig(
                    modality_preprocessing=preproc
                )

            self.decoder = decoder
            self.logger.info("Decoder created successfully")
            return decoder

        except Exception as e:
            self.logger.error(f"Error creating decoder: {e}", exc_info=True)
            self.decoder = None
            return None

    def _validate_decoder_channels(
        self, decoder,
    ) -> tuple[str | None, int | None, int | None]:
        """Validate decoder input shapes against mode's channel selection.

        Returns (primary_modality, expected_channels, expected_time_samples).
        Returns (None, None, None) when shapes or channel_selection are absent.
        Raises ValueError on channel count mismatch.
        """
        shapes = decoder.input_shapes or {}
        if not shapes or not self.channel_selection:
            return None, None, None

        primary = next(
            (m for m in self.channel_selection if m in shapes), None,
        )
        if not primary:
            return None, None, None

        expected_ch = shapes[primary][0]
        actual_ch = len(self.channel_selection[primary])
        if actual_ch != expected_ch:
            raise ValueError(
                f"Channel mismatch: decoder expects {expected_ch}, "
                f"mode has {actual_ch}"
            )

        expected_t = shapes[primary][1] if len(shapes[primary]) > 1 else None
        return primary, expected_ch, expected_t

    def _start_background_decoder_load(self, path: str, result_info: dict):
        """Load decoder from disk in a background thread.

        Stores the loaded decoder in ``_pending_decoder_load`` for the main
        loop to pick up.  File I/O in ``load_decoder`` releases the GIL so
        the main loop continues processing data.
        """
        import threading

        def _bg_load():
            try:
                decoder = load_decoder(path)
                try:
                    self._validate_decoder_channels(decoder)
                except ValueError as e:
                    self.logger.error(f"Decoder rejected: {e}")
                    return
                self._pending_decoder_load = {**result_info, "_decoder": decoder}
                self.logger.info("Decoder loaded from disk, pending activation")
            except Exception as e:
                self.logger.error(f"Failed to load decoder from {path}: {e}")

        threading.Thread(target=_bg_load, daemon=True, name="DecoderLoad").start()

    def _load_decoder(self, model_path: str, source: str = "decoder") -> bool:
        """Load decoder from path. Returns True on success."""
        if not model_path.endswith(".json") and os.path.exists(model_path + ".json"):
            model_path += ".json"

        self.logger.info(f"Loading {source} from {model_path}")
        try:
            self.decoder = load_decoder(model_path)
            self.logger.info(f"Successfully loaded {source}")

            # Inherit event mapping from decoder if mode has none
            if not self.event_mapping and self.decoder.event_mapping:
                self.event_mapping = self.decoder.event_mapping
                self._generate_label_mapping(self.event_mapping)
                self.logger.info("Inherited event mapping from decoder")

            return True
        except (FileNotFoundError, RuntimeError) as e:
            self.logger.error(f"Error loading {source}: {e}")
            return False

    def _predict(self, X_input) -> tuple[int, float, float]:
        """Run prediction with timing. Returns (prediction, confidence, inference_ms)."""
        if not self.is_decoder_ready:
            return 0, 0.5, 0.0

        X_array = X_input[next(iter(X_input))] if isinstance(X_input, dict) else X_input

        try:
            start_ns = time.perf_counter_ns()
            prediction, confidence = self.decoder.predict_sample(X_array)
            inference_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            if self.shared_state and inference_ms > 0:
                self.shared_state.set(
                    mode_metric_key(self.mode_name, "inference_ms"), inference_ms
                )
            return prediction, confidence, inference_ms
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0, 0.5, 0.0

    def _update_gpu_metrics(self) -> None:
        """Track GPU memory if using CUDA. Throttled."""
        if not self.decoder or not self.shared_state:
            return
        if (time.time() - self._gpu_last_emit_time) < MODE_GPU_EMIT_INTERVAL:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            self.shared_state.set(
                mode_metric_key(self.mode_name, "gpu_mb"), float(allocated_mb)
            )
            self._gpu_last_emit_time = time.time()
        except Exception:
            pass

    def _generate_label_mapping(self, event_mapping: dict):
        """Generate label mapping from event mapping."""
        self.label_mapping, self.reverse_label_mapping, self.index_to_event_code = (
            generate_label_mapping(event_mapping)
        )

    def _setup_preprocessor(self):
        """Create SamplePreprocessor from instance config. Sets effective_sample_rate."""
        preproc_config = self.instance_config.get("mode_preprocessing", {})
        if not preproc_config:
            self._sample_preprocessor = None
            self.effective_sample_rate = self.sample_rate
            return

        self._sample_preprocessor = SamplePreprocessor(
            preproc_config=preproc_config,
            sample_rate=self.sample_rate,
            channel_selection=self.channel_selection,
            modality_labels=self.modality_labels,
            shared_state=self.shared_state,
            logger=self.logger,
        )
        self.effective_sample_rate = self._sample_preprocessor.effective_sample_rate

    def _preprocess_sample(self, sample: Sample) -> Sample | None:
        """Delegate to SamplePreprocessor. Returns None if accumulating for downsample."""
        if self._sample_preprocessor is None:
            return sample
        try:
            return self._sample_preprocessor.process(sample)
        except ValueError as e:
            self.logger.error(f"{e} — stopping mode.")
            self.stop_event.set()
            return None

