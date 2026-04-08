"""
Pipeline Orchestrator

Manages pipeline component subprocesses with independent lifecycles.
DAQ writes to shared memory ring buffers; all consumers read directly.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any

from dendrite import __version__
from dendrite.constants import (
    QUEUE_SIZE_LARGE,
    TIMEOUT_DATA_ACQUISITION,
    TIMEOUT_DATA_SAVER,
    TIMEOUT_METRICS_SAVER,
    TIMEOUT_MODE_PROCESS,
    get_study_paths,
)
from dendrite.data.acquisition import DataAcquisition
from dendrite.data.shared_buffers import SharedRingBuffer, compute_max_samples
from dendrite.data.storage.data_saver import DataSaver
from dendrite.data.storage.metrics_saver import MetricsSaver
from dendrite.processing.modes import create_mode
from dendrite.processing.modes.mode_utils import FanOutQueue
from dendrite.utils.logger_central import setup_logger
from dendrite.utils.state_keys import component_error_key, component_state_key


class PipelineOrchestrator:
    """Manages pipeline component subprocesses.

    Architecture: DAQ → SharedRingBuffer ← consumers (modes, viz)
    No DataProcessor — all consumers read shared memory directly.
    """

    def __init__(self, shared_state, logger=None):
        self.shared_state = shared_state
        self.logger = logger or setup_logger("Orchestrator", level=logging.INFO)

        # Core processes
        self._daq_process: multiprocessing.Process | None = None
        self._daq_stop = multiprocessing.Event()
        self._data_saver_process: multiprocessing.Process | None = None
        self._data_saver_stop = multiprocessing.Event()
        self._metrics_saver_process: multiprocessing.Process | None = None
        self._metrics_saver_stop = multiprocessing.Event()

        # Queues (low-frequency only — mode outputs, events)
        self._event_queue: multiprocessing.Queue | None = None
        self._prediction_queue: multiprocessing.Queue | None = None
        self._visualization_queue: multiprocessing.Queue | None = None
        self._training_queue: multiprocessing.Queue | None = None
        self._pid_queue: multiprocessing.Queue | None = None
        self._shared_metrics_queue: multiprocessing.Queue | None = None

        # Mode tracking
        self._mode_processes: dict[str, multiprocessing.Process] = {}
        self._mode_stops: dict[str, multiprocessing.Event] = {}
        self._mode_output_queues: dict[str, FanOutQueue] = {}

        # Shared memory ring buffers (owned by orchestrator)
        self._ring_buffers: dict[str, SharedRingBuffer] = {}
        self._ring_buffer_channel_maps: dict[str, dict] = {}

        # Session config
        self._file_identifier: str = ""
        self._study_name: str = ""
        self._stream_configs: list = []

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def start_core(
        self,
        *,
        file_identifier: str,
        stream_configs: list,
        study_name: str = "",
        recording_name: str = "",
        subject_id: str = "",
        session_id: str = "",
        run_number: int = 1,
        experiment_description: str = "",
        mode_instances: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Start core pipeline: DAQ (ring buffers), DataSaver (reads buffers + events), MetricsSaver."""
        mode_instances = mode_instances or {}

        self._file_identifier = file_identifier
        self._study_name = study_name
        self._stream_configs = stream_configs

        # Create queues (low-frequency only)
        self._event_queue = multiprocessing.Queue()
        self._prediction_queue = multiprocessing.Queue()
        self._training_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE)
        self._pid_queue = multiprocessing.Queue()
        self._visualization_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE)
        self._shared_metrics_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE)

        # Reset stop events
        self._daq_stop.clear()
        self._data_saver_stop.clear()
        self._metrics_saver_stop.clear()

        # Prepare directories
        paths = get_study_paths(study_name)
        paths["raw"].mkdir(parents=True, exist_ok=True)
        paths["metrics"].mkdir(parents=True, exist_ok=True)
        raw_data_filename = str(paths["raw"] / f"{file_identifier}_raw.h5")
        metrics_filename = str(paths["metrics"] / f"{file_identifier}_metrics.h5")

        # Create shared memory ring buffers
        self._create_shared_buffers(stream_configs)

        # 1. Start DAQ
        ring_buffer_names = {st: rb.name for st, rb in self._ring_buffers.items()}
        self._daq_process = DataAcquisition(
            event_queue=self._event_queue,
            stop_event=self._daq_stop,
            stream_configs=stream_configs,
            shared_state=self.shared_state,
            ring_buffer_names=ring_buffer_names,
        )
        self._daq_process.daemon = True
        self._daq_process.start()
        self.logger.info("DAQ started")

        # 2. Start DataSaver
        self._data_saver_process = DataSaver(
            filename=raw_data_filename,
            stop_event=self._data_saver_stop,
            shared_state=self.shared_state,
            ring_buffer_names=ring_buffer_names,
            ring_buffer_channel_maps=self._ring_buffer_channel_maps,
            event_queue=self._event_queue,
            global_metadata={
                "version": __version__,
                "study_name": study_name,
                "recording_name": recording_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "run_number": run_number,
                "file_identifier": file_identifier,
                "experiment_description": experiment_description,
            },
            stream_configs=stream_configs,
        )
        self._data_saver_process.daemon = True
        self._data_saver_process.start()
        self.logger.info("DataSaver started")

        # 3. Create output queues for initial modes
        for name in mode_instances:
            self._mode_output_queues[name] = FanOutQueue(
                [self._shared_metrics_queue, self._visualization_queue]
            )

        # 4. Start MetricsSaver
        self._metrics_saver_process = MetricsSaver(
            filename=metrics_filename,
            stop_event=self._metrics_saver_stop,
            script_metadata={
                "version": __version__,
                "study_name": study_name,
                "recording_name": recording_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "run_number": run_number,
                "file_identifier": file_identifier,
                "mode_instances": {},
            },
            metrics_queue=self._shared_metrics_queue,
            shared_state=self.shared_state,
        )
        self._metrics_saver_process.daemon = True
        self._metrics_saver_process.start()
        self.logger.info("MetricsSaver started")

        self.logger.info("Core pipeline started successfully")

    def start_mode(
        self,
        instance_name: str,
        instance_config: dict[str, Any],
    ) -> int | None:
        """Start a mode subprocess that reads from shared ring buffer."""
        if instance_name in self._mode_processes:
            self.logger.warning(f"Mode {instance_name} already running")
            return self._mode_processes[instance_name].pid

        mode_type = instance_config["mode"]
        mode_stop = multiprocessing.Event()

        if instance_name not in self._mode_output_queues:
            self._mode_output_queues[instance_name] = FanOutQueue(
                [self._shared_metrics_queue, self._visualization_queue]
            )

        # Inject ring buffer config into mode config
        enhanced_config = instance_config.copy()
        enhanced_config.update({
            "file_identifier": self._file_identifier,
            "study_name": self._study_name,
        })

        # Find ring buffer that contains the mode's selected modality
        required = list(instance_config.get("channel_selection", {}).keys())
        preferred = instance_config.get("source_stream")
        if required:
            for stream_key, channel_map in self._ring_buffer_channel_maps.items():
                if preferred and stream_key != preferred:
                    continue
                if any(mod in channel_map["modalities"] for mod in required):
                    enhanced_config["ring_buffer"] = channel_map
                    break
            if "ring_buffer" not in enhanced_config:
                self.logger.warning(
                    f"Mode {instance_name}: no ring buffer found for modalities {required}"
                )
        else:
            self.logger.warning(f"Mode {instance_name}: no channel_selection, skipping buffer")

        try:
            mode_params = {
                "output_queue": self._mode_output_queues[instance_name],
                "stop_event": mode_stop,
                "instance_config": enhanced_config,
                "prediction_queue": self._prediction_queue,
                "shared_state": self.shared_state,
                "training_queue": self._training_queue,
            }

            proc = create_mode(mode_type, **mode_params)
            proc.daemon = True
            proc.start()

            if proc.is_alive():
                self._mode_processes[instance_name] = proc
                self._mode_stops[instance_name] = mode_stop
                self.logger.info(f"Mode {instance_name} started (PID={proc.pid})")
                if self._pid_queue:
                    try:
                        self._pid_queue.put({"mode_name": instance_name, "pid": proc.pid})
                    except Exception:
                        pass
                return proc.pid
            else:
                self.logger.error(f"Mode {instance_name} failed to start")
                return None

        except Exception as e:
            self.logger.error(f"Failed to start mode {instance_name}: {e}", exc_info=True)
            self._mode_output_queues.pop(instance_name, None)
            return None

    def stop_mode(self, instance_name: str) -> None:
        """Stop a mode subprocess."""
        stop_event = self._mode_stops.pop(instance_name, None)
        process = self._mode_processes.pop(instance_name, None)
        self._mode_output_queues.pop(instance_name, None)

        if stop_event:
            stop_event.set()
        if process and process.is_alive():
            process.join(timeout=TIMEOUT_MODE_PROCESS)
            if process.is_alive():
                self.logger.warning(f"Mode {instance_name} didn't stop cleanly, terminating")
                process.terminate()
                process.join(timeout=1)

        self.logger.info(f"Mode {instance_name} stopped")

    def stop_all(self) -> None:
        """Coordinated shutdown of all components."""
        self.logger.info("Stopping all pipeline components...")

        # 1. Stop DAQ (stop producing data)
        self._daq_stop.set()

        # 2. Stop all modes
        for name in list(self._mode_processes.keys()):
            self.stop_mode(name)

        # 3. Stop MetricsSaver
        self._metrics_saver_stop.set()
        if self._metrics_saver_process and self._metrics_saver_process.is_alive():
            self._metrics_saver_process.join(timeout=TIMEOUT_METRICS_SAVER)
            if self._metrics_saver_process.is_alive():
                self._metrics_saver_process.terminate()
                self._metrics_saver_process.join(timeout=1)

        # 4. Stop DataSaver (last — flushes remaining ring buffer data + events)
        self._data_saver_stop.set()
        if self._data_saver_process and self._data_saver_process.is_alive():
            self._data_saver_process.join(timeout=TIMEOUT_DATA_SAVER)
            if self._data_saver_process.is_alive():
                self.logger.warning("DataSaver didn't stop cleanly, terminating")
                self._data_saver_process.terminate()
                self._data_saver_process.join(timeout=1)
                # On Windows, terminate() is a hard kill — clear dirty HDF5 flags
                self._clear_h5_flags(self._data_saver_process.filename)

        # Wait for DAQ
        if self._daq_process and self._daq_process.is_alive():
            self._daq_process.join(timeout=TIMEOUT_DATA_ACQUISITION)
            if self._daq_process.is_alive():
                self._daq_process.terminate()
                self._daq_process.join(timeout=1)

        self._cleanup_shared_buffers()
        self._cleanup_queues()
        self.logger.info("All pipeline components stopped")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_core_running(self) -> bool:
        return self._daq_process is not None and self._daq_process.is_alive()

    def get_component_states(self) -> dict[str, str]:
        states = {}
        component_ids = ["daq", "data_saver", "metrics_saver"]
        for name in self._mode_processes:
            component_ids.append(f"mode:{name}")
        for cid in component_ids:
            state = self.shared_state.get(component_state_key(cid))
            if state:
                states[cid] = state
        return states

    def get_mode_pids(self) -> dict[str, int]:
        return {name: proc.pid for name, proc in self._mode_processes.items() if proc.is_alive()}

    @property
    def core_pids(self) -> dict[str, int]:
        pids: dict[str, int] = {}
        if self._daq_process and self._daq_process.is_alive():
            pids["DAQ"] = self._daq_process.pid
        if self._data_saver_process and self._data_saver_process.is_alive():
            pids["DataSaver"] = self._data_saver_process.pid
        if self._metrics_saver_process and self._metrics_saver_process.is_alive():
            pids["MetricsSaver"] = self._metrics_saver_process.pid
        return pids

    def is_mode_running(self, instance_name: str) -> bool:
        proc = self._mode_processes.get(instance_name)
        return proc is not None and proc.is_alive()

    def check_mode_health(self) -> list[str]:
        """Return names of dead mode processes and clean them up."""
        dead: list[str] = []
        for name, proc in list(self._mode_processes.items()):
            if not proc.is_alive():
                dead.append(name)
                self._mode_processes.pop(name, None)
                self._mode_stops.pop(name, None)
                self._mode_output_queues.pop(name, None)
                # Clear SharedState keys for this mode
                if self.shared_state:
                    self.shared_state.clear(component_state_key(f"mode:{name}"))
                    self.shared_state.clear(component_error_key(f"mode:{name}"))
                self.logger.error(f"Mode {name} (PID={proc.pid}) crashed")
        return dead

    # ------------------------------------------------------------------
    # Queue/buffer accessors
    # ------------------------------------------------------------------

    @property
    def event_queue(self) -> multiprocessing.Queue | None:
        return self._event_queue

    @property
    def prediction_queue(self) -> multiprocessing.Queue | None:
        return self._prediction_queue

    @property
    def training_queue(self) -> multiprocessing.Queue | None:
        return self._training_queue

    @property
    def visualization_queue(self) -> multiprocessing.Queue | None:
        return self._visualization_queue

    @property
    def pid_queue(self) -> multiprocessing.Queue | None:
        return self._pid_queue

    @property
    def mode_output_queues(self) -> dict[str, FanOutQueue]:
        return self._mode_output_queues

    @property
    def ring_buffer_channel_maps(self) -> dict[str, dict]:
        return self._ring_buffer_channel_maps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear_h5_flags(self, filename: str) -> None:
        """Clear dirty HDF5 consistency flags after a force-kill."""
        import h5py

        try:
            h5py.File(filename, "r+", libver="latest").close()
            self.logger.info(f"Cleared HDF5 consistency flags: {filename}")
        except Exception as e:
            self.logger.warning(f"Could not clear HDF5 flags for {filename}: {e}")

    def _create_shared_buffers(self, stream_configs: list) -> None:
        """Create ring buffers: raw_channels + 1 markers column per stream."""
        self._ring_buffers.clear()
        self._ring_buffer_channel_maps.clear()

        for config in stream_configs:
            if config.type.upper() == "EVENTS":
                continue
            if getattr(config, "channel_format", None) == "string":
                continue

            key = config.stream_key or config.type
            raw_channels = config.channel_count
            buf_channels = raw_channels + 1
            marker_col = buf_channels - 1
            sr = config.sample_rate or 500.0
            max_samples = compute_max_samples(sr)
            buf_name = f"dendrite_rb_{key.lower()}"

            try:
                rb = SharedRingBuffer.create(buf_name, buf_channels, max_samples, sr)
            except Exception as e:
                self.logger.error(f"Failed to create ring buffer for {key}: {e}")
                continue

            self._ring_buffers[key] = rb

            channel_types = config.channel_types or []
            modalities: dict[str, list[int]] = {}
            if channel_types:
                for i, ch_type in enumerate(channel_types):
                    if ch_type.lower() != "markers":
                        modalities.setdefault(ch_type.lower(), []).append(i)
            else:
                modalities[config.type.lower()] = list(range(raw_channels))

            # Per-modality channel labels for interpolation montage lookup
            modality_labels: dict[str, list[str]] = {}
            if config.labels:
                for mod, indices in modalities.items():
                    modality_labels[mod] = [config.labels[i] for i in indices]

            self._ring_buffer_channel_maps[key] = {
                "buffer_name": buf_name,
                "modalities": modalities,
                "modality_labels": modality_labels,
                "marker_col": marker_col,
                "sample_rate": sr,
            }
            self.logger.info(f"Ring buffer: {buf_name} ({buf_channels}ch, {max_samples}s)")

    def _cleanup_shared_buffers(self) -> None:
        for rb in self._ring_buffers.values():
            try:
                rb.close()
                rb.unlink()
            except Exception:
                pass
        self._ring_buffers.clear()
        self._ring_buffer_channel_maps.clear()

    def _cleanup_queues(self) -> None:
        for q in [
            self._event_queue,
            self._prediction_queue,
            self._training_queue,
            self._pid_queue,
            self._visualization_queue,
            self._shared_metrics_queue,
        ]:
            if q:
                try:
                    q.close()
                    q.cancel_join_thread()
                except Exception:
                    pass
        self._mode_output_queues.clear()
