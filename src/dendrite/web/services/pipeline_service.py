"""
Pipeline Service

Owns the entire pipeline lifecycle between Start and Stop.
Uses PipelineOrchestrator for independent component management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import queue
import sqlite3
import time
from datetime import datetime
from multiprocessing.queues import Queue as MpQueue
from typing import Any

from dendrite.constants import get_study_paths
from dendrite.data.storage.database import Database, RecordingRepository, StudyRepository
from dendrite.data.streaming.output_protocol_manager import OutputProtocolManager
from dendrite.processing.orchestrator import PipelineOrchestrator
from dendrite.processing.pipeline_schemas import PipelineConfig
from dendrite.utils import SharedState
from dendrite.utils.logger_central import (
    configure_file_logging,
    get_logger,
    set_study_name,
    setup_logger,
)


class PipelineService:
    """Controls the processing pipeline lifecycle.

    Uses PipelineOrchestrator for per-component process management.
    Supports dynamic mode start/stop during a recording session.
    """

    def __init__(self):
        self.logger = setup_logger("PipelineService", level=logging.INFO)

        # Orchestrator (created per session)
        self._orchestrator: PipelineOrchestrator | None = None

        # Output protocols
        self._stop_event = multiprocessing.Event()
        self._output_protocol_manager: OutputProtocolManager | None = None

        # PID tracking
        self._mode_pids: dict[str, int] = {}
        self._system_processes: dict[str, Any] = {}
        self._pid_task: asyncio.Task | None = None

        # Shared state
        self._shared_state = None

        # Stream metadata (set at start, used by visualization bridge)
        self._stream_configs: list = []

        # Session metadata
        self._log_file: str | None = None
        self._start_time: float | None = None
        self._study_name: str | None = None
        self._recording_id: int | None = None
        self._run_number: int | None = None

        # Viz preprocessing config (mutable during recording)
        self._viz_preproc_config: dict = {}

        # State events (replace Qt signals)
        self._recording_event = asyncio.Event()
        self._stopped_event = asyncio.Event()

        # Online training task (started/stopped with pipeline)
        self._training_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, config: PipelineConfig) -> None:
        """Start the processing pipeline.

        Three-phase async startup:
        1. Session I/O (DB, file ops, SharedState)
        2. Start core components via orchestrator (DAQ, Processor, DataSaver)
        3. Start initial modes and output protocols

        Args:
            config: Validated PipelineConfig from config_service.build_configuration()

        Raises:
            RuntimeError: If already recording or startup fails.
        """
        if self.is_recording:
            raise RuntimeError("Pipeline is already recording")

        self._stopped_event.clear()
        self._recording_event.clear()
        self._study_name = config.study_name

        # Phase 1: Background I/O
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_session_io,
                    config.subject_id,
                    config.session_id,
                    config.recording_name,
                    config.study_name,
                    config.model_dump(mode="json"),
                ),
                timeout=30,
            )
        except TimeoutError as e:
            raise RuntimeError("Session I/O timed out after 30s") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize session: {e}") from e

        self._log_file = result.get("log_file")
        self._recording_id = result.get("recording_id")
        self._run_number = result.get("run_number")
        self._shared_state = result["shared_state"]
        self._stream_configs = config.stream_configs

        # Phase 2: Start core via orchestrator
        try:
            self._orchestrator = PipelineOrchestrator(
                shared_state=self._shared_state, logger=self.logger
            )

            self._orchestrator.start_core(
                file_identifier=result["file_identifier"],
                stream_configs=config.stream_configs,
                study_name=config.study_name,
                recording_name=config.recording_name,
                subject_id=config.subject_id,
                session_id=config.session_id,
                run_number=result["run_number"],
                experiment_description=config.experiment_description,
                mode_instances=config.mode_instances,
            )

            # Phase 3: Start initial modes
            for instance_name, instance_config in config.mode_instances.items():
                pid = self._orchestrator.start_mode(instance_name, instance_config)
                if pid:
                    self._mode_pids[instance_name] = pid

            # Start output protocols
            self._stop_event.clear()
            self._start_output_protocols(config.model_dump(mode="json"))
            self._start_pid_collection()
            self._start_time = time.monotonic()
            self._recording_event.set()

            # Start online training loop (drains training queue from modes)
            training_q = self.training_queue
            shared_state = self.shared_state
            if training_q and shared_state:
                from dendrite.web.deps import get_ml_service
                self._training_task = asyncio.create_task(
                    get_ml_service().run_online_training_loop(training_q, shared_state)
                )

            self.logger.info("Pipeline started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            if self._orchestrator:
                self._orchestrator.stop_all()
                self._orchestrator = None
            raise RuntimeError(str(e)) from e

    async def stop(self) -> None:
        """Stop the processing pipeline gracefully."""
        if not self.is_recording:
            return

        self.logger.info("Stopping pipeline...")

        # Cancel online training loop
        if self._training_task:
            self._training_task.cancel()
            await asyncio.gather(self._training_task, return_exceptions=True)
            self._training_task = None

        self._stop_event.set()

        if self._output_protocol_manager:
            self._output_protocol_manager.signal_stop()

        # Stop output protocols
        stop_targets = []
        if self._output_protocol_manager:
            stop_targets.extend(self._output_protocol_manager.get_stop_targets())
        if stop_targets:
            await self._shutdown_targets(stop_targets)

        # Stop all pipeline components via orchestrator
        if self._orchestrator:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._orchestrator.stop_all), timeout=15
                )
            except TimeoutError:
                self.logger.error("Orchestrator stop timed out after 15s")

        self._finalize_stop()

    async def start_mode(self, instance_name: str, instance_config: dict) -> int | None:
        """Start a mode during a recording session.

        Args:
            instance_name: Unique mode instance name
            instance_config: Full mode configuration dict

        Returns:
            PID of started mode, or None on failure.

        Raises:
            RuntimeError: If not recording or orchestrator not available.
        """
        if not self.is_recording or not self._orchestrator:
            raise RuntimeError("Cannot start mode: pipeline not recording")

        pid = await asyncio.to_thread(
            self._orchestrator.start_mode, instance_name, instance_config
        )
        if pid:
            self._mode_pids[instance_name] = pid
        return pid

    async def stop_mode(self, instance_name: str) -> None:
        """Stop a mode during a recording session.

        Args:
            instance_name: Mode instance name to stop.

        Raises:
            RuntimeError: If not recording or orchestrator not available.
        """
        if not self._orchestrator:
            raise RuntimeError("Cannot stop mode: no orchestrator")

        await asyncio.to_thread(self._orchestrator.stop_mode, instance_name)
        self._mode_pids.pop(instance_name, None)

    @property
    def is_recording(self) -> bool:
        if self._orchestrator is None:
            return False
        return self._orchestrator.is_core_running

    def is_mode_running(self, instance_name: str) -> bool:
        """Check if a specific mode is currently running."""
        if not self._orchestrator:
            return False
        return self._orchestrator.is_mode_running(instance_name)

    @property
    def shared_state(self):
        return self._shared_state

    @property
    def study_name(self) -> str | None:
        return self._study_name

    @property
    def recording_id(self) -> int | None:
        return self._recording_id

    @property
    def run_number(self) -> int | None:
        return self._run_number

    @property
    def log_file(self) -> str | None:
        return self._log_file

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None or not self.is_recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def mode_pids(self) -> dict[str, int]:
        if self._orchestrator:
            return self._orchestrator.get_mode_pids()
        return self._mode_pids.copy()

    def get_session_events(self) -> dict[str, Any]:
        """Read unique event codes and names from the live recording file."""
        if not self._shared_state:
            return {"events": [], "event_mapping": {}, "total": 0}
        recording_file = self._shared_state.get("recording_file")
        if not recording_file:
            return {"events": [], "event_mapping": {}, "total": 0}

        import h5py
        try:
            with h5py.File(recording_file, "r", swmr=True) as h5f:
                if "Event" not in h5f:
                    return {"events": [], "event_mapping": {}, "total": 0}
                event_data = h5f["Event"]
                if not isinstance(event_data, h5py.Dataset) or len(event_data) == 0:
                    return {"events": [], "event_mapping": {}, "total": 0}
                event_data.refresh()
                event_map: dict[int, str] = {}
                for e in event_data:
                    code = int(e["event_id"])
                    name = e["event_type"]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    if code not in event_map:
                        event_map[code] = name
                return {
                    "events": sorted(event_map.keys()),
                    "event_mapping": {k: event_map[k] for k in sorted(event_map)},
                    "total": len(event_data),
                }
        except Exception:
            return {"events": [], "event_mapping": {}, "total": 0}

    @property
    def configured_stream_types(self) -> list[str]:
        """Get unique stream types from current configuration."""
        return list({sc.type for sc in self._stream_configs})

    @property
    def system_pids(self) -> dict[str, int]:
        pids: dict[str, int] = {
            name: proc.pid  # type: ignore[misc]
            for name, proc in self._system_processes.items()
            if proc and proc.is_alive()
        }
        if self._orchestrator:
            pids.update(self._orchestrator.core_pids)
        return pids

    @property
    def visualization_data_queue(self):
        """Ring buffer channel maps for VizConsumer to read raw data."""
        if self._orchestrator:
            return self._orchestrator.ring_buffer_channel_maps
        return None

    @property
    def visualization_queue(self) -> MpQueue[Any] | None:
        """Expose mode output visualization queue for WebSocket bridge."""
        if self._orchestrator:
            return self._orchestrator.visualization_queue
        return None

    @property
    def training_queue(self) -> MpQueue[Any] | None:
        """Expose training queue for MLService online training loop."""
        if self._orchestrator:
            return self._orchestrator.training_queue
        return None

    @property
    def recording_event(self) -> asyncio.Event:
        return self._recording_event

    @property
    def stopped_event(self) -> asyncio.Event:
        return self._stopped_event

    @property
    def viz_sample_rate(self) -> int:
        """Target visualization sample rate (all streams decimate to this)."""
        from dendrite.web.ws.visualization_bridge import TARGET_VIZ_RATE
        return TARGET_VIZ_RATE

    def get_component_states(self) -> dict[str, str]:
        """Get states of all pipeline components."""
        if self._orchestrator:
            return self._orchestrator.get_component_states()
        return {}

    @property
    def viz_preproc_config(self) -> dict:
        return self._viz_preproc_config

    def set_viz_preproc_config(self, config: dict) -> None:
        self._viz_preproc_config = config

    def cleanup(self) -> None:
        """Final cleanup."""
        if self._shared_state:
            self._shared_state.cleanup()
            self._shared_state = None

    # ------------------------------------------------------------------
    # Output protocols
    # ------------------------------------------------------------------

    def _start_output_protocols(self, config: dict):
        if self._orchestrator is None or self._orchestrator.prediction_queue is None:
            raise RuntimeError("Cannot start output protocols: pipeline not initialized")
        self._output_protocol_manager = OutputProtocolManager(
            stop_event=self._stop_event,
            prediction_queue=self._orchestrator.prediction_queue,
            shared_state=self._shared_state,
        )
        streamer_processes = self._output_protocol_manager.initialize(config)
        self._system_processes.update(streamer_processes)

    # ------------------------------------------------------------------
    # PID collection (async loop replaces QTimer)
    # ------------------------------------------------------------------

    def _start_pid_collection(self):
        self._mode_pids = {}
        pid_queue = self._orchestrator.pid_queue if self._orchestrator else None
        if pid_queue:
            self._pid_task = asyncio.ensure_future(self._pid_collection_loop(pid_queue))

    async def _pid_collection_loop(self, pid_queue):
        """Collect mode PIDs and check mode health every 500ms."""
        while self.is_recording:
            try:
                while not pid_queue.empty():
                    pid_info = pid_queue.get_nowait()
                    mode_name = pid_info["mode_name"]
                    pid = pid_info["pid"]

                    if mode_name not in self._mode_pids:
                        self.logger.info(f"Collected PID {pid} for mode {mode_name}")
                    self._mode_pids[mode_name] = pid
            except (queue.Empty, KeyError):
                pass

            # Check for crashed modes (cleanup happens in orchestrator,
            # frontend sees mode vanish from component states via telemetry)
            if self._orchestrator and self._mode_pids:
                for name in self._orchestrator.check_mode_health():
                    pid = self._mode_pids.pop(name, None)
                    self.logger.error(f"Mode {name} (PID={pid}) crashed")

            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Session I/O (runs in background thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _run_session_io(
        self,
        subject_id: str,
        session_id: str,
        recording_name: str,
        study_name: str,
        config_to_save: dict,
    ) -> dict:
        """Execute all blocking I/O for session startup."""
        logger = get_logger("SessionIO")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        db = Database()
        repo = RecordingRepository(db)
        run_number = repo.get_next_run_number(subject_id, session_id, recording_name)

        # Build BIDS file identifier
        parts = []
        if subject_id:
            parts.append(f"sub-{subject_id}")
        if session_id:
            parts.append(f"ses-{session_id}")
        parts.append(f"task-{recording_name}")
        parts.append(f"run-{run_number:02d}")
        parts.append(timestamp)
        file_identifier = "_".join(parts)

        config_to_save = {
            **config_to_save, "run_number": run_number, "file_identifier": file_identifier,
        }

        # Setup session logging
        set_study_name(study_name)
        log_file = configure_file_logging(file_identifier=file_identifier, level=logging.DEBUG)
        logger.info(f"Session: {file_identifier} | Study: {study_name}")

        # Save configuration to file
        paths = get_study_paths(study_name)
        config_dir = paths["config"]
        os.makedirs(config_dir, exist_ok=True)
        config_path = config_dir / f"{file_identifier}_config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)
        logger.info(f"Configuration saved: {config_path}")

        # Add recording to database
        recording_id = self._add_recording_to_database(
            db, study_name, subject_id, session_id,
            recording_name, timestamp, file_identifier, run_number, logger,
        )

        shared_state = SharedState()

        return {
            "run_number": run_number,
            "recording_id": recording_id,
            "file_identifier": file_identifier,
            "timestamp": timestamp,
            "log_file": log_file,
            "shared_state": shared_state,
        }

    @staticmethod
    def _add_recording_to_database(
        db: Database,
        study_name: str,
        subject_id: str,
        session_id: str,
        recording_name: str,
        timestamp: str,
        file_identifier: str,
        run_number: int,
        logger: logging.Logger,
    ) -> int | None:
        paths = get_study_paths(study_name)
        hdf5_path = str(paths["raw"] / f"{file_identifier}_raw.h5")
        try:
            study_repo = StudyRepository(db)
            recording_repo = RecordingRepository(db)
            study = study_repo.get_or_create(study_name)
            study_id = study["study_id"]
            recording_id = recording_repo.add_recording(
                study_id=study_id,
                recording_name=recording_name,
                session_timestamp=timestamp,
                hdf5_file_path=hdf5_path,
                subject_id=subject_id,
                session_id=session_id,
                run_number=run_number,
                file_identifier=file_identifier,
            )
            if recording_id:
                logger.info(f"Recording added to database with ID: {recording_id}")
            else:
                logger.warning("Failed to add recording to database")
            return recording_id
        except (OSError, sqlite3.Error) as e:
            logger.error(f"Database error: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown_targets(self, targets: list[tuple[str, Any, float]]) -> None:
        """Shut down processes/threads in order with per-target timeouts."""
        for name, target, timeout in targets:
            self.logger.info(f"Stopping {name}...")
            start = time.monotonic()
            while target.is_alive() and (time.monotonic() - start) < timeout:
                await asyncio.sleep(0.05)
            if target.is_alive():
                self.logger.warning(f"Force terminating {name}")
                if hasattr(target, "terminate"):
                    target.terminate()
        self.logger.info("All targets stopped")

    def _finalize_stop(self):
        self._orchestrator = None

        if self._output_protocol_manager:
            self._output_protocol_manager.cleanup()

        if self._pid_task and not self._pid_task.done():
            self._pid_task.cancel()
        self._pid_task = None

        self._mode_pids = {}
        self._system_processes = {}

        if self._shared_state:
            self._shared_state.cleanup()
            self._shared_state = None

        self._viz_preproc_config = {}
        self._stream_configs = []
        self._start_time = None
        self._log_file = None
        self._study_name = None
        self._recording_id = None
        self._run_number = None
        self._recording_event.clear()
        self._stopped_event.set()
        self.logger.info("Pipeline stopped")
