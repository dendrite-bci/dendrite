"""
Pipeline Controller

Owns the entire pipeline lifecycle between Start and Stop:
queue creation, process spawning, PID monitoring, shutdown sequencing,
and IO thread management. Communicates back to the UI via Qt signals only.
"""

import logging
import multiprocessing
import queue

from PyQt6 import QtCore

from dendrite.constants import (
    DEFAULT_RECORDING_NAME, DEFAULT_STUDY_NAME,
    QUEUE_SIZE_LARGE, TIMEOUT_MAIN_PROCESS, TIMEOUT_VISUALIZATION,
    VISUALIZATION_STREAM_INFO, PID_COLLECTION_INTERVAL_MS,
)
from dendrite.data.streaming import VisualizationStreamer
from dendrite.gui.output_protocol_manager import OutputProtocolManager
from dendrite.gui.shutdown_sequencer import ShutdownSequencer
from dendrite.gui.workers import SessionIOWorker
from dendrite.processing.pipeline import run_pipeline
from dendrite.processing.pipeline_config import PipelineConfig
from dendrite.processing.queue_utils import FanOutQueue
from dendrite.utils.logger_central import setup_logger


class PipelineController(QtCore.QObject):
    """Controls the processing pipeline lifecycle.

    Owns all process/queue/IPC state between Start and Stop.
    Communicates results back via Qt signals — holds zero widget references.
    """

    recording_started = QtCore.pyqtSignal()
    recording_stopped = QtCore.pyqtSignal()
    start_failed = QtCore.pyqtSignal(str)          # error message
    pids_updated = QtCore.pyqtSignal(dict, dict)    # mode_pids, system_pids
    log_file_ready = QtCore.pyqtSignal(str)         # log file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_logger("PipelineController", level=logging.INFO)

        # Process tracking
        self._processing_process = None
        self._stop_event = multiprocessing.Event()
        self._stop_process_event = multiprocessing.Event()
        self._visualization_streamer = None
        self._output_protocol_manager = None
        self._shutdown_sequencer = None

        # Queues
        self._data_queue = None
        self._save_queue = None
        self._visualization_data_queue = None
        self._prediction_queue = None
        self._visualization_queue = None
        self._mode_output_queues = {}
        self._pid_queue = None

        # PID tracking
        self._mode_pids = {}
        self._system_processes = {}
        self._pid_collection_timer = None

        # Shared state
        self._shared_state = None

        # IO thread
        self._io_thread = None
        self._io_worker = None
        self._session_config = None
        self._status_callback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, config: dict, status_callback=None) -> None:
        """Start the processing pipeline (non-blocking).

        Runs a 3-phase async startup:
        - Phase 1 (GUI thread): Create queues
        - Phase 2 (Background thread): DB queries, file I/O
        - Phase 3 (GUI thread): Spawn processes, start streamers

        Args:
            config: Full configuration dict from retrieve_parameters().
            status_callback: Optional callback(protocol, connected) for protocol status.
        """
        self._status_callback = status_callback

        # Phase 1: Create queues (GUI thread)
        self._initialize_queues(config['mode_instances'])

        # Extract primitive data for background thread (no widget references)
        subject_id = config.get('subject_id', '')
        session_id = config.get('session_id', '')
        recording_name = config.get('recording_name', DEFAULT_RECORDING_NAME)
        study_name = config.get('study_name', DEFAULT_STUDY_NAME)

        # Store config for phase 3
        self._session_config = config

        # Phase 2: Background thread - I/O operations
        self._io_thread = QtCore.QThread()
        self._io_worker = SessionIOWorker(
            subject_id, session_id, recording_name, study_name, config.copy())
        self._io_worker.moveToThread(self._io_thread)
        self._io_worker.finished.connect(self._on_session_io_complete)
        self._io_worker.error.connect(self._on_session_io_error)
        self._io_thread.started.connect(self._io_worker.run)
        self._io_thread.start()

    def stop(self, blocking: bool = False) -> None:
        """Stop the processing pipeline.

        Args:
            blocking: If True, block (while pumping Qt events) until all processes stop.
                     Used during application shutdown.
        """
        self.logger.info("Stopping BMI processing pipeline...")

        self._stop_event.set()
        self._stop_process_event.set()
        if self._output_protocol_manager:
            self._output_protocol_manager.signal_stop()

        # Collect stop targets
        stop_targets = []
        if self._output_protocol_manager:
            stop_targets.extend(self._output_protocol_manager.get_stop_targets())
        if self._visualization_streamer and self._visualization_streamer.is_alive():
            stop_targets.append(('Visualization', self._visualization_streamer, TIMEOUT_VISUALIZATION))
        if self._processing_process and self._processing_process.is_alive():
            stop_targets.append(('Processor', self._processing_process, TIMEOUT_MAIN_PROCESS))

        self._shutdown_sequencer = ShutdownSequencer(self)
        self._shutdown_sequencer.finished.connect(self._finalize_stop)
        self._shutdown_sequencer.stop(stop_targets, blocking=blocking)

    @property
    def shared_state(self):
        """Expose shared state for telemetry and other consumers."""
        return self._shared_state

    def is_recording(self) -> bool:
        """Check if a recording is currently active."""
        return self._processing_process is not None and self._processing_process.is_alive()

    def cleanup(self) -> None:
        """Final cleanup, called from MainWindow.closeEvent()."""
        if self._shared_state:
            self._shared_state.cleanup()
            self._shared_state = None

    # ------------------------------------------------------------------
    # Phase 2 → Phase 3 transition
    # ------------------------------------------------------------------

    def _on_session_io_complete(self, result: dict):
        """Handle completion of session I/O (Phase 3 — GUI thread)."""
        self._cleanup_io_thread()

        config = self._session_config

        if result.get('log_file'):
            self.log_file_ready.emit(result['log_file'])

        # SharedState was created in background thread to avoid blocking GUI
        self._shared_state = result['shared_state']

        pipeline_config = PipelineConfig(
            sample_rate=config['sample_rate'],
            mode_instances=config['mode_instances'],
            file_identifier=result['file_identifier'],
            stop_event=self._stop_process_event,
            data_queue=self._data_queue,
            save_queue=self._save_queue,
            plot_queue=self._visualization_data_queue,
            prediction_queue=self._prediction_queue,
            mode_output_queues=self._mode_output_queues,
            preprocess_data=config.get('preprocess_data', False),
            modality_preprocessing=config.get('modality_preprocessing', {}),
            quality_control=config.get('quality_control', {}),
            stream_configs=config.get('stream_configs', []),
            study_name=config.get('study_name', DEFAULT_STUDY_NAME),
            recording_name=config.get('recording_name', DEFAULT_RECORDING_NAME),
            subject_id=config.get('subject_id', ''),
            session_id=config.get('session_id', ''),
            run_number=result['run_number'],
            experiment_description=config.get('experiment_description', ''),
            pid_queue=self._pid_queue,
            shared_state=self._shared_state,
            modality_data=config.get('modality_data', {}),
            output=config.get('output', {}),
            sync_to_async_sharing=config.get('sync_to_async_sharing', {}),
        )

        try:
            self._start_pipeline_process(pipeline_config)
            self._start_streamers(config)
            self._start_pid_collection()
            self.recording_started.emit()
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.start_failed.emit(str(e))

        self._session_config = None

    def _on_session_io_error(self, error_msg: str):
        """Handle session I/O error."""
        self._cleanup_io_thread()
        self.logger.error(f"Session I/O failed: {error_msg}")
        self.start_failed.emit(f"Failed to initialize session:\n{error_msg}")
        self._session_config = None

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _initialize_queues(self, mode_instances: dict):
        """Initialize all processing queues for the session."""
        self._stop_event.clear()
        self._stop_process_event.clear()

        self._data_queue = multiprocessing.Queue()
        self._save_queue = multiprocessing.Queue()
        self._visualization_data_queue = multiprocessing.Queue()
        self._prediction_queue = multiprocessing.Queue()
        self._pid_queue = multiprocessing.Queue()

        # Single shared queue for visualization (all modes push to same queue)
        self._visualization_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE)

        self._mode_output_queues = {
            name: FanOutQueue(
                primary_queue=multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE),
                secondary_queue=self._visualization_queue,
            )
            for name in mode_instances
        }

    def _clear_queues(self):
        """Close all queues on session end."""
        plain_queues = [
            self._data_queue, self._save_queue, self._visualization_data_queue,
            self._prediction_queue, self._pid_queue, self._visualization_queue,
        ]
        if self._mode_output_queues:
            plain_queues.extend(
                fq.primary_queue for fq in self._mode_output_queues.values()
                if fq.primary_queue
            )
            self._mode_output_queues = {}

        for q in plain_queues:
            if q:
                q.close()
                q.cancel_join_thread()

    # ------------------------------------------------------------------
    # Process spawning
    # ------------------------------------------------------------------

    def _start_pipeline_process(self, config: PipelineConfig):
        """Start the main processing pipeline process."""
        self._processing_process = multiprocessing.Process(
            target=run_pipeline,
            args=(config,)
        )
        self._processing_process.start()
        self._system_processes['Processor'] = self._processing_process

    def _start_visualization_streamer(self, config: dict):
        """Initialize the visualization streamer."""
        self.logger.debug("Initializing visualization streamer...")

        try:
            modality_data = config.get('modality_data', {})
            channel_labels = {m: d['channel_labels'] for m, d in modality_data.items()}

            self._visualization_streamer = VisualizationStreamer(
                plot_queue=self._visualization_data_queue,
                stream_info=VISUALIZATION_STREAM_INFO,
                stop_event=self._stop_event,
                output_queue=self._visualization_queue,
                history_length=1000,
                shared_state=self._shared_state,
                channel_labels=channel_labels,
            )
            self._visualization_streamer.daemon = True
            self._visualization_streamer.start()
            self._system_processes['Viz'] = self._visualization_streamer
            self.logger.info("Visualization streamer started successfully")
            QtCore.QTimer.singleShot(500, self._check_visualization_streamer)

        except Exception as e:
            self.logger.error(f"Failed to create/start visualization streamer: {e}")
            self.logger.error("Continuing without visualization output")
            self._visualization_streamer = None

    def _check_visualization_streamer(self):
        """Check if visualization streamer started successfully (called via QTimer)."""
        if self._visualization_streamer and not self._visualization_streamer.is_alive():
            self.logger.error("Visualization streamer process failed to start")
            self._visualization_streamer = None

    def _start_streamers(self, config: dict):
        """Initialize visualization and output protocol streamers."""
        self._start_visualization_streamer(config)

        self._output_protocol_manager = OutputProtocolManager(
            stop_event=self._stop_event,
            prediction_queue=self._prediction_queue,
            shared_state=self._shared_state,
            status_callback=self._status_callback,
        )
        streamer_processes = self._output_protocol_manager.initialize(config)
        self._system_processes.update(streamer_processes)

    # ------------------------------------------------------------------
    # PID collection
    # ------------------------------------------------------------------

    def _get_system_pids(self) -> dict:
        """Get PIDs of all alive system processes."""
        return {
            name: proc.pid
            for name, proc in self._system_processes.items()
            if proc and proc.is_alive()
        }

    def _start_pid_collection(self):
        """Start collecting mode process PIDs from the PID queue."""
        self._mode_pids = {}

        def collect_pids():
            try:
                while not self._pid_queue.empty():
                    pid_info = self._pid_queue.get_nowait()
                    mode_name = pid_info['mode_name']
                    pid = pid_info['pid']

                    if mode_name not in self._mode_pids:
                        self.logger.info(f"Collected PID {pid} for mode {mode_name}")

                    self._mode_pids[mode_name] = pid

                self.pids_updated.emit(self._mode_pids.copy(), self._get_system_pids())
            except queue.Empty:
                pass
            except KeyError as e:
                self.logger.debug(f"PID collection key error: {e}")

        self._pid_collection_timer = QtCore.QTimer()
        self._pid_collection_timer.timeout.connect(collect_pids)
        self._pid_collection_timer.start(PID_COLLECTION_INTERVAL_MS)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _finalize_stop(self):
        """Complete shutdown after all processes stopped."""
        self._processing_process = None
        self._visualization_streamer = None

        if self._output_protocol_manager:
            self._output_protocol_manager.cleanup()

        self._clear_queues()

        if self._pid_collection_timer:
            self._pid_collection_timer.stop()
            self._pid_collection_timer = None

        self._mode_pids = {}
        self._system_processes = {}

        if self._shared_state:
            self._shared_state.cleanup()
            self._shared_state = None

        self.recording_stopped.emit()

    # ------------------------------------------------------------------
    # IO thread helpers
    # ------------------------------------------------------------------

    def _cleanup_io_thread(self):
        """Clean up session I/O thread and worker."""
        if self._io_thread:
            self._io_thread.quit()
            self._io_thread.wait()
        self._io_thread = None
        self._io_worker = None
