"""
Provides the main graphical interface for Dendrite using modular components from dendrite/gui/. 
"""

import signal
import sys
import logging
import multiprocessing
import threading
import time
from typing import List, Dict, Optional, Any
import queue
import copy

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.config.stream_config_manager import initialize_stream_config_manager, get_stream_config_manager
from dendrite.gui.config.mode_config_manager import initialize_mode_config_manager, get_mode_config_manager
from dendrite.data.streaming import (
    VisualizationStreamer, LSLStreamer, SocketStreamer, ZMQStreamer, ROS2Streamer
)
from dendrite.constants import (
    VERSION, APP_NAME, PREDICTION_STREAM_INFO, VISUALIZATION_STREAM_INFO,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_POSITION, APPLICATION_TITLE,
    DEFAULT_RECORDING_NAME, DEFAULT_STUDY_NAME
)
from dendrite.processing.queue_utils import FanOutQueue
from dendrite.gui.widgets import (
    ControlButtonsWidget, TopNavigationBar,
    GeneralParametersWidget, StreamConfigurationWidget,
    ModeInstanceBadge, ModeInstanceConfigDialog,
    PreprocessingWidget, OutputWidget,
    LogDisplayWidget, TelemetryWidget,
)
from dendrite.gui.styles.widget_styles import WidgetStyles, apply_app_styles
from dendrite.gui.styles.design_tokens import BG_MAIN, BG_PANEL, SEPARATOR_SUBTLE
from dendrite.gui.workers import SessionIOWorker
from dendrite.ml.decoders import get_available_decoders
from dendrite.gui.config import SystemConfigurationManager
from dendrite.utils.logger_central import setup_logger, get_logger
from dendrite.data.storage.database import Database
from dendrite.processing.pipeline import run_pipeline

QUEUE_SIZE_LARGE = 1000
TIMEOUT_MAIN_PROCESS = 10
TIMEOUT_STREAMER_DEFAULT = 2
TIMEOUT_VISUALIZATION = 3
PID_COLLECTION_INTERVAL_MS = 500
PREDICTION_FANOUT_TIMEOUT = 0.1
SUPPORTED_PROTOCOLS = ['lsl', 'socket', 'zmq', 'ros2']

class MainWindow(QtWidgets.QMainWindow):
    """
    Main BMI application window managing GUI and process coordination.

    Manages:
    - UI layout and navigation (control/display panels)
    - Multi-process pipeline coordination (acquisition, processing, modes)
    - Data streaming (visualization, LSL, ROS2, socket, ZMQ)
    - Configuration persistence and session management
    """

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setup_variables()
        self.config_manager = SystemConfigurationManager(self)
        self.stream_config_manager = initialize_stream_config_manager(self)
        self.mode_config_manager = initialize_mode_config_manager(self)

        # Connect mode config manager signals for reactive badge management
        self.mode_config_manager.instance_added.connect(self._on_instance_added)
        self.mode_config_manager.instance_removed.connect(self._on_instance_removed)
        self.mode_config_manager.instance_renamed.connect(self._on_instance_renamed)
        self.mode_config_manager.instances_cleared.connect(self._on_instances_cleared)

        self.logger = setup_logger("GUI", level=logging.INFO)
        self.logger.info(f"{APPLICATION_TITLE} v{VERSION} starting up")
        self.available_decoders = get_available_decoders()
        self.init_ui()

    def setup_variables(self):
        """Initialize all instance variables."""
        self.processing_process = None
        self.stop_event = multiprocessing.Event()
        self.stop_process_event = multiprocessing.Event()

        self.data_queue = None
        self.save_queue = None
        self.visualization_data_queue = None
        self.prediction_queue = None
        self.visualization_queue = None
        self.mode_output_queues = {}

        # Shared state for cross-process communication (created fresh per recording session)
        self.shared_state = None

        self.visualization_streamer = None
        self.lsl_stop_event = None
        self.output_streamers = {}
        self.protocol_queues = {}
        self.prediction_fanout_thread = None

        # Process tracking for resource monitoring
        self.mode_pids = {}
        self.system_processes = {}
        self.pid_queue = None
        self.pid_collection_timer = None

        # Session I/O thread (for async startup)
        self._io_thread = None
        self._io_worker = None
        self._session_config = None

        # UI components (will be initialized in init_ui)
        self.output_widget = None
        self.log_display_widget = None
        self.telemetry_widget = None
        self.display_stack = None

        # View state (must be set before signals connect in __init__)
        self._compact_mode = False

    def is_recording(self) -> bool:
        """Check if a recording is currently active."""
        return self.processing_process is not None and self.processing_process.is_alive()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        self.setGeometry(*DEFAULT_WINDOW_POSITION, *DEFAULT_WINDOW_SIZE)

        central_widget = QtWidgets.QWidget()
        central_widget.setStyleSheet(f"background-color: {BG_MAIN};")
        self.setCentralWidget(central_widget)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        central_widget.setLayout(self.main_layout)

        # Top navigation bar (full-width)
        self.top_nav_bar = TopNavigationBar(
            tabs=[
                ("general", "General"),
                ("modes", "Modes"),
                ("preprocessing", "Preprocessing"),
                ("output", "Output"),
            ],
            parent=self,
        )
        self.top_nav_bar.section_changed.connect(self.on_navigation_changed)
        self.top_nav_bar.display_changed.connect(self.on_display_mode_changed)
        self.main_layout.addWidget(self.top_nav_bar)

        # Content area with control and display panels
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.main_layout.addLayout(content_layout, 1)

        self.setup_control_panel(content_layout)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setStyleSheet(f"border: none; border-left: 1px solid {SEPARATOR_SUBTLE}; max-width: 1px;")
        content_layout.addWidget(separator)

        self.setup_display_panel(content_layout)
        self.set_all_badge_statuses('ready') 

    def setup_control_panel(self, parent_layout: QtWidgets.QHBoxLayout):
        """Set up the control panel with all sections."""
        # Wrap in container to limit max width (keeps current size, doesn't grow with window)
        controls_container = QtWidgets.QWidget()
        controls_container.setStyleSheet(f"background-color: {BG_PANEL};")
        controls_container.setMaximumWidth(560)
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        parent_layout.addWidget(controls_container, 1)

        controls_layout.setContentsMargins(
            WidgetStyles.layout['margin'], WidgetStyles.layout['margin'],
            WidgetStyles.layout['spacing'], WidgetStyles.layout['margin']
        )
        controls_layout.setSpacing(WidgetStyles.layout['spacing'])

        self.control_stack = QtWidgets.QStackedWidget()
        controls_layout.addWidget(self.control_stack, 1)
        self._create_control_sections()

        controls_layout.addSpacing(WidgetStyles.layout['spacing'])
        self.control_buttons_widget = ControlButtonsWidget(self)
        self.control_buttons_widget.start_requested.connect(self.start_processing)
        self.control_buttons_widget.stop_requested.connect(self.confirm_stop_processing)
        controls_layout.addWidget(self.control_buttons_widget)

    def _apply_section_layout(self, layout):
        """Apply standard margins and spacing to a control section layout."""
        layout.setContentsMargins(
            WidgetStyles.layout['margin'], WidgetStyles.layout['spacing_xl'],
            WidgetStyles.layout['margin'], WidgetStyles.layout['margin']
        )
        layout.setSpacing(WidgetStyles.layout['spacing'])

    def _create_section(self, setup_content) -> QtWidgets.QWidget:
        """Create a control panel section with standard layout."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self._apply_section_layout(layout)
        setup_content(layout)
        return widget

    def _create_control_sections(self):
        """Create all control panel sections."""
        self.control_stack.addWidget(self._create_section(self._setup_general_section))
        self.control_stack.addWidget(self._create_section(self._setup_mode_section))
        self.control_stack.addWidget(self._create_section(self._setup_preprocessing_section))
        self.control_stack.addWidget(self._create_section(self._setup_output_section))

    def _setup_general_section(self, layout: QtWidgets.QVBoxLayout):
        """Set up the general parameters section content."""
        self.general_params_widget = GeneralParametersWidget(self)
        self.general_params_widget.load_config_requested.connect(self.load_configuration)
        layout.addWidget(self.general_params_widget)

        self.stream_config_widget = StreamConfigurationWidget(self)
        self.stream_config_widget.streams_configured.connect(self.on_streams_configured)
        layout.addWidget(self.stream_config_widget)
        layout.addStretch()

    def _setup_mode_section(self, layout: QtWidgets.QVBoxLayout):
        """Set up the mode configuration section content."""
        self.add_instance_button = QtWidgets.QPushButton("Add Mode")
        self.add_instance_button.setStyleSheet(WidgetStyles.button(variant='primary', padding="8px 12px"))
        self.add_instance_button.clicked.connect(lambda: self.add_mode_instance_badge())
        layout.addWidget(self.add_instance_button)

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addStretch()
        self.view_toggle_button = QtWidgets.QPushButton("☰")
        self.view_toggle_button.setToolTip("Toggle compact view")
        self.view_toggle_button.setFixedSize(20, 20)
        self.view_toggle_button.setStyleSheet(WidgetStyles.button(variant='icon', fixed_size=20, transparent=True, blend=True))
        self.view_toggle_button.clicked.connect(self._toggle_compact_view)
        toggle_row.addWidget(self.view_toggle_button)
        layout.addLayout(toggle_row)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(WidgetStyles.scrollarea)
        layout.addWidget(scroll_area, 1)

        self.badges_container = QtWidgets.QWidget()
        self.badges_container.setStyleSheet(WidgetStyles.transparent_container())
        self.badges_layout = QtWidgets.QVBoxLayout(self.badges_container)
        self.badges_layout.setContentsMargins(0, WidgetStyles.layout['margin'], 0, WidgetStyles.layout['margin'])
        self.badges_layout.setSpacing(WidgetStyles.layout['spacing_lg'])
        self.badges_layout.addStretch()
        scroll_area.setWidget(self.badges_container)

        self.mode_instance_badges = []

    def _setup_preprocessing_section(self, layout: QtWidgets.QVBoxLayout):
        """Set up the preprocessing section content."""
        self.preprocessing_widget = PreprocessingWidget(self)
        layout.addWidget(self.preprocessing_widget, 1)

    def _setup_output_section(self, layout: QtWidgets.QVBoxLayout):
        """Set up the output section content."""
        self.output_widget = OutputWidget(self)
        layout.addWidget(self.output_widget)
        layout.addStretch() 

    def setup_display_panel(self, parent_layout: QtWidgets.QHBoxLayout):
        """Set up the display panel with log and telemetry views."""
        self.display_layout = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(self.display_layout, 2)
        self.display_layout.setContentsMargins(
            12, WidgetStyles.layout['margin'],
            WidgetStyles.layout['margin'], WidgetStyles.layout['margin']
        )
        self.display_layout.setSpacing(WidgetStyles.layout['spacing'])

        self.display_stack = QtWidgets.QStackedWidget()
        self.log_display_widget = LogDisplayWidget(self)
        self.telemetry_widget = TelemetryWidget(self)
        self.display_stack.addWidget(self.log_display_widget)
        self.display_stack.addWidget(self.telemetry_widget)
        self.display_stack.setCurrentIndex(1)
        self.display_layout.addWidget(self.display_stack)

    def on_navigation_changed(self, section: str):
        """Handle navigation section change."""
        section_map = {
            "general": 0,
            "modes": 1,
            "preprocessing": 2,
            "output": 3
        }
        index = section_map.get(section, 0)
        self.control_stack.setCurrentIndex(index)

    def on_display_mode_changed(self, mode: str):
        """Handle display mode change between log and telemetry."""
        if mode == "log":
            self.display_stack.setCurrentIndex(0)
        elif mode == "telemetry":
            self.display_stack.setCurrentIndex(1)

    def on_streams_configured(self, stream_data: dict):
        """Handle when streams are configured by the StreamConfigurationWidget."""
        self._refresh_open_mode_dialogs()
        modality_data = self.stream_config_manager.get_modality_data()
        eeg_count = modality_data.get('eeg', {}).get('total_count', 0)
        sample_rate = self.stream_config_manager.get_system_sample_rate()
        stream_count = len(self.stream_config_manager.get_streams())

        self.logger.debug(f"Stream configuration updated: {stream_count} streams, "
                   f"{eeg_count} EEG channels, {sample_rate} Hz")

    def add_mode_instance_badge(self, mode_instance_config: Optional[dict] = None) -> Optional[ModeInstanceBadge]:
        """
        Add a new mode instance.

        Adds config to manager, which triggers instance_added signal to create the badge.

        Args:
            mode_instance_config: Mode configuration dict. If None, creates default Synchronous mode.

        Returns:
            Created badge widget, or None if creation failed.
        """
        if mode_instance_config is None:
            mode_instance_config = {
                'name': self.mode_config_manager.generate_unique_name('Synchronous', sanitize=True),
                'mode': 'synchronous'
            }

        instance_name = mode_instance_config.get('name')
        if not instance_name:
            return None

        self.mode_config_manager.add_instance(instance_name, mode_instance_config)

        for badge in self.mode_instance_badges:
            if badge.instance_name == instance_name:
                return badge
        return None

    def remove_mode_instance_badge(self, badge):
        """Remove a mode instance by removing from manager (triggers signal for badge cleanup)."""
        if badge in self.mode_instance_badges:
            instance_name = badge.instance_name
            if instance_name:
                self.mode_config_manager.remove_instance(instance_name)
    
    def _on_instance_added(self, instance_name: str, config: dict):
        """Create badge when instance added to manager."""
        if any(b.instance_name == instance_name for b in self.mode_instance_badges):
            return

        badge = ModeInstanceBadge(instance_name, self, compact=self._compact_mode)
        badge.configure_button.clicked.connect(lambda: self.open_mode_instance_dialog(badge))
        badge.remove_button.clicked.connect(lambda: self.remove_mode_instance_badge(badge))

        # Insert before stretch
        self.badges_layout.insertWidget(self.badges_layout.count() - 1, badge)
        self.mode_instance_badges.append(badge)

        self.highlight_linked_badges()
        self._refresh_open_mode_dialogs()

    def _on_instance_removed(self, instance_name: str):
        """Remove badge when instance removed from manager."""
        for badge in self.mode_instance_badges[:]:
            if badge.instance_name == instance_name:
                self.mode_instance_badges.remove(badge)
                badge.deleteLater()
                break

        self.highlight_linked_badges()
        self._refresh_open_mode_dialogs()

    def _on_instance_renamed(self, old_name: str, new_name: str):
        """Update badge when instance renamed in manager."""
        for badge in self.mode_instance_badges:
            if badge.instance_name == old_name:
                badge.instance_name = new_name
                badge.update_summary()
                break
        self.highlight_linked_badges()

    def _on_instances_cleared(self):
        """Remove all badges when manager is cleared."""
        for badge in self.mode_instance_badges[:]:
            badge.deleteLater()
        self.mode_instance_badges.clear()
        self.highlight_linked_badges()

    def open_mode_instance_dialog(self, badge):
        """Open configuration dialog for a mode instance."""
        dialog = ModeInstanceConfigDialog(
            badge.instance_name,
            self.available_decoders,
            self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Dialog updates manager on accept, badge updates via signals
            badge.set_enabled(True)
            self.highlight_linked_badges()

    def set_all_badge_statuses(self, status: str):
        """Set the same status on all enabled mode instance badges."""
        for badge in self.mode_instance_badges:
            if badge.is_enabled():
                badge.set_status(status)

    def _toggle_compact_view(self):
        """Toggle between card view and compact list view for mode badges."""
        self._compact_mode = not self._compact_mode
        for badge in self.mode_instance_badges:
            badge.set_compact(self._compact_mode)

        self.badges_layout.setSpacing(
            WidgetStyles.layout['spacing'] if self._compact_mode
            else WidgetStyles.layout['spacing_lg']
        )

        self.view_toggle_button.setText("▤" if self._compact_mode else "☰")
        self.view_toggle_button.setToolTip(
            "Switch to card view" if self._compact_mode else "Switch to compact view"
        )

    def highlight_linked_badges(self):
        """
        Highlight async mode badges linked to sync modes.

        Async modes can reference sync mode trial structures. This updates badge
        visual state to show dependency relationships.
        """
        sync_instances = {}
        for badge in self.mode_instance_badges:
            mode = badge.instance_data.get('mode', '').lower()
            if mode == 'synchronous':
                sync_instances[badge.instance_name] = badge

        for badge in self.mode_instance_badges:
            mode = badge.instance_data.get('mode', '').lower()
            if mode == 'asynchronous':
                linked_sync = badge.get_linked_sync_instance()
                if linked_sync and linked_sync in sync_instances:
                    badge.set_status("linking" if badge.status == "ready" else badge.status)
                else:
                    badge.update_summary()

    def _refresh_open_mode_dialogs(self):
        """Refresh any open mode configuration dialogs when stream data changes."""
        for widget in QtWidgets.QApplication.allWidgets():
            if isinstance(widget, ModeInstanceConfigDialog) and widget.isVisible():
                if hasattr(widget, 'refresh_sync_mode_options'):
                    widget.refresh_sync_mode_options()

    def load_configuration(self):
        """Load configuration from a JSON file."""
        self.config_manager.load_configuration()

    def retrieve_parameters(self) -> dict:
        """Retrieve all configuration parameters from the GUI."""
        return self.config_manager.build_configuration()

    def start_processing(self):
        """
        Start the BMI processing pipeline (non-blocking).

        Uses 3-phase async startup:
        - Phase 1 (GUI thread): Validate, collect config, create queues
        - Phase 2 (Background thread): DB queries, file I/O
        - Phase 3 (GUI thread): Attach queues, start processes
        """
        if not self._validate_start_configuration():
            return

        self._set_starting_state()

        # Phase 1: GUI thread - collect config, create queues
        config = self.retrieve_parameters()
        self._initialize_queues(config['mode_instances'])

        # Extract primitive data for background thread (no widget references)
        subject_id = config.get('subject_id', '')
        session_id = config.get('session_id', '')
        recording_name = config.get('recording_name', DEFAULT_RECORDING_NAME)
        study_name = config.get('study_name', DEFAULT_STUDY_NAME)

        # Store config for phase 3 (will be updated with I/O results)
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

    def _set_starting_state(self):
        """Set GUI to starting state."""
        self.control_buttons_widget.set_running_state(True)
        self.stream_config_widget.set_recording_state(True)

    def _cleanup_io_thread(self):
        """Clean up session I/O thread and worker."""
        if self._io_thread:
            self._io_thread.quit()
            self._io_thread.wait()
        self._io_thread = None
        self._io_worker = None

    def _on_session_io_complete(self, result: dict):
        """Handle completion of session I/O (Phase 3 - GUI thread)."""
        self._cleanup_io_thread()

        config = self._session_config
        config['run_number'] = result['run_number']
        config['file_identifier'] = result['file_identifier']

        if self.log_display_widget and result.get('log_file'):
            self.log_display_widget.set_log_file(result['log_file'])

        # SharedState was created in background thread to avoid blocking GUI
        self.shared_state = result['shared_state']

        config['stop_event'] = self.stop_process_event
        config['data_queue'] = self.data_queue
        config['save_queue'] = self.save_queue
        config['plot_queue'] = self.visualization_data_queue
        config['prediction_queue'] = self.prediction_queue
        config['mode_output_queues'] = self.mode_output_queues
        config['pid_queue'] = self.pid_queue
        config['shared_state'] = self.shared_state

        try:
            self._start_processing_pipeline(config)
            self._initialize_streamers(config)
            self._finalize_recording_state()
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self._revert_start_state()

        self._session_config = None

    def _on_session_io_error(self, error_msg: str):
        """Handle session I/O error."""
        self._cleanup_io_thread()

        self.logger.error(f"Session I/O failed: {error_msg}")
        QtWidgets.QMessageBox.critical(
            self,
            "Session Start Failed",
            f"Failed to initialize session:\n{error_msg}",
            QtWidgets.QMessageBox.StandardButton.Ok
        )
        self._revert_start_state()
        self._session_config = None

    def _revert_start_state(self):
        """Revert GUI state if startup fails."""
        self.control_buttons_widget.set_running_state(False)
        self.stream_config_widget.set_recording_state(False)

    def _validate_start_configuration(self) -> bool:
        """Validate EEG, BIDS fields, and preprocessing configuration before starting."""
        is_valid, error_msg = self.general_params_widget.validate_required_fields()
        if not is_valid:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Required Field",
                f"{error_msg}",
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False

        modality_data = self.stream_config_manager.get_modality_data()
        eeg_channel_count = modality_data.get('eeg', {}).get('total_count', 0)
        if eeg_channel_count == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No EEG Stream",
                "Connect at least one EEG stream before starting.",
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False

        self.logger.info(f"EEG validation passed: {eeg_channel_count} EEG channels configured")

        is_valid, error_msg = self.preprocessing_widget.validate_inputs()
        if not is_valid:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Preprocessing",
                f"{error_msg}",
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False

        self.logger.info("Preprocessing validation passed")
        return True

    def _initialize_queues(self, mode_instances: dict):
        """Initialize all processing queues for the session."""
        self.stop_event.clear()
        self.stop_process_event.clear()

        self.data_queue = multiprocessing.Queue()
        self.save_queue = multiprocessing.Queue()
        self.visualization_data_queue = multiprocessing.Queue()
        self.prediction_queue = multiprocessing.Queue()
        self.pid_queue = multiprocessing.Queue()

        # Single shared queue for visualization (all modes push to same queue)
        self.visualization_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE)

        self.mode_output_queues = {}
        for instance_name in mode_instances.keys():
            self.mode_output_queues[instance_name] = FanOutQueue(
                primary_queue=multiprocessing.Queue(maxsize=QUEUE_SIZE_LARGE),
                secondary_queue=self.visualization_queue  # All modes share this queue
            )

    def _start_processing_pipeline(self, config: dict):
        """Start the main processing pipeline process."""
        self.processing_process = multiprocessing.Process(
            target=run_pipeline,
            args=(config,)
        )
        self.processing_process.start()
        self.system_processes['Processor'] = self.processing_process

    def _initialize_streamers(self, config: dict):
        """Initialize visualization and output protocol streamers."""
        self.initialize_visualization_streamer(config)

        self.lsl_stop_event = multiprocessing.Event()
        self.initialize_output_protocols(config)

    def _finalize_recording_state(self):
        """Finalize UI after processes started."""
        if self.telemetry_widget:
            self.telemetry_widget.on_recording_started()

        self._start_pid_collection()
        self.set_all_badge_statuses('running')

    def _get_system_pids(self) -> dict:
        """Get PIDs of all alive system processes."""
        return {
            name: proc.pid
            for name, proc in self.system_processes.items()
            if proc and proc.is_alive()
        }

    def _start_pid_collection(self):
        """Start collecting mode process PIDs from the PID queue."""
        self.mode_pids = {}  # Reset PID dict

        def collect_pids():
            """Poll PID queue for mode process PIDs."""
            try:
                while not self.pid_queue.empty():
                    pid_info = self.pid_queue.get_nowait()
                    mode_name = pid_info['mode_name']
                    pid = pid_info['pid']

                    # Log only first time for each mode
                    if mode_name not in self.mode_pids:
                        self.logger.info(f"Collected PID {pid} for mode {mode_name}")

                    self.mode_pids[mode_name] = pid

                if self.telemetry_widget:
                    self.telemetry_widget.set_mode_pids(self.mode_pids)
                    self.telemetry_widget.set_system_pids(self._get_system_pids())
            except queue.Empty:
                pass  # Expected - queue polling
            except KeyError as e:
                self.logger.debug(f"PID collection key error: {e}")

        self.pid_collection_timer = QtCore.QTimer()
        self.pid_collection_timer.timeout.connect(collect_pids)
        self.pid_collection_timer.start(PID_COLLECTION_INTERVAL_MS)  # Poll regularly

    def confirm_stop_processing(self):
        """Show confirmation dialog before stopping processing."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Stop Recording",
            "This will end the current recording session.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.stop_processing()

    def stop_processing(self, blocking: bool = False):
        """Stop the BMI processing pipeline (non-blocking by default).

        Args:
            blocking: If True, wait for all processes to stop before returning.
                     Used during application shutdown.
        """
        self.logger.info("Stopping BMI processing pipeline...")

        self.stop_event.set()
        self.stop_process_event.set()
        if self.lsl_stop_event:
            self.lsl_stop_event.set()

        self._stop_targets = []
        if self.prediction_fanout_thread and self.prediction_fanout_thread.is_alive():
            self._stop_targets.append(('Fan-out', self.prediction_fanout_thread, 2))
        for protocol, streamer in (self.output_streamers or {}).items():
            if streamer and streamer.is_alive():
                self._stop_targets.append((protocol, streamer, TIMEOUT_STREAMER_DEFAULT))
        if self.visualization_streamer and self.visualization_streamer.is_alive():
            self._stop_targets.append(('Visualization', self.visualization_streamer, TIMEOUT_VISUALIZATION))
        if self.processing_process and self.processing_process.is_alive():
            self._stop_targets.append(('Processor', self.processing_process, TIMEOUT_MAIN_PROCESS))

        self._stop_index = 0
        self._stop_start_time = None

        if blocking:
            self._stop_blocking()
        else:
            self._stop_poll_timer = QtCore.QTimer()
            self._stop_poll_timer.timeout.connect(self._poll_stop_progress)
            self._poll_stop_progress()

    def _stop_blocking(self):
        """Stop all targets, blocking but processing Qt events."""
        for name, target, timeout in self._stop_targets:
            self.logger.info(f"Stopping {name}...")
            start = time.time()
            while target.is_alive() and (time.time() - start) < timeout:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.05)
            if target.is_alive():
                self.logger.warning(f"Force terminating {name}")
                if hasattr(target, 'terminate'):
                    target.terminate()
        self._finalize_stop()

    def _poll_stop_progress(self):
        """Poll current target, advance when done or timed out."""
        if self._stop_index >= len(self._stop_targets):
            self._stop_poll_timer.stop()
            self._finalize_stop()
            return

        name, target, timeout = self._stop_targets[self._stop_index]

        if self._stop_start_time is None:
            self._stop_start_time = time.time()
            self.logger.info(f"Stopping {name}...")
            self._stop_poll_timer.start(50)
            return

        if not target.is_alive():
            self._stop_start_time = None
            self._stop_index += 1
            self._poll_stop_progress()
            return

        if time.time() - self._stop_start_time > timeout:
            self.logger.warning(f"Force terminating {name}")
            if hasattr(target, 'terminate'):
                target.terminate()
            self._stop_start_time = None
            self._stop_index += 1
            self._poll_stop_progress()

    def _finalize_stop(self):
        """Complete shutdown after all processes stopped."""
        self.processing_process = None
        self.visualization_streamer = None
        self.output_streamers = {}

        self.clear_queues()
        self._clear_display_layout()
        self.control_buttons_widget.set_running_state(False)
        self.stream_config_widget.set_recording_state(False)

        if self.pid_collection_timer:
            self.pid_collection_timer.stop()
            self.pid_collection_timer = None

        self.mode_pids = {}
        self.system_processes = {}

        if self.telemetry_widget:
            self.telemetry_widget.on_recording_stopped()

        if self.shared_state:
            self.shared_state.cleanup()
            self.shared_state = None

        self.set_all_badge_statuses('ready')
        self._update_protocol_status(False, all_protocols=True)

    def _clear_display_layout(self):
        """Clear plot layout except display stack."""
        widgets_to_keep = {self.display_stack}
        items_to_remove = []

        for i in range(self.display_layout.count()):
            item = self.display_layout.itemAt(i)
            if item and item.widget() and item.widget() not in widgets_to_keep:
                items_to_remove.append(item)

        for item in items_to_remove:
            widget = item.widget()
            if widget:
                self.display_layout.removeWidget(widget)
                widget.deleteLater()

    def clear_queues(self):
        """Close all queues on session end."""
        for q in [self.data_queue, self.save_queue, self.visualization_data_queue,
                  self.prediction_queue, self.pid_queue, self.visualization_queue]:
            if q:
                q.close()
                q.cancel_join_thread()

        if self.mode_output_queues:
            for fanout_queue in self.mode_output_queues.values():
                if fanout_queue.primary_queue:
                    fanout_queue.primary_queue.close()
                    fanout_queue.primary_queue.cancel_join_thread()
                # secondary_queue is shared (visualization_queue) - already closed above

    def initialize_visualization_streamer(self, config: dict):
        """Initialize the visualization streamer and dashboard metadata outlet."""
        self.logger.debug("Initializing visualization streamer...")

        try:
            # Use modality_data from config (already computed in build_configuration)
            modality_data = config.get('modality_data', {})
            channel_labels = {m: d['channel_labels'] for m, d in modality_data.items()}

            self.visualization_streamer = VisualizationStreamer(
                plot_queue=self.visualization_data_queue,
                stream_info=VISUALIZATION_STREAM_INFO,
                stop_event=self.stop_event,
                output_queue=self.visualization_queue,
                history_length=1000,
                shared_state=self.shared_state,
                channel_labels=channel_labels,
            )
            self.visualization_streamer.daemon = True
            self.visualization_streamer.start()
            self.system_processes['Viz'] = self.visualization_streamer
            self.logger.info("Visualization streamer started successfully")
            QtCore.QTimer.singleShot(500, self._check_visualization_streamer)

        except Exception as e:
            self.logger.error(f"Failed to create/start visualization streamer: {e}")
            self.logger.error("Continuing without visualization output")
            self.visualization_streamer = None

    def _check_visualization_streamer(self):
        """Check if visualization streamer started successfully (called via QTimer)."""
        if self.visualization_streamer and not self.visualization_streamer.is_alive():
            self.logger.error("Visualization streamer process failed to start")
            self.visualization_streamer = None

    def initialize_output_protocols(self, config: dict):
        """Initialize all output protocols based on configuration."""
        output_config = config.get('output', {}).get('protocols', {})

        self.logger.debug(f"Output configuration: {output_config}")

        enabled_protocols = self._setup_protocol_queues(output_config)

        if not enabled_protocols:
            self.logger.warning("No output protocols enabled - predictions will not be streamed")
            return

        self.logger.info(f"Initializing output protocols: {enabled_protocols}")
        self.start_prediction_fanout_thread()

        for protocol in enabled_protocols:
            self._init_streamer_with_config(protocol, output_config.get(protocol, {}), config)
    
    def _setup_protocol_queues(self, output_config: dict) -> List[str]:
        """Set up queues for enabled protocols and return list of enabled protocols."""
        self.protocol_queues = {}
        enabled_protocols = []

        for protocol in SUPPORTED_PROTOCOLS:
            protocol_config = output_config.get(protocol, {})
            is_enabled = protocol_config.get('enabled', False)

            self.logger.debug(f"Protocol {protocol}: enabled={is_enabled}, config={protocol_config}")

            if is_enabled:
                enabled_protocols.append(protocol)
                self.protocol_queues[protocol] = multiprocessing.Queue()

        if not output_config and not enabled_protocols:
            self.logger.info("No output configuration found, defaulting to LSL")
            enabled_protocols = ['lsl']
            self.protocol_queues['lsl'] = multiprocessing.Queue()
            output_config['lsl'] = {'enabled': True, 'config': {}}

        return enabled_protocols
    
    def _get_streamer_config(self, protocol: str, protocol_config: dict, full_config: dict):
        """Get streamer class and initialization kwargs for a protocol."""
        base_kwargs = {
            'input_queue': self.protocol_queues[protocol],
            'stop_event': self.lsl_stop_event,
            'shared_state': self.shared_state
        }

        protocol_map = {
            'lsl': (LSLStreamer, {
                **base_kwargs,
                'stream_info': copy.deepcopy(PREDICTION_STREAM_INFO),
                'lsl_config': protocol_config.get('config', {})
            }),
            'socket': (SocketStreamer, {
                **base_kwargs,
                'socket_config': protocol_config.get('config', {})
            }),
            'zmq': (ZMQStreamer, {
                **base_kwargs,
                'zmq_config': protocol_config.get('config', {})
            }),
            'ros2': (ROS2Streamer, {
                **base_kwargs,
                'ros2_config': protocol_config.get('config', {}),
                'stream_name': "BMI_Predictions",
                'classifier_names': list(full_config.get('mode_instances', {}).keys())
            })
        }

        if protocol not in protocol_map:
            raise ValueError(f"Unknown protocol: {protocol}")
        return protocol_map[protocol]
    
    def _init_streamer_with_config(self, protocol: str, protocol_config: dict, full_config: dict):
        """Initialize a single output streamer."""
        try:
            streamer_class, streamer_kwargs = self._get_streamer_config(
                protocol, protocol_config, full_config
            )
            streamer = streamer_class(**streamer_kwargs)
            streamer.daemon = True
            streamer.start()
            self.output_streamers[protocol] = streamer
            self.system_processes[protocol.upper()] = streamer
            self._update_protocol_status(True, protocol=protocol)
            self.logger.info(f"{protocol.upper()} streamer started")
        except Exception as e:
            self.logger.error(f"Failed to initialize {protocol} streamer: {e}")
            self._update_protocol_status(False, protocol=protocol)
    
    def _update_protocol_status(self, connected: bool, protocol: str = None, all_protocols: bool = False):
        """Update UI status indicators for output protocol connections."""
        if self.output_widget:
            if all_protocols:
                for p in SUPPORTED_PROTOCOLS:
                    self.output_widget.set_protocol_connected(p, connected)
            elif protocol:
                self.output_widget.set_protocol_connected(protocol, connected)
                    
    def start_prediction_fanout_thread(self):
        """Start thread to distribute predictions to all protocol queues."""
        def fanout_predictions():
            logger = get_logger("PredictionFanOut")
            logger.info(f"Fan-out thread started for: {list(self.protocol_queues.keys())}")
            
            while not (self.stop_event and self.stop_event.is_set()):
                try:
                    prediction_data = self.prediction_queue.get(timeout=PREDICTION_FANOUT_TIMEOUT)
                    
                    for protocol, protocol_queue in self.protocol_queues.items():
                        try:
                            protocol_queue.put(prediction_data, block=False)
                        except queue.Full:
                            logger.warning(f"{protocol} queue full, dropping prediction")
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Fan-out error: {e}")
                    break
                    
            logger.info("Fan-out thread stopped")
        
        self.prediction_fanout_thread = threading.Thread(target=fanout_predictions, daemon=True, name="PredictionFanOut")
        self.prediction_fanout_thread.start()

    def closeEvent(self, event) -> None:
        """Handle application close event."""
        if self.processing_process and self.processing_process.is_alive():
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Recording Active",
                "Close application and end current recording?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.logger.warning("User confirmed closing application during active recording")
            self.stop_processing(blocking=True)

        if self.log_display_widget:
            self.log_display_widget.stop_monitoring()

        if self.telemetry_widget:
            self.telemetry_widget.cleanup()

        if self.shared_state:
            self.shared_state.cleanup()
            self.shared_state = None

        self.logger.info("Application shutting down")

        super().closeEvent(event)


def main():
    """Main application entry point."""
    try:
        multiprocessing.set_start_method('spawn')
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError as e:
        print(f"Note: {str(e)}")

    print("Application starting")

    try:
        print("Initializing database...")
        db = Database()
        db.init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()

    app = QtWidgets.QApplication(sys.argv)
    apply_app_styles(app)

    window = MainWindow()
    window.show()

    def _handle_sigint(signum, frame):
        """Handle Ctrl+C by triggering Qt's close event for clean shutdown."""
        print("\nReceived interrupt signal, initiating shutdown...")
        QtCore.QMetaObject.invokeMethod(
            window, "close", QtCore.Qt.ConnectionType.QueuedConnection
        )

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    sys.exit(app.exec())


if __name__ == "__main__":
    main() 