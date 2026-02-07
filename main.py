"""
Provides the main graphical interface for Dendrite using modular components from dendrite/gui/.
"""

import signal
import sys
import logging
import multiprocessing

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.config.stream_config_manager import initialize_stream_config_manager
from dendrite.gui.config.mode_config_manager import initialize_mode_config_manager
from dendrite.constants import (
    VERSION, APP_NAME,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_POSITION, APPLICATION_TITLE,
)
from dendrite.gui.widgets import (
    ControlButtonsWidget, TopNavigationBar,
    GeneralParametersWidget, StreamConfigurationWidget,
    ModeInstanceBadge, ModeInstanceConfigDialog,
    PreprocessingWidget, OutputWidget,
    LogDisplayWidget, TelemetryWidget,
)
from dendrite.gui.styles.widget_styles import WidgetStyles, apply_app_styles
from dendrite.gui.styles.design_tokens import BG_MAIN, BG_PANEL, SEPARATOR_SUBTLE
from dendrite.gui.pipeline_controller import PipelineController
from dendrite.ml.decoders import get_available_decoders
from dendrite.gui.config import SystemConfigurationManager
from dendrite.utils.logger_central import setup_logger
from dendrite.data.storage.database import Database

class MainWindow(QtWidgets.QMainWindow):
    """
    Main BMI application window — pure UI shell.

    Manages:
    - UI layout and navigation (control/display panels)
    - Configuration persistence and session management
    - User interactions (dialogs, badge management)

    Pipeline lifecycle (processes, queues, IPC) is delegated to PipelineController.
    """

    def __init__(self):
        super().__init__()

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

        # Pipeline controller — owns all process/queue/IPC lifecycle
        self.pipeline_controller = PipelineController(parent=self)
        self.pipeline_controller.recording_started.connect(self._on_recording_started)
        self.pipeline_controller.recording_stopped.connect(self._on_recording_stopped)
        self.pipeline_controller.start_failed.connect(self._on_start_failed)
        self.pipeline_controller.pids_updated.connect(self._on_pids_updated)
        self.pipeline_controller.log_file_ready.connect(self._on_log_file_ready)

        self.init_ui()

    def setup_variables(self):
        """Initialize all instance variables."""
        # UI components (will be initialized in init_ui)
        self.output_widget = None
        self.log_display_widget = None
        self.telemetry_widget = None
        self.display_stack = None

        # View state (must be set before signals connect in __init__)
        self._compact_mode = False

    def is_recording(self) -> bool:
        """Check if a recording is currently active."""
        return self.pipeline_controller.is_recording()

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

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Mode instance badges
    # ------------------------------------------------------------------

    def add_mode_instance_badge(self, mode_instance_config: dict | None = None) -> ModeInstanceBadge | None:
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

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_configuration(self):
        """Load configuration from a JSON file."""
        self.config_manager.load_configuration()

    def retrieve_parameters(self) -> dict:
        """Retrieve all configuration parameters from the GUI."""
        return self.config_manager.build_configuration()

    # ------------------------------------------------------------------
    # Pipeline start / stop (delegates to PipelineController)
    # ------------------------------------------------------------------

    def start_processing(self):
        """Start the BMI processing pipeline."""
        if not self._validate_start_configuration():
            return

        self._set_starting_state()
        config = self.retrieve_parameters()

        def _on_protocol_status(protocol, connected):
            if self.output_widget:
                self.output_widget.set_protocol_connected(protocol, connected)

        self.pipeline_controller.start(config, status_callback=_on_protocol_status)

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
        """Stop the BMI processing pipeline."""
        self.pipeline_controller.stop(blocking=blocking)

    # ------------------------------------------------------------------
    # Pipeline controller signal handlers
    # ------------------------------------------------------------------

    def _on_recording_started(self):
        """Handle successful pipeline start."""
        if self.telemetry_widget:
            self.telemetry_widget.on_recording_started()
        self.set_all_badge_statuses('running')

    def _on_recording_stopped(self):
        """Handle pipeline stop completion."""
        self._clear_display_layout()
        self.control_buttons_widget.set_running_state(False)
        self.stream_config_widget.set_recording_state(False)
        if self.telemetry_widget:
            self.telemetry_widget.on_recording_stopped()
        self.set_all_badge_statuses('ready')

    def _on_start_failed(self, error_msg: str):
        """Handle pipeline start failure."""
        self.logger.error(f"Session start failed: {error_msg}")
        QtWidgets.QMessageBox.critical(
            self,
            "Session Start Failed",
            error_msg,
            QtWidgets.QMessageBox.StandardButton.Ok
        )
        self._revert_start_state()

    def _on_pids_updated(self, mode_pids: dict, system_pids: dict):
        """Handle PID updates from pipeline controller."""
        if self.telemetry_widget:
            self.telemetry_widget.set_mode_pids(mode_pids)
            self.telemetry_widget.set_system_pids(system_pids)

    def _on_log_file_ready(self, log_file: str):
        """Handle log file path from pipeline controller."""
        if self.log_display_widget:
            self.log_display_widget.set_log_file(log_file)

    # ------------------------------------------------------------------
    # GUI state helpers
    # ------------------------------------------------------------------

    def _set_starting_state(self):
        """Set GUI to starting state."""
        self.control_buttons_widget.set_running_state(True)
        self.stream_config_widget.set_recording_state(True)

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
                error_msg,
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
                error_msg,
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False

        self.logger.info("Preprocessing validation passed")
        return True

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

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Handle application close event."""
        if self.pipeline_controller.is_recording():
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
            self.pipeline_controller.stop(blocking=True)

        if self.log_display_widget:
            self.log_display_widget.stop_monitoring()

        if self.telemetry_widget:
            self.telemetry_widget.cleanup()

        self.pipeline_controller.cleanup()

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
        db = Database()
        db.init_db()
    except Exception as e:
        # App can still function without DB (recording works); decoder history
        # and study management will be unavailable.
        logging.warning(f"Database initialization failed: {e}. "
                        "Decoder history and study management will be unavailable.")

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
