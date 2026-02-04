"""Main window for offline streaming GUI."""

import multiprocessing
import re
import sys
import time
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.stream_manager.backend import OfflineDataStreamer
from dendrite.gui.styles.design_tokens import BG_MAIN, BG_PANEL, SPACING, STATUS_ERROR
from dendrite.gui.styles.widget_styles import WidgetStyles, apply_app_styles
from dendrite.gui.utils import set_app_icon
from dendrite.gui.widgets.common.pill_navigation import PillNavigation
from dendrite.utils.logger_central import INFO, get_logger, setup_logger

from .dialogs import FileStreamDialog, MOAABSubjectDialog, StreamConfigDialog
from .widgets import InternalDatasetsPanel, MOAABPresetPanel, StreamCard


class StreamManager(QtWidgets.QMainWindow):
    """Main window for stream management with remove functionality."""

    def __init__(self):
        super().__init__()
        setup_logger("StreamManager", level=INFO)
        self.logger = get_logger("StreamManager")
        self.streams: dict = {}
        self.processes: dict = {}
        self.stops: dict = {}
        self.info_queues: dict = {}  # stream_id -> Queue for MOABB duration
        self.setup_ui()
        self.apply_styles()

        # Poll for MOABB duration info
        self.info_timer = QtCore.QTimer()
        self.info_timer.timeout.connect(self._poll_info_queues)
        self.info_timer.start(100)

        self.logger.info("Stream Manager initialized")

    def _clip_text_for_status(self, text: str, max_length: int = 60) -> str:
        """Clip text for status bar display to prevent overflow."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def _extract_clean_filename(self, file_path: str) -> str:
        """Extract a clean filename prefix by removing common timestamp suffixes."""
        filename = Path(file_path).stem
        # Remove timestamp patterns from end of filename
        clean_name = re.sub(r"_\d{6,8}(_\d{6})?$", "", filename)
        clean_name = re.sub(r"_(session|run)\d+$", "", clean_name, flags=re.IGNORECASE)
        return clean_name

    def setup_ui(self):
        self.setWindowTitle("Stream Manager")
        self.setGeometry(100, 100, 900, 500)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for dataset panels and stream area
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Left: Panel with pill navigation
        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("left_panel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        left_layout.setSpacing(SPACING["md"])

        # Pill navigation for dataset source
        self.source_nav = PillNavigation(
            tabs=[("moabb", "MOABB"), ("internal", "Internal")], size="medium"
        )
        self.source_nav.section_changed.connect(self._on_source_changed)
        left_layout.addWidget(self.source_nav)

        # Stacked widget for panels
        self.panel_stack = QtWidgets.QStackedWidget()

        self.moabb_panel = MOAABPresetPanel()
        self.moabb_panel.preset_selected.connect(self._on_moabb_preset_selected)
        self.panel_stack.addWidget(self.moabb_panel)

        self.internal_panel = InternalDatasetsPanel()
        self.internal_panel.recording_selected.connect(self._on_internal_recording_selected)
        self.panel_stack.addWidget(self.internal_panel)

        left_layout.addWidget(self.panel_stack)
        left_panel.setFixedWidth(280)
        splitter.addWidget(left_panel)

        # Right: Stream cards area
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(SPACING["xl"], SPACING["xl"], SPACING["xl"], SPACING["xl"])

        self.title = QtWidgets.QLabel("Streams")
        right_layout.addWidget(self.title)

        # Stream area
        self.stream_area = QtWidgets.QScrollArea()
        self.stream_area.setWidgetResizable(True)
        self.stream_widget = QtWidgets.QWidget()
        self.stream_layout = QtWidgets.QVBoxLayout(self.stream_widget)
        self.stream_layout.addStretch()
        self.stream_area.setWidget(self.stream_widget)
        right_layout.addWidget(self.stream_area)

        buttons = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("+ Add Custom")
        self.add_btn.clicked.connect(self._add)
        self.stop_all_btn = QtWidgets.QPushButton("Stop All")
        self.stop_all_btn.setStyleSheet(WidgetStyles.button_stop)
        self.stop_all_btn.clicked.connect(self._stop_all)
        self.stop_all_btn.setEnabled(False)
        buttons.addStretch()
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.stop_all_btn)
        right_layout.addLayout(buttons)

        splitter.addWidget(right_widget)

        # Set initial splitter sizes (left panel: 250px, right: rest)
        splitter.setSizes([250, 650])

        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Ready")

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_MAIN};
            }}
            #left_panel {{
                background: {BG_PANEL};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
        """)

        # Apply V2 button styling
        self.add_btn.setStyleSheet(WidgetStyles.button())
        self.stop_all_btn.setStyleSheet(
            WidgetStyles.button(text_color=STATUS_ERROR, severity="error")
        )

        # Apply title styling
        self.title.setStyleSheet(WidgetStyles.label(variant="title"))

    def _add(self):
        dialog = StreamConfigDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._create_stream_card(config)

    def _on_source_changed(self, source: str):
        """Handle dataset source tab change."""
        if source == "moabb":
            self.panel_stack.setCurrentIndex(0)
        else:
            self.panel_stack.setCurrentIndex(1)

    def _on_moabb_preset_selected(self, preset_name: str, config):
        """Handle MOABB preset selection from sidebar."""
        # Load details if not already loaded (BIDS datasets are lazy-loaded)
        if not config.subjects:
            from dendrite.data import load_moabb_dataset_details

            load_moabb_dataset_details(config)
        dialog = MOAABSubjectDialog(preset_name, config, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            stream_config = dialog.get_config()
            self._create_stream_card(stream_config)

    def _on_internal_recording_selected(self, fif_path: str):
        """Handle internal dataset recording selection - show dialog with events option."""
        dialog = FileStreamDialog(fif_path, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._create_stream_card(config)
            self.statusBar().showMessage(f"Added: {config['name']}")

    def _create_stream_card(self, config: dict):
        """Create and add a stream card with the given config."""
        stream_id = f"stream_{time.strftime('%H%M%S')}_{len(self.streams)}"
        self.streams[stream_id] = config
        card = StreamCard(stream_id, config, self)
        card.start.connect(self._start)
        card.stop.connect(self._stop)
        card.remove.connect(self._remove)
        self.stream_layout.insertWidget(self.stream_layout.count() - 1, card)
        self.logger.info(f"Added stream: {config['name']}")

    def _start(self, config: dict):
        stream_id = next(sid for sid, sc in self.streams.items() if sc == config)
        try:
            stop_event = multiprocessing.Event()
            self.stops[stream_id] = stop_event

            # Extract basic parameters
            stream_type = config["type"]
            channels = config.get("channels", 1)
            sample_rate = config.get("sample_rate", 100.0)

            # Handle MOABB source type
            if config.get("source_type") == "moabb":
                self.logger.info(
                    f"Starting MOABB stream: {config['name']} "
                    f"(preset={config['preset_name']}, subject={config['subject_id']})"
                )

                info_queue = multiprocessing.Queue()
                self.info_queues[stream_id] = info_queue

                streamer = OfflineDataStreamer(
                    sample_rate=sample_rate,
                    stop_event=stop_event,
                    data_type="EEG",
                    channels=channels,
                    moabb_preset=config["preset_name"],
                    moabb_subject=config["subject_id"],
                    enable_event_stream=config.get("enable_events", False),
                    stream_name_prefix=config["name"],
                    info_queue=info_queue,
                )
            else:
                # File path handling
                stream_name_prefix = None
                data_file_path = None

                if config.get("source_type") == "file" and config.get("file_path"):
                    clean_name = self._extract_clean_filename(config["file_path"])
                    stream_name_prefix = clean_name
                    data_file_path = config["file_path"]

                self.logger.info(
                    f"Starting {stream_type} stream: {config['name']} ({channels}ch @ {sample_rate}Hz)"
                )

                # Create info_queue for file-based streams to report progress
                info_queue = None
                if config.get("source_type") == "file":
                    info_queue = multiprocessing.Queue()
                    self.info_queues[stream_id] = info_queue

                streamer = OfflineDataStreamer(
                    sample_rate=sample_rate,
                    stop_event=stop_event,
                    data_type=stream_type,
                    channels=channels,
                    data_file_path=data_file_path,
                    stream_name_prefix=stream_name_prefix,
                    info_queue=info_queue,
                    enable_event_stream=config.get("enable_events", False),
                )

            streamer.start()
            self.processes[stream_id] = streamer

            # Show loading state for MOABB (data may need to download)
            is_moabb = config.get("source_type") == "moabb"
            self._update_card(stream_id, True, loading=is_moabb)
            self.stop_all_btn.setEnabled(True)

            status_msg = f"Started: {config['name']} ({stream_type})"
            if is_moabb:
                status_msg += " - Loading data..."
            self.statusBar().showMessage(self._clip_text_for_status(status_msg))
            self.logger.info(f"Successfully started stream: {config['name']} ({stream_type})")

        except Exception as e:
            error_msg = f"Failed to start stream {config['name']}: {e!s}"
            self.logger.error(error_msg)
            self.statusBar().showMessage(error_msg)
            if stream_id in self.stops:
                del self.stops[stream_id]

    def _stop(self, stream_id: str):
        if stream_id not in self.processes:
            return

        stream_name = self.streams.get(stream_id, {}).get("name", stream_id)

        try:
            self.logger.info(f"Stopping stream: {stream_name}")

            # Signal stop and wait for graceful shutdown
            self.stops[stream_id].set()
            process = self.processes[stream_id]
            process.join(timeout=3.0)

            # Force terminate if still alive
            if process.is_alive():
                self.logger.warning(f"Force terminating stream: {stream_name}")
                process.terminate()
                process.join(timeout=1.0)

            del self.processes[stream_id]
            del self.stops[stream_id]
            self.info_queues.pop(stream_id, None)
            self._update_card(stream_id, False)
            self.stop_all_btn.setEnabled(bool(self.processes))

            status_msg = f"Stopped: {stream_name}"
            self.statusBar().showMessage(self._clip_text_for_status(status_msg))
            self.logger.info(f"Successfully stopped stream: {stream_name}")

        except Exception as e:
            self.logger.error(f"Error stopping stream {stream_name}: {e!s}")
            self.statusBar().showMessage(f"Error stopping {stream_name}")
            self.processes.pop(stream_id, None)
            self.stops.pop(stream_id, None)
            self.info_queues.pop(stream_id, None)
            self._update_card(stream_id, False)

    def _remove(self, stream_id: str):
        if stream_id not in self.streams:
            return

        stream_name = self.streams[stream_id].get("name", stream_id)

        # Stop if running
        if stream_id in self.processes:
            self._stop(stream_id)

        # Remove from streams and UI
        del self.streams[stream_id]

        # Find and remove the card widget
        for i in range(self.stream_layout.count() - 1):
            widget = self.stream_layout.itemAt(i).widget()
            if isinstance(widget, StreamCard) and widget.stream_id == stream_id:
                widget.deleteLater()
                break

        status_msg = f"Removed: {stream_name}"
        self.statusBar().showMessage(self._clip_text_for_status(status_msg))
        self.logger.info(f"Removed stream: {stream_name}")

    def _stop_all(self):
        if not self.processes:
            return

        stream_count = len(self.processes)
        self.logger.info(f"Stopping all {stream_count} running streams...")
        self.statusBar().showMessage(f"Stopping {stream_count} streams...")

        # Stop all streams
        for stream_id in list(self.processes.keys()):
            self._stop(stream_id)

        self.statusBar().showMessage("All streams stopped")
        self.logger.info("All streams stopped")

    def _update_card(self, stream_id: str, running: bool, loading: bool = False):
        for i in range(self.stream_layout.count() - 1):
            widget = self.stream_layout.itemAt(i).widget()
            if isinstance(widget, StreamCard) and widget.stream_id == stream_id:
                widget.set_running(running, loading=loading)
                break

    def _get_card(self, stream_id: str) -> StreamCard | None:
        """Get StreamCard widget by stream_id."""
        for i in range(self.stream_layout.count() - 1):
            widget = self.stream_layout.itemAt(i).widget()
            if isinstance(widget, StreamCard) and widget.stream_id == stream_id:
                return widget
        return None

    def _poll_info_queues(self):
        """Poll info queues for duration and progress updates."""
        for stream_id, queue in list(self.info_queues.items()):
            try:
                # Process all available messages
                while not queue.empty():
                    msg = queue.get_nowait()
                    card = self._get_card(stream_id)
                    if not card:
                        continue

                    # Handle dict messages (new format)
                    if isinstance(msg, dict):
                        msg_type = msg.get("type")
                        value = msg.get("value")
                        if msg_type == "duration":
                            card.set_duration(value)
                        elif msg_type == "progress":
                            card.set_progress(value)
                    else:
                        # Legacy: plain float = duration
                        card.set_duration(msg)
            except (AttributeError, TypeError, KeyError):
                pass

    def closeEvent(self, event):
        self._stop_all()
        event.accept()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Stream Manager")

    # Set application icon
    set_app_icon(app, "icons/stream.svg")

    # Apply global app styles
    apply_app_styles(app)

    window = StreamManager()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
