"""
Stream Configuration Widget

A pure UI widget for LSL stream discovery, selection, and display.
Delegates all data processing to StreamConfigManager.
"""

from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.design_tokens import ACCENT, BG_ELEVATED, STATUS_OK, TEXT_MUTED
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.utils import load_icon
from dendrite.utils.logger_central import get_logger

from .components import (
    STATUS_DOT,
    clear_layout,
)
from .stream_setup_dialog import StreamSetupDialog


class StreamConfigurationWidget(QtWidgets.QWidget):
    """Pure UI widget for stream discovery, selection, and display."""

    streams_configured = QtCore.pyqtSignal(dict)  # Emitted when streams are configured

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent

        self.stream_manager = get_stream_config_manager()
        self.stream_manager.streams_updated.connect(self._on_streams_updated)

        # UI state
        self._is_recording = False
        self._discover_button_text = "Discover LSL Streams"  # Track button text state

        self.setup_ui()

    def _on_streams_updated(self, streams: dict[str, Any]):
        """Handle stream updates from manager."""
        # Update display to reflect current stream configuration
        self.update_stream_display()

        # Notify main window of stream configuration changes
        # Note: payload is for backward compatibility, main_window should use stream_config_manager directly
        self.streams_configured.emit({"configured_streams": streams})

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(LAYOUT["spacing_sm"])

        # Gap after APPS section
        main_layout.addSpacing(LAYOUT["spacing_xl"])

        # Telemetry-style section header
        header_label = QtWidgets.QLabel("STREAMS")
        header_label.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        main_layout.addWidget(header_label)

        # Discover button
        self.setup_header(main_layout)

        # Spacing before badges
        main_layout.addSpacing(LAYOUT["spacing_sm"])

        # Stream badges
        self.setup_content_area(main_layout)

    def setup_header(self, parent_layout):
        """Set up full-width discover button with integrated status."""
        # Full-width discover/status button
        self.discover_streams_button = QtWidgets.QPushButton("Discover LSL Streams")
        self.discover_streams_button.setStyleSheet(
            WidgetStyles.button(variant="primary", padding="10px 16px")
        )
        self.discover_streams_button.clicked.connect(self.run_lsl_discovery)

        # Add refresh icon
        refresh_icon = load_icon("icons/refresh.svg")
        if not refresh_icon.isNull():
            self.discover_streams_button.setIcon(refresh_icon)
            self.discover_streams_button.setIconSize(QtCore.QSize(16, 16))

        parent_layout.addWidget(self.discover_streams_button)

    def setup_content_area(self, parent_layout):
        """Set up the vertical layout for stream badges."""
        self.stream_container = QtWidgets.QWidget()
        self.stream_layout = QtWidgets.QVBoxLayout(self.stream_container)
        self.stream_layout.setContentsMargins(0, 0, 0, 0)
        self.stream_layout.setSpacing(LAYOUT["spacing_sm"])
        parent_layout.addWidget(self.stream_container)

    def set_recording_state(self, is_recording: bool):
        """Enable or disable stream discovery based on recording state."""
        self._is_recording = is_recording

        if is_recording:
            # Store current text before modifying
            self._discover_button_text = self.discover_streams_button.text()
            self.discover_streams_button.setEnabled(False)
            self.discover_streams_button.setText(f"{self._discover_button_text} (Recording)")
            self.discover_streams_button.setToolTip(
                "Stream discovery is disabled during recording to prevent interruptions.\n"
                "Stop the current recording to discover new streams."
            )
        else:
            # Restore original text
            self.discover_streams_button.setEnabled(True)
            self.discover_streams_button.setText(self._discover_button_text)
            self.discover_streams_button.setToolTip("Click to discover available LSL streams")

    def run_lsl_discovery(self):
        """Open stream discovery dialog (dialog handles discovery internally)."""
        logger = get_logger("StreamConfigWidget")
        logger.info("Opening Stream Discovery dialog...")

        self.discover_streams_button.setEnabled(False)

        try:
            # Dialog handles all discovery internally
            dialog = StreamSetupDialog(self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                logger.info("Stream configuration cancelled by user.")
                return

            # Get configured streams
            configured_streams = dialog.get_configured_streams()

            if not configured_streams:
                return

            # Build selected_streams dict (uid -> type) for compatibility
            selected_streams = {uid: stream.type for uid, stream in configured_streams.items()}
            logger.info(f"User configured {len(configured_streams)} stream(s)")

            # Update GUI with the configured stream information
            self.update_from_stream_selection(
                selected_streams, configured_streams, lambda msg: logger.info(msg)
            )

        except Exception as e:
            logger.error(f"Error during stream discovery: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Stream discovery failed: {e}")
        finally:
            self.discover_streams_button.setEnabled(True)
            self.update_stream_display()

    def update_from_stream_selection(
        self,
        selected_streams: dict[str, str],
        discovered_streams: dict[str, dict[str, Any]],
        gui_log_callback: Callable,
    ):
        """Update the widget configuration based on the user's stream selection."""
        gui_log_callback("Updating stream configuration from selected streams...")

        # Delegate all processing to the manager
        self.stream_manager.process_discovered_streams(selected_streams, discovered_streams)

        gui_log_callback("Stream configuration update completed.")

    def create_stream_badge(self, uid: str, stream_info: dict) -> QtWidgets.QWidget:
        """Create a stream badge with type indicator."""
        badge = QtWidgets.QFrame()
        badge.setStyleSheet(
            WidgetStyles.frame(bg=BG_ELEVATED, border=False, padding=LAYOUT["padding_sm"])
        )

        layout = QtWidgets.QHBoxLayout(badge)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Type indicator (blue accent color)
        stream_type = stream_info.type or "Other"
        type_label = QtWidgets.QLabel(stream_type)
        type_label.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 11px;")
        layout.addWidget(type_label)

        # Stream name (truncated to 25 chars)
        name = stream_info.name or "Unknown"
        display_name = name[:25] + "..." if len(name) > 25 else name
        name_label = QtWidgets.QLabel(display_name)
        name_label.setStyleSheet(WidgetStyles.label(size=11))
        if len(name) > 25:
            name_label.setToolTip(name)
        layout.addWidget(name_label)

        layout.addStretch()

        # Channels and sample rate (muted, right-aligned)
        channels = stream_info.channel_count
        rate = int(stream_info.sample_rate)
        specs_label = QtWidgets.QLabel(f"{channels}ch @ {rate}Hz")
        specs_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(specs_label)

        # Status dot
        status_dot = QtWidgets.QLabel(STATUS_DOT)
        status_dot.setStyleSheet(WidgetStyles.status_icon(STATUS_OK, size=10))
        status_dot.setToolTip("Stream active")
        layout.addWidget(status_dot)

        return badge

    def update_stream_display(self):
        """Update the stream configuration display with discovered streams."""
        streams = self.stream_manager.get_streams()

        # Clear existing stream badges
        clear_layout(self.stream_layout)

        if not streams:
            self.stream_container.hide()
            if not self._is_recording:
                self._discover_button_text = "Discover LSL Streams"
                self.discover_streams_button.setText(self._discover_button_text)
            return

        # Update button text
        if not self._is_recording:
            self._discover_button_text = "Refresh Streams"
            self.discover_streams_button.setText(self._discover_button_text)

        # Create new stream badges
        for uid, stream_info in streams.items():
            stream_badge = self.create_stream_badge(uid, stream_info)
            self.stream_layout.addWidget(stream_badge)

        self.stream_container.show()
