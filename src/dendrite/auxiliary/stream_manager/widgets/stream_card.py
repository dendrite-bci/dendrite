"""StreamCard widget for offline streaming GUI."""

import time
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.gui.styles.design_tokens import (
    ACCENT,
    ACCENT_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER_INPUT,
    RADIUS,
    SPACING,
    STATUS_ERROR,
    STATUS_OK,
    TEXT_DISABLED,
)
from dendrite.gui.styles.widget_styles import WidgetStyles
from dendrite.gui.widgets.common import StatusPillWidget

from dendrite.auxiliary.stream_manager.dialogs.stream_dialogs import StreamDetailsDialog
from dendrite.auxiliary.stream_manager.utils import get_file_info, get_source_display


class StreamCard(QtWidgets.QFrame):
    """Stream configuration card with remove and details functionality."""

    start = QtCore.pyqtSignal(dict)
    stop = QtCore.pyqtSignal(str)
    remove = QtCore.pyqtSignal(str)

    def __init__(self, stream_id: str, config: dict, parent=None):
        super().__init__(parent)
        self.stream_id = stream_id
        self.config = config
        self.is_running = False
        self.total_duration = 0.0  # Total duration in seconds
        self.current_position = 0.0  # Current position in seconds
        self.start_time = None  # When streaming started

        # Calculate duration for file-based streams
        if self.config.get("source_type") == "file" and self.config.get("file_path"):
            self._calculate_file_duration()

        self.setup_ui()
        self.apply_styles()
        self.setToolTip(self._build_tooltip())
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

    def _build_tooltip(self) -> str:
        """Build rich tooltip with stream metadata."""
        source_display = get_source_display(self.config)
        source_type = self.config.get("source_type", "generated")

        lines = [
            f"<b>{self.config.get('name', 'Unnamed')}</b>",
            f"Type: {self.config['type']}",
            f"Source: {source_display}",
            f"Channels: {self.config.get('channels', '-')}",
            f"Sample Rate: {self.config.get('sample_rate', '-')} Hz",
        ]

        if source_type == "moabb":
            lines.append(f"Subject: {self.config.get('subject_id', '?')}")
            if self.config.get("enable_events"):
                lines.append("Events: Separate stream enabled")

        elif source_type == "file" and self.config.get("file_path"):
            lines.append(f"File: {Path(self.config['file_path']).name}")
            file_info = get_file_info(self.config["file_path"])
            if file_info:
                duration, events, event_ids = file_info
                lines.append(f"Duration: {duration:.1f}s")
                if event_ids:
                    lines.append(f"Events: {len(events)} ({len(event_ids)} types)")
            if self.config.get("enable_events"):
                lines.append("Events: Separate stream enabled")

        return "<br>".join(lines)

    def mousePressEvent(self, event):
        """Open details dialog on click."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            dialog = StreamDetailsDialog(self.config, self)
            dialog.exec()

    def _calculate_file_duration(self):
        """Calculate the total duration of the file in seconds using cache."""
        file_path = self.config["file_path"]
        file_info = get_file_info(file_path)

        if file_info:
            self.total_duration = file_info[0]  # duration is first element
        else:
            self.total_duration = 0.0

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        layout.setSpacing(SPACING["md"])

        # Stream type and name
        header = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel(self.config["type"])
        type_label.setStyleSheet(WidgetStyles.label(variant="header"))
        header.addWidget(type_label)
        header.addStretch()

        # Status indicator (pill-shaped slider)
        self.status = StatusPillWidget()
        header.addWidget(self.status)
        layout.addLayout(header)

        details = QtWidgets.QHBoxLayout()

        # Clip name to reasonable length
        stream_name = self.config.get("name", "Unnamed")
        if len(stream_name) > 60:
            stream_name = stream_name[:57] + "..."
        name = QtWidgets.QLabel(stream_name)
        name.setStyleSheet(WidgetStyles.label())

        # Source label based on source type
        source_type = self.config.get("source_type", "generated")
        if source_type == "moabb":
            subject = self.config.get("subject_id", "?")
            source_text = f"MOABB: S{subject}"
        elif source_type == "file" and self.config.get("file_path"):
            file_name = Path(self.config["file_path"]).stem
            stream_name_check = self.config.get("name", "").lower()
            # If filename is already in the stream name, just show "File"
            if file_name.lower() in stream_name_check:
                source_text = "File"
            else:
                file_display = Path(self.config["file_path"]).name
                if len(file_display) > 40:
                    file_display = file_display[:37] + "..."
                source_text = file_display
        else:
            source_text = "Synthetic"

        source = QtWidgets.QLabel(source_text)
        source.setStyleSheet(WidgetStyles.label())
        details.addWidget(name)
        details.addWidget(QtWidgets.QLabel(" | "))
        details.addWidget(source)
        details.addStretch()
        layout.addLayout(details)

        # Progress bar for file-based and MOABB streams
        source_type = self.config.get("source_type")
        show_progress = (
            source_type == "file" and self.total_duration > 0
        ) or source_type == "moabb"

        if show_progress:
            progress_layout = QtWidgets.QHBoxLayout()

            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedHeight(6)

            if source_type == "moabb":
                # MOABB: start static, animate only when loading
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.time_label = QtWidgets.QLabel("Ready")
            else:
                # File: Known duration
                self.progress_bar.setMinimum(0)
                self.progress_bar.setMaximum(int(self.total_duration))
                self.progress_bar.setValue(0)
                self.time_label = QtWidgets.QLabel(
                    self._format_time_display(0, self.total_duration)
                )

            self.time_label.setStyleSheet(WidgetStyles.label(size="small"))

            progress_layout.addWidget(self.progress_bar)
            progress_layout.addWidget(self.time_label)
            layout.addLayout(progress_layout)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(SPACING["sm"])
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.remove_btn = QtWidgets.QPushButton("\u2715")
        self.remove_btn.clicked.connect(self._remove)
        self.remove_btn.setFixedWidth(32)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.remove_btn)
        layout.addLayout(buttons)

        # Adjust height based on whether progress bar is present
        height = 140 if show_progress else 115
        self.setFixedHeight(height)

    def _format_time_display(self, current: float, total: float) -> str:
        """Format time display as current/total (mm:ss format)."""

        def format_seconds(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        current_str = format_seconds(current)
        total_str = format_seconds(total)
        return f"{current_str} / {total_str}"

    def set_progress(self, progress_pct: float):
        """Update progress bar from backend progress report.

        Args:
            progress_pct: Progress as fraction (0.0 to 1.0)
        """
        if not self.is_running or not hasattr(self, "progress_bar"):
            return

        self.current_position = progress_pct * self.total_duration

        # Update progress bar
        self.progress_bar.setValue(int(self.current_position))

        # Update time label
        if hasattr(self, "time_label"):
            self.time_label.setText(
                self._format_time_display(self.current_position, self.total_duration)
            )

        # Auto-stop stream when file playback completes
        if progress_pct >= 1.0:
            source_type = self.config.get("source_type")
            if source_type in ("file", "moabb") and self.is_running:
                from ..utils import logger

                logger.info(
                    f"Playback completed for {self.config.get('name')}, auto-stopping stream"
                )
                self._stop()

    def apply_styles(self):
        # Stream card frame styling - subtle outline with lighter base
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border: 1px solid {BORDER_INPUT};
                border-radius: {RADIUS["md"]}px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QProgressBar {{
                background: {BG_INPUT};
                border: none;
                border-radius: {RADIUS["sm"]}px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {ACCENT_HOVER});
                border-radius: {RADIUS["sm"]}px;
            }}
        """)

        self.start_btn.setStyleSheet(WidgetStyles.button())
        self.stop_btn.setStyleSheet(WidgetStyles.button(text_color=STATUS_ERROR, severity="error"))
        self.remove_btn.setStyleSheet(WidgetStyles.button(variant="secondary", size="small"))

    def _start(self):
        self.start.emit(self.config)

    def _stop(self):
        self.stop.emit(self.stream_id)

    def _remove(self):
        self.remove.emit(self.stream_id)

    def set_running(self, running: bool, loading: bool = False):
        self.is_running = running

        # Update status pill (green when running, gray when stopped)
        status_color = STATUS_OK if running else TEXT_DISABLED
        self.status.set_status(status_color, active=running)

        # Set button text based on state
        if running:
            if loading:
                self.start_btn.setText("Loading...")
            else:
                self.start_btn.setText("Running")
        else:
            self.start_btn.setText("Start")

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.remove_btn.setEnabled(not running)

        # Handle progress tracking
        if hasattr(self, "progress_bar"):
            source_type = self.config.get("source_type")
            if running:
                self.start_time = time.time()
                self.current_position = 0.0

                if source_type == "file" and self.total_duration > 0:
                    # File: known duration, show timed progress
                    self.progress_bar.setValue(0)
                    if hasattr(self, "time_label"):
                        self.time_label.setText(self._format_time_display(0, self.total_duration))
                elif source_type == "moabb":
                    # MOABB: indeterminate animation until duration received
                    self.progress_bar.setRange(0, 0)  # Looping animation
                    if hasattr(self, "time_label"):
                        self.time_label.setText("Loading...")
            else:
                self.start_time = None
                # Reset progress bar to static empty state
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                if source_type == "moabb" and hasattr(self, "time_label"):
                    self.time_label.setText("Ready")

    def set_duration(self, duration: float):
        """Switch from indeterminate to timed progress (called when MOABB data loads)."""
        self.total_duration = duration
        if hasattr(self, "progress_bar"):
            self.progress_bar.setRange(0, int(duration))
            self.progress_bar.setValue(0)
        if hasattr(self, "time_label"):
            self.time_label.setText(self._format_time_display(0, duration))
        self.start_btn.setText("Running")
