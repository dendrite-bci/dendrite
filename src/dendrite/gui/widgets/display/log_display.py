"""
Log Display Widget

A widget for displaying and managing application logs with filtering and controls.
Encapsulates all log-related functionality that was previously in the main window.
"""

import logging
import os

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.utils.logger_central import get_logger, set_level

logger = get_logger(__name__)


class LogDisplayWidget(QtWidgets.QGroupBox):
    """Widget for displaying and managing application logs."""

    LOG_TIMER_INTERVAL = 100  # ms
    MAX_LOG_BLOCKS = 5000
    MIN_LOG_HEIGHT = 300
    CONTROL_HEIGHT = 32

    # Log level configuration: level -> (numeric_value, logging_constant, is_critical)
    LOG_LEVEL_CONFIG = {
        "DEBUG": (10, logging.DEBUG, False),
        "INFO": (20, logging.INFO, False),
        "WARNING": (30, logging.WARNING, False),
        "ERROR": (40, logging.ERROR, True),
        "CRITICAL": (50, logging.CRITICAL, True),
    }
    LOG_LEVELS = list(LOG_LEVEL_CONFIG.keys())
    CRITICAL_LEVELS = [k for k, v in LOG_LEVEL_CONFIG.items() if v[2]]

    def __init__(self, parent=None) -> None:
        super().__init__("", parent)
        self.parent_window = parent

        self.current_log_file = None
        self.current_log_position = 0
        self.current_process_filter = "All"
        self.detected_processes = set(["All"])

        self.setup_ui()
        self.setup_log_monitoring()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        self.setStyleSheet(WidgetStyles.groupbox())
        log_layout = QtWidgets.QVBoxLayout(self)
        # Reduced top margin since no title is displayed
        log_layout.setContentsMargins(LAYOUT["margin"], 4, LAYOUT["margin"], LAYOUT["margin"])
        log_layout.setSpacing(LAYOUT["spacing"])

        self.log_display = QtWidgets.QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_display.setMaximumBlockCount(self.MAX_LOG_BLOCKS)
        self.log_display.setStyleSheet(WidgetStyles.textedit_log)
        self.log_display.setMinimumHeight(self.MIN_LOG_HEIGHT)

        log_layout.addWidget(self.log_display)

        log_controls = QtWidgets.QHBoxLayout()
        log_controls.setSpacing(LAYOUT["spacing"])

        clear_log_button = QtWidgets.QPushButton("Clear Log")
        clear_log_button.setStyleSheet(WidgetStyles.button(variant="secondary"))
        clear_log_button.setFixedHeight(self.CONTROL_HEIGHT)
        clear_log_button.clicked.connect(self.log_display.clear)
        log_controls.addWidget(clear_log_button)

        log_level_label = QtWidgets.QLabel("Log Level:")
        log_level_label.setStyleSheet(WidgetStyles.label())
        log_controls.addWidget(log_level_label)

        self.log_level_combo = QtWidgets.QComboBox()
        self.log_level_combo.addItems(self.LOG_LEVELS)
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.setStyleSheet(WidgetStyles.combobox())
        self.log_level_combo.setFixedHeight(self.CONTROL_HEIGHT)
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)
        log_controls.addWidget(self.log_level_combo)

        process_filter_label = QtWidgets.QLabel("Filter by Process:")
        process_filter_label.setStyleSheet(WidgetStyles.label())
        self.process_filter_combobox = QtWidgets.QComboBox()
        self.process_filter_combobox.addItem("All")
        self.process_filter_combobox.setCurrentText("All")
        self.process_filter_combobox.setStyleSheet(WidgetStyles.combobox())
        self.process_filter_combobox.setFixedHeight(self.CONTROL_HEIGHT)
        self.process_filter_combobox.setMinimumWidth(120)
        self.process_filter_combobox.currentTextChanged.connect(self.update_process_filter)

        log_controls.addWidget(process_filter_label)
        log_controls.addWidget(self.process_filter_combobox)
        log_controls.addStretch(1)

        self.autoscroll_checkbox = QtWidgets.QCheckBox("Auto-scroll")
        self.autoscroll_checkbox.setChecked(True)
        self.autoscroll_checkbox.setStyleSheet(WidgetStyles.checkbox)
        self.autoscroll_checkbox.stateChanged.connect(self.toggle_autoscroll)
        log_controls.addWidget(self.autoscroll_checkbox)

        log_layout.addLayout(log_controls)

    def setup_log_monitoring(self) -> None:
        """Set up log file monitoring."""
        self.log_timer = QtCore.QTimer()
        self.log_timer.timeout.connect(self.read_log_file)
        self.log_timer.start(self.LOG_TIMER_INTERVAL)
        self.log_display.appendPlainText("Log display initialized - waiting for log file")

    def set_log_file(self, log_file_path: str):
        """Set the log file to monitor."""
        self.current_log_file = log_file_path
        self.current_log_position = 0
        if log_file_path:
            self.log_display.appendPlainText(f"Monitoring log file: {log_file_path}")

    def extract_process_name(self, log_line: str) -> str | None:
        """Extract process name and any sub-types from log line."""
        parts = log_line.split(" - ", 3)
        if len(parts) < 3:
            return None

        process_name = parts[2].strip()

        if len(parts) >= 4:
            message = parts[3].strip()

            if "decoder_type=" in message and "model_type=" in message:
                decoder = message.split("decoder_type=")[1].split(",")[0].strip()
                model = message.split("model_type=")[1].split(",")[0].split(")")[0].strip()
                return f"{process_name}-{decoder}-{model}"

            elif "for classifier" in message:
                clf_name = message.split("for classifier")[1].split()[0].strip().rstrip(".:,")
                if clf_name:
                    return f"{process_name}-{clf_name}"

        return process_name

    def _extract_log_level(self, log_line: str) -> str | None:
        """Extract the log level from a log line."""
        parts = log_line.split(" - ", 3)
        if len(parts) >= 2:
            return parts[1].strip()
        return None

    def _should_show_log_line(self, log_line, current_level_filter):
        """Check if a log line should be shown based on the current severity filter."""
        # Get the minimum level to show (default to INFO = 20)
        min_level_value = self.LOG_LEVEL_CONFIG.get(current_level_filter, (20,))[0]

        # Extract the level from the log line
        line_level = self._extract_log_level(log_line)
        if not line_level:
            return True  # Show lines without clear level (e.g., stack traces)

        # Get the numeric value for the line's level
        line_level_value = self.LOG_LEVEL_CONFIG.get(line_level, (0,))[0]

        # Show if line level is >= minimum level
        return line_level_value >= min_level_value

    def update_process_filter_dropdown(self) -> None:
        """Update the process filter dropdown with newly detected processes."""
        current_selection = self.process_filter_combobox.currentText()

        self.process_filter_combobox.blockSignals(True)

        self.process_filter_combobox.clear()
        for process_name in sorted(self.detected_processes):
            self.process_filter_combobox.addItem(process_name)

        if current_selection in self.detected_processes:
            self.process_filter_combobox.setCurrentText(current_selection)
        else:
            self.process_filter_combobox.setCurrentText("All")

        self.process_filter_combobox.blockSignals(False)

    def read_log_file(self) -> None:
        """Read new content from the log file and update the display."""
        if not self.current_log_file or not os.path.exists(self.current_log_file):
            return

        try:
            with open(self.current_log_file, encoding="utf-8", errors="replace") as f:
                f.seek(self.current_log_position)
                new_lines = f.readlines()
                self.current_log_position = f.tell()

            if not new_lines:
                return

            process_filter = self.current_process_filter
            current_log_level = self.log_level_combo.currentText()
            process_dropdown_updated = False

            for line in new_lines:
                line = line.rstrip()
                if not line:
                    continue

                process_name = self.extract_process_name(line)
                if process_name and process_name not in self.detected_processes:
                    self.detected_processes.add(process_name)
                    process_dropdown_updated = True

                # Apply filters
                if self._apply_filters_to_line(line, process_filter, current_log_level):
                    self.log_display.appendPlainText(line)

            if process_dropdown_updated:
                self.update_process_filter_dropdown()

            self._autoscroll_to_bottom()

        except OSError as e:
            if not hasattr(self, "_last_log_error") or self._last_log_error != str(e):
                logger.error(f"Error reading log file '{self.current_log_file}': {e}")
                self._last_log_error = str(e)

    def change_log_level(self, level_text: str) -> None:
        """Change the log level for the display handler and refresh display."""
        level = self.LOG_LEVEL_CONFIG.get(level_text, (0, logging.INFO))[1]

        set_level(level)
        log_display_logger = get_logger("LogDisplayWidget")
        log_display_logger.info(f"Log level changed to {level_text}")

        # Refresh the display to apply the new filter
        self._refresh_display_with_current_filters()

    def toggle_autoscroll(self, state: int) -> None:
        """Toggle autoscroll in the log display."""
        is_checked = state == QtCore.Qt.CheckState.Checked.value
        self.autoscroll_checkbox.setChecked(is_checked)

    def update_process_filter(self, filter_text: str) -> None:
        """Store the selected process filter and refresh display."""
        self.current_process_filter = filter_text
        log_display_logger = get_logger("LogDisplayWidget")
        log_display_logger.info(f"Log filter set to process: {filter_text}")

        # Refresh the display to apply the new filter
        self._refresh_display_with_current_filters()

    def append_message(self, message: str) -> None:
        """Append a message to the log display."""
        self.log_display.appendPlainText(message)
        self._autoscroll_to_bottom()

    def _refresh_display_with_current_filters(self) -> None:
        """Re-read and filter the entire log file with current filter settings."""
        if not self.current_log_file or not os.path.exists(self.current_log_file):
            return

        try:
            # Clear current display
            self.log_display.clear()

            # Re-read entire file and apply filters
            with open(self.current_log_file, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                self.current_log_position = f.tell()

            process_filter = self.current_process_filter
            current_log_level = self.log_level_combo.currentText()

            for line in all_lines:
                line = line.rstrip()
                if not line:
                    continue

                # Apply filters
                if self._apply_filters_to_line(line, process_filter, current_log_level):
                    self.log_display.appendPlainText(line)

            # Auto-scroll to bottom
            self._autoscroll_to_bottom()

        except OSError as e:
            logger.error(f"Error refreshing log display from '{self.current_log_file}': {e}")

    def _autoscroll_to_bottom(self) -> None:
        """Scroll log display to bottom if autoscroll enabled."""
        if self.autoscroll_checkbox.isChecked():
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _apply_filters_to_line(self, line: str, process_filter: str, level_filter: str) -> bool:
        """
        Check if line passes process and severity filters.

        Args:
            line: Log line to check
            process_filter: Process name filter ("All" or specific process)
            level_filter: Minimum log level to show

        Returns:
            True if line should be displayed, False otherwise
        """
        # Process filter
        if process_filter != "All":
            if process_filter not in line and not any(lvl in line for lvl in self.CRITICAL_LEVELS):
                return False

        # Severity filter
        return self._should_show_log_line(line, level_filter)

    def stop_monitoring(self) -> None:
        """Stop log file monitoring."""
        if hasattr(self, "log_timer") and self.log_timer is not None:
            self.log_timer.stop()
            self.log_timer = None
