"""
Control Buttons Widget

A toggle button for Start/Stop with proper state management.
"""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import STATUS_ERROR
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class ControlButtonsWidget(QtWidgets.QWidget):
    """Widget with a single toggle button for start/stop control."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(LAYOUT["margin"], 0, LAYOUT["margin"], LAYOUT["margin"])

        self.toggle_button = QtWidgets.QPushButton("Start")
        self.toggle_button.setStyleSheet(WidgetStyles.button(padding="10px 16px"))
        self.toggle_button.clicked.connect(self._on_click)
        layout.addWidget(self.toggle_button)

    def _on_click(self):
        """Handle button click - emit appropriate signal based on state."""
        if self._is_running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_running_state(self, is_running: bool):
        """Update button appearance based on running state."""
        self._is_running = is_running
        if is_running:
            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet(
                WidgetStyles.button(padding="10px 16px", text_color=STATUS_ERROR, severity="error")
            )
        else:
            self.toggle_button.setText("Start")
            self.toggle_button.setStyleSheet(WidgetStyles.button(padding="10px 16px"))

    def set_enabled(self, enabled: bool):
        """Enable or disable the button."""
        self.toggle_button.setEnabled(enabled)
