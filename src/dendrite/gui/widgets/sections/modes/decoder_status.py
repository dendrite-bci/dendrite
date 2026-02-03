"""
Decoder status display widget for mode configuration dialogs.

Provides a structured widget showing decoder validation status with icon badges
instead of text-based status indicators.
"""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    BG_ERROR_SUBTLE,
    BG_INPUT,
    BG_OK_SUBTLE,
    BG_WARN_SUBTLE,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class DecoderStatusWidget(QtWidgets.QFrame):
    """Structured decoder status display with icon badges and details section."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.set_empty("No decoder selected")

    def _setup_ui(self):
        self.setMinimumHeight(120)
        self.setStyleSheet(
            WidgetStyles.frame(bg=BG_INPUT, border=True, radius=LAYOUT["radius"], padding=0)
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header row: status dot + decoder name + status badge
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(8)

        self._status_dot = QtWidgets.QLabel()
        self._status_dot.setFixedSize(10, 10)
        header_row.addWidget(self._status_dot)

        self._name_label = QtWidgets.QLabel()
        self._name_label.setStyleSheet(WidgetStyles.label("small", weight="bold"))
        header_row.addWidget(self._name_label, stretch=1)

        self._status_badge = QtWidgets.QLabel()
        self._status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status_badge.setFixedHeight(20)
        self._status_badge.setMinimumWidth(50)
        header_row.addWidget(self._status_badge)

        layout.addLayout(header_row)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {TEXT_MUTED}; max-height: 1px;")
        layout.addWidget(separator)

        # Details section
        self._details_widget = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(self._details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)

        self._requires_label = QtWidgets.QLabel()
        self._requires_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_LABEL))
        self._requires_label.setWordWrap(True)
        details_layout.addWidget(self._requires_label)

        self._system_label = QtWidgets.QLabel()
        self._system_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_LABEL))
        self._system_label.setWordWrap(True)
        details_layout.addWidget(self._system_label)

        self._note_label = QtWidgets.QLabel()
        self._note_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_MUTED))
        self._note_label.setWordWrap(True)
        self._note_label.setVisible(False)
        details_layout.addWidget(self._note_label)

        layout.addWidget(self._details_widget)
        layout.addStretch()

    def _set_status_dot(self, color: str):
        self._status_dot.setStyleSheet(f"""
            background-color: {color};
            border-radius: 5px;
        """)

    def _set_status_badge(self, text: str, bg_color: str, text_color: str):
        if text:
            self._status_badge.setText(text)
            self._status_badge.setStyleSheet(f"""
                background-color: {bg_color};
                color: {text_color};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            """)
            self._status_badge.setVisible(True)
        else:
            self._status_badge.setVisible(False)

    def _set_background(self, color: str):
        self.setStyleSheet(
            WidgetStyles.frame(bg=color, border=True, radius=LAYOUT["radius"], padding=0)
        )

    def set_valid(self, name: str, requires: str, system: str, note: str = ""):
        """Set status to valid/ready state."""
        self._set_background(BG_OK_SUBTLE)
        self._set_status_dot(STATUS_OK)
        self._set_status_badge("Ready", STATUS_OK, TEXT_MAIN)

        self._name_label.setText(name)
        self._name_label.setStyleSheet(WidgetStyles.label("small", weight="bold", color=TEXT_MAIN))

        self._requires_label.setText(f"Requires: {requires}")
        self._system_label.setText(f"System:   {system}")
        self._details_widget.setVisible(True)

        if note:
            self._note_label.setText(note)
            self._note_label.setVisible(True)
        else:
            self._note_label.setVisible(False)

    def set_warning(self, name: str, requires: str, system: str, issues: list[str]):
        """Set status to warning state with issues list."""
        self._set_background(BG_WARN_SUBTLE)
        self._set_status_dot(STATUS_WARN)
        self._set_status_badge("Issues", STATUS_WARN, TEXT_MAIN)

        self._name_label.setText(name)
        self._name_label.setStyleSheet(WidgetStyles.label("small", weight="bold", color=TEXT_MAIN))

        self._requires_label.setText(f"Requires: {requires}")
        self._system_label.setText(f"System:   {system}")
        self._details_widget.setVisible(True)

        if issues:
            issues_text = "\n".join(f"  - {issue}" for issue in issues)
            self._note_label.setText(f"{len(issues)} issue(s):\n{issues_text}")
            self._note_label.setStyleSheet(WidgetStyles.label("tiny", color=STATUS_WARN))
            self._note_label.setVisible(True)
        else:
            self._note_label.setVisible(False)

    def set_error(self, name: str, error: str):
        """Set status to error state."""
        self._set_background(BG_ERROR_SUBTLE)
        self._set_status_dot(STATUS_ERROR)
        self._set_status_badge("Error", STATUS_ERROR, TEXT_MAIN)

        self._name_label.setText(name)
        self._name_label.setStyleSheet(WidgetStyles.label("small", weight="bold", color=TEXT_MAIN))

        self._requires_label.setText(error)
        self._requires_label.setStyleSheet(WidgetStyles.label("tiny", color=STATUS_ERROR))
        self._system_label.setVisible(False)
        self._note_label.setVisible(False)
        self._details_widget.setVisible(True)

    def set_empty(self, message: str):
        """Set status to empty/no selection state."""
        self._set_background(BG_INPUT)
        self._set_status_dot(TEXT_MUTED)
        self._set_status_badge("", "", "")

        self._name_label.setText(message)
        self._name_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))

        self._requires_label.setText("")
        self._system_label.setText("")
        self._note_label.setVisible(False)
        self._details_widget.setVisible(False)

    def set_info(self, name: str, message: str):
        """Set status to informational state (e.g., linked sync mode without decoder)."""
        self._set_background(BG_INPUT)
        self._set_status_dot(TEXT_LABEL)
        self._set_status_badge("", "", "")

        self._name_label.setText(name)
        self._name_label.setStyleSheet(WidgetStyles.label("small", weight="bold", color=TEXT_LABEL))

        self._requires_label.setText(message)
        self._requires_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_LABEL))
        self._system_label.setVisible(False)
        self._note_label.setVisible(False)
        self._details_widget.setVisible(True)
