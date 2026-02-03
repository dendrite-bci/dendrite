"""
Compact Resource Progress Bar Widget

Tiny inline progress bar for compact resource display in telemetry.
"""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_LABEL,
    TEXT_MAIN,
)


class CompactResourceBar(QtWidgets.QWidget):
    """Tiny inline progress bar for compact resource display."""

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self.label_text = label
        self.value = 0.0
        self.maximum = 100.0
        self.medium_threshold = 50.0
        self.high_threshold = 80.0
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label (fixed width for alignment)
        self.label = QtWidgets.QLabel(self.label_text)
        self.label.setStyleSheet(f"color: {TEXT_LABEL}; font-size: 11px; min-width: 28px;")
        layout.addWidget(self.label)

        # Tiny progress bar
        self.bar = QtWidgets.QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(50, 6)
        layout.addWidget(self.bar)

        # Value (right-aligned, fixed width for consistent alignment)
        self.value_label = QtWidgets.QLabel("0%")
        self.value_label.setStyleSheet(
            f"color: {TEXT_MAIN}; font-size: 11px; font-weight: 600; min-width: 48px;"
        )
        self.value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.value_label)

        self._update_style()

    def set_value(self, value: float, display_text: str | None = None):
        """Update bar value and optional custom display text."""
        self.value = value
        pct = min(100, max(0, (value / self.maximum) * 100))
        self.bar.setValue(int(pct))
        self.value_label.setText(display_text or f"{value:.0f}%")
        self._update_style()

    def set_thresholds(self, medium: float, high: float):
        self.medium_threshold = medium
        self.high_threshold = high

    def _update_style(self):
        pct = (self.value / self.maximum) * 100 if self.maximum > 0 else 0
        if pct >= self.high_threshold:
            color = STATUS_ERROR
        elif pct >= self.medium_threshold:
            color = STATUS_WARN
        else:
            color = STATUS_OK

        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(255,255,255,0.1);
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
