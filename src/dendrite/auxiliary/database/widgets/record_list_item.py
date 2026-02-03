"""
Record List Item Widget

Compact clickable list item for database records (recordings/decoders).
Used in split-panel layout for browsing records.
"""

import os

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    BG_INPUT,
    BG_PANEL,
    STATUS_ERROR,
    STATUS_WARN,
    TEXT_MUTED,
)


class RecordListItem(QtWidgets.QFrame):
    """Compact clickable record item for the left panel list."""

    clicked = QtCore.pyqtSignal(dict, str)  # Emits (record, record_type)

    def __init__(self, record: dict, record_type: str, parent=None):
        super().__init__(parent)
        self.record = record
        self.record_type = record_type
        self.is_selected = False
        self.file_status = self._check_file_status()
        self._setup_ui()

    def _check_file_status(self) -> str:
        """Check if associated files exist."""
        if self.record_type == "Recording":
            h5_path = self.record.get("hdf5_file_path", "")
            if not h5_path or not os.path.exists(h5_path):
                return "missing"
            try:
                if os.path.getsize(h5_path) < 1024:
                    return "corrupted"
            except OSError:
                return "missing"
            return "ok"
        else:  # Decoder
            decoder_path = self.record.get("decoder_path", "")
            if not decoder_path or not os.path.exists(decoder_path):
                return "missing"
            return "ok"

    def _setup_ui(self):
        self.setFixedHeight(48)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._update_style()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        # Info column
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 0, 0, 0)

        # Name row with optional warning icon
        name_row = QtWidgets.QHBoxLayout()
        name_row.setSpacing(4)

        # Get display name
        if self.record_type == "Recording":
            name = self.record.get("recording_name", "Unnamed")
        else:
            name = self.record.get("decoder_name", "Unnamed")

        name_label = QtWidgets.QLabel(name)
        name_label.setStyleSheet("color: white; font-weight: 500; font-size: 11px;")
        name_row.addWidget(name_label)

        # Warning icon if file issue
        if self.file_status != "ok":
            icon = "⚠"
            color = STATUS_ERROR if self.file_status == "missing" else STATUS_WARN
            warn_label = QtWidgets.QLabel(icon)
            warn_label.setStyleSheet(f"color: {color}; font-size: 10px;")
            tooltip = "File missing" if self.file_status == "missing" else "File may be corrupted"
            warn_label.setToolTip(tooltip)
            name_row.addWidget(warn_label)

        name_row.addStretch()
        info_layout.addLayout(name_row)

        # Details row
        if self.record_type == "Recording":
            study = self.record.get("study_name", "No study")
            timestamp = self.record.get("session_timestamp", "")
            # Format: "study · timestamp"
            details = f"{study}"
            if timestamp:
                # Truncate timestamp to just date if it's full datetime
                date_part = timestamp[:10] if len(timestamp) > 10 else timestamp
                details += f" · {date_part}"
        else:
            decoder_type = self.record.get("model_type", "Unknown")
            source = self.record.get("source", "")
            details = decoder_type
            if source:
                details += f" · {source}"

        details_label = QtWidgets.QLabel(details)
        details_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        info_layout.addWidget(details_label)

        layout.addLayout(info_layout, stretch=1)

    def _update_style(self):
        """Update frame style based on selection state."""
        if self.is_selected:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {BG_INPUT};
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background: transparent;
                    border-radius: 4px;
                }}
                QFrame:hover {{
                    background: {BG_PANEL};
                }}
            """)

    def set_selected(self, selected: bool):
        """Set whether this item is currently selected."""
        self.is_selected = selected
        self._update_style()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.record, self.record_type)
        super().mousePressEvent(event)
