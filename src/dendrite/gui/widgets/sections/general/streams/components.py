"""
Stream Configuration Shared Components

Shared UI components, table models, and helpers used across stream configuration dialogs.
Includes:
- ComboBoxDelegate: Table cell combo box editor
- ChannelTableModel: Table model for channel type/unit configuration
- Table configuration helpers
- Metadata validation helpers
"""

from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.data.stream_schemas import StreamMetadata

# Channel type to default unit mapping
CHANNEL_UNIT_MAPPING = {
    "EEG": "µV",
    "EOG": "µV",
    "VEOG": "µV",
    "HEOG": "µV",
    "EMG": "mV",
    "ECG": "mV",
    "AUX": "unknown",
    "Markers": "n/a",
    "Other": "unknown",
}
CHANNEL_TYPES = list(CHANNEL_UNIT_MAPPING.keys())
UNIT_TYPES = ["µV", "mV", "V", "n/a", "unknown", "Hz", "bpm"]
from dendrite.gui.styles.widget_styles import (
    WidgetStyles,
)

# UI Symbols
STATUS_DOT = "●"

# Metadata validation: Keys to filter from user display (technical/internal)
TECHNICAL_ISSUE_KEYS = ["inlet_extraction_failed", "user_reviewed"]

# Metadata validation: Issues that require user attention vs optional
REQUIRED_METADATA_ISSUES = ["fallback_labels", "channel_metadata_missing"]


def show_message_dialog(
    parent: QtWidgets.QWidget,
    title: str,
    message: str,
    icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Icon.Information,
    buttons: QtWidgets.QMessageBox.StandardButton = QtWidgets.QMessageBox.StandardButton.Ok,
) -> int:
    """Show a styled message dialog with consistent appearance."""
    msg_box = QtWidgets.QMessageBox(icon, title, message, buttons, parent)
    msg_box.setStyleSheet(WidgetStyles.messagebox)
    return msg_box.exec()


def clear_layout(layout: QtWidgets.QLayout, preserve_count: int = 0) -> None:
    """Clear all widgets from layout, optionally preserving last N items."""
    while layout.count() > preserve_count:
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


def configure_channel_table(
    table: QtWidgets.QTableView,
    min_visible_rows: int = 8,
    column_resize_modes: list[QtWidgets.QHeaderView.ResizeMode] | None = None,
) -> None:
    """Apply standard configuration to channel assignment tables."""
    # Table appearance
    table.setAlternatingRowColors(False)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setSortingEnabled(False)

    # Header configuration
    h_header = table.horizontalHeader()
    h_header.setDefaultAlignment(
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    )

    # Default column resize modes for channel tables: Index, Label, Type, Unit
    if column_resize_modes is None:
        column_resize_modes = [
            QtWidgets.QHeaderView.ResizeMode.Fixed,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        ]

    for col_idx, mode in enumerate(column_resize_modes):
        h_header.setSectionResizeMode(col_idx, mode)

    # Fixed width for index column
    if column_resize_modes[0] == QtWidgets.QHeaderView.ResizeMode.Fixed:
        h_header.resizeSection(0, 50)
        h_header.setMinimumSectionSize(50)

    # Vertical header
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(32)

    # Set minimum height based on row count
    row_height = table.verticalHeader().defaultSectionSize()
    header_height = table.horizontalHeader().height()
    min_table_height = header_height + (min_visible_rows * row_height) + 20
    table.setMinimumHeight(min_table_height)


def setup_channel_delegates(table: QtWidgets.QTableView, parent: QtWidgets.QWidget) -> None:
    """Set up standard type and unit delegates for channel assignment tables."""
    type_delegate = ComboBoxDelegate(CHANNEL_TYPES, editable=False, parent=parent)
    table.setItemDelegateForColumn(2, type_delegate)

    unit_delegate = ComboBoxDelegate(UNIT_TYPES, editable=True, parent=parent)
    table.setItemDelegateForColumn(3, unit_delegate)


def get_issue_severity(metadata_issues: dict[str, Any]) -> str:
    """Determine the severity of metadata issues based on required vs optional metadata."""
    if not metadata_issues:
        return "ok"

    # Filter out technical tracking flags
    actual_issues = {k: v for k, v in metadata_issues.items() if k not in TECHNICAL_ISSUE_KEYS}
    if not actual_issues:
        return "ok"

    # Check for required metadata issues
    has_required = any(issue in actual_issues for issue in REQUIRED_METADATA_ISSUES)
    user_reviewed = metadata_issues.get("user_reviewed", False)

    if has_required:
        return "warning" if user_reviewed else "critical"

    return "warning"


def evaluate_metadata_issues(
    stream_data: dict[str, Any], user_reviewed: bool = False
) -> dict[str, Any]:
    """Evaluate metadata completeness and return issue dictionary."""
    issues = {}

    # Check for fallback channel labels
    labels = stream_data.get("labels", [])
    if not labels or all(
        label.startswith(("Ch", "EEG", "Event"))
        and label.replace("Ch", "").replace("EEG", "").replace("Event", "").isdigit()
        for label in labels
    ):
        fallback_type = "generic_eeg" if stream_data.get("type") == "EEG" else "generic_events"
        issues["fallback_labels"] = fallback_type

    # Check for missing channel metadata
    channel_types = stream_data.get("channel_types", [])
    channel_units = stream_data.get("channel_units", [])

    if (not channel_types or not any(channel_types)) and (
        not channel_units or not any(channel_units)
    ):
        issues["channel_metadata_missing"] = "types_and_units"

    # Check for missing acquisition info
    acquisition_info = stream_data.get("acquisition_info", {})
    if not acquisition_info or not any(acquisition_info.values()):
        issues["acquisition_info_missing"] = "device_info"

    if user_reviewed:
        issues["user_reviewed"] = True

    return issues


class ComboBoxDelegate(QtWidgets.QStyledItemDelegate):
    """Generic combo box delegate for table cell editing."""

    def __init__(self, items: list[str], editable: bool = False, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.items = items
        self.editable = editable

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtWidgets.QComboBox:
        """Create combo box editor for table cell."""
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(self.items)
        combo.setFrame(False)
        combo.setEditable(self.editable)
        combo.setStyleSheet(WidgetStyles.combobox())
        return combo

    def setEditorData(self, editor: QtWidgets.QComboBox, index: QtCore.QModelIndex) -> None:
        """Populate editor with current cell value."""
        value = index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)
        if value:
            editor.setCurrentText(value)

    def setModelData(
        self,
        editor: QtWidgets.QComboBox,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex,
    ) -> None:
        """Save editor value back to model."""
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        """Position editor to match cell geometry."""
        editor.setGeometry(option.rect)


class ChannelTableModel(QtCore.QAbstractTableModel):
    """Table model for channel assignments with type and unit configuration."""

    def __init__(self, stream_uid: str, stream_data: StreamMetadata, parent=None):
        super().__init__(parent)
        self.stream_uid = stream_uid
        self.stream_data = stream_data

        # Initialize channel assignments
        labels = stream_data.labels or []
        self.saved_channel_types = stream_data.channel_types or []
        self.saved_channel_units = stream_data.channel_units or []

        self.channels = []
        for i in range(stream_data.channel_count):
            label = labels[i] if i < len(labels) else f"Ch{i + 1}"
            channel_type = (
                self.saved_channel_types[i]
                if i < len(self.saved_channel_types)
                else stream_data.type
            )
            channel_unit = (
                self.saved_channel_units[i]
                if i < len(self.saved_channel_units)
                else CHANNEL_UNIT_MAPPING.get(channel_type, "unknown")
            )

            self.channels.append(
                {"index": i + 1, "label": label, "type": channel_type, "unit": channel_unit}
            )

    def rowCount(self, parent=None):
        return len(self.channels)

    def columnCount(self, parent=None):
        return 4  # Index, Label, Type, Unit

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.channels):
            return None

        channel = self.channels[index.row()]
        column = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole or role == QtCore.Qt.ItemDataRole.EditRole:
            if column == 0:
                return str(channel["index"])
            elif column == 1:
                return channel["label"]
            elif column == 2:
                return channel["type"]
            elif column == 3:
                return channel["unit"]
        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if column == 0:
                return QtCore.Qt.AlignmentFlag.AlignCenter

        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.EditRole:
            return False

        if index.column() == 1:  # Label column
            self.channels[index.row()]["label"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        elif index.column() == 2:  # Type column
            self.channels[index.row()]["type"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        elif index.column() == 3:  # Unit column
            self.channels[index.row()]["unit"] = value
            self.dataChanged.emit(index, index, [role])
            return True

        return False

    def flags(self, index):
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if index.column() in (1, 2, 3):  # Label, Type, and Unit columns are editable
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            headers = ["#", "Channel Label", "Type", "Unit"]
            return headers[section]
        return None

    def get_channel_types(self) -> list[str]:
        """Get current channel type assignments."""
        return [ch["type"] for ch in self.channels]

    def get_channel_units(self) -> list[str]:
        """Get current channel unit assignments."""
        return [ch["unit"] for ch in self.channels]

    def set_channel_types(self, types: list[str]):
        """Set all channel types."""
        for i, ch_type in enumerate(types):
            if i < len(self.channels) and ch_type in CHANNEL_TYPES:
                self.channels[i]["type"] = ch_type
        self.dataChanged.emit(self.index(0, 2), self.index(len(self.channels) - 1, 2))

    def set_channel_units(self, units: list[str]):
        """Set all channel units."""
        for i, unit in enumerate(units):
            if i < len(self.channels):
                self.channels[i]["unit"] = unit
        self.dataChanged.emit(self.index(0, 3), self.index(len(self.channels) - 1, 3))

    def apply_saved_configuration(self):
        """Apply saved channel configuration if available."""
        if self.saved_channel_types and len(self.saved_channel_types) == len(self.channels):
            valid_types = []
            for saved_type in self.saved_channel_types:
                if saved_type in CHANNEL_TYPES:
                    valid_types.append(saved_type)
                else:
                    valid_types.append("EEG")  # Fallback for invalid types
            self.set_channel_types(valid_types)
            return True
        return False

    def has_saved_configuration(self) -> bool:
        """Check if saved channel configuration is available and matches this stream."""
        return bool(
            self.saved_channel_types and len(self.saved_channel_types) == len(self.channels)
        )

    def bulk_assign_type(self, channel_type: str, selected_rows: set | None = None) -> None:
        """Bulk assign channel type to selected rows or all channels."""
        if channel_type not in CHANNEL_TYPES:
            return

        if selected_rows is None:
            # Apply to all channels
            types = [channel_type] * len(self.channels)
            self.set_channel_types(types)
        else:
            # Apply to selected rows only
            current_types = self.get_channel_types()
            for row in selected_rows:
                if row < len(current_types):
                    current_types[row] = channel_type
            self.set_channel_types(current_types)

    def get_channel_updates(self) -> dict[str, Any]:
        """Get current channel assignments as a dict for model_copy updates."""
        return {
            "labels": [ch["label"] for ch in self.channels],
            "channel_types": self.get_channel_types(),
            "channel_units": self.get_channel_units(),
        }
