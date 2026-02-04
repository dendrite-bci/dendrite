"""Shared form widgets for dataset dialogs."""

import json

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import STATUS_SUCCESS, TEXT_MUTED
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)

# Import from shared preprocessing utilities
from dendrite.processing.preprocessing.utils import get_valid_sample_rates


def create_epoch_group(
    tmin: float = 0.0,
    tmax: float = 0.5,
) -> tuple[QtWidgets.QGroupBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]:
    """Create Epoch Settings group box.

    Args:
        tmin: Initial epoch start time
        tmax: Initial epoch end time

    Returns:
        (group_box, tmin_spin, tmax_spin)
    """
    group = QtWidgets.QGroupBox("Epoch Settings")
    layout = QtWidgets.QFormLayout(group)
    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    tmin_spin = QtWidgets.QDoubleSpinBox()
    tmin_spin.setRange(-5.0, 5.0)
    tmin_spin.setDecimals(2)
    tmin_spin.setValue(tmin)
    tmin_spin.setSuffix(" sec")
    layout.addRow("Epoch Start (tmin):", tmin_spin)

    tmax_spin = QtWidgets.QDoubleSpinBox()
    tmax_spin.setRange(0.1, 10.0)
    tmax_spin.setDecimals(2)
    tmax_spin.setValue(tmax)
    tmax_spin.setSuffix(" sec")
    layout.addRow("Epoch End (tmax):", tmax_spin)

    return group, tmin_spin, tmax_spin


def create_preproc_group(
    lowcut: float = 0.5,
    highcut: float = 50.0,
    rereference: bool = False,
    original_sample_rate: float | None = None,
    target_sample_rate: float | None = None,
) -> tuple[
    QtWidgets.QGroupBox,
    QtWidgets.QDoubleSpinBox,
    QtWidgets.QDoubleSpinBox,
    QtWidgets.QCheckBox,
    QtWidgets.QCheckBox,
    QtWidgets.QComboBox,
]:
    """Create Preprocessing group box.

    Args:
        lowcut: Initial high-pass cutoff
        highcut: Initial low-pass cutoff
        rereference: Initial CAR setting
        original_sample_rate: Source file sample rate (for populating valid rates)
        target_sample_rate: Target sample rate (None = no resampling)

    Returns:
        (group_box, lowcut_spin, highcut_spin, rereference_check, resample_check, resample_combo)
    """
    group = QtWidgets.QGroupBox("Preprocessing")
    layout = QtWidgets.QFormLayout(group)
    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    lowcut_spin = QtWidgets.QDoubleSpinBox()
    lowcut_spin.setRange(0.1, 10.0)
    lowcut_spin.setDecimals(1)
    lowcut_spin.setValue(lowcut)
    lowcut_spin.setSuffix(" Hz")
    layout.addRow("High-pass (lowcut):", lowcut_spin)

    highcut_spin = QtWidgets.QDoubleSpinBox()
    highcut_spin.setRange(10.0, 100.0)
    highcut_spin.setDecimals(1)
    highcut_spin.setValue(highcut)
    highcut_spin.setSuffix(" Hz")
    layout.addRow("Low-pass (highcut):", highcut_spin)

    rereference_check = QtWidgets.QCheckBox("Apply Common Average Reference")
    rereference_check.setChecked(rereference)
    layout.addRow("", rereference_check)

    # Resample row with ComboBox for valid rates
    resample_row = QtWidgets.QHBoxLayout()
    resample_check = QtWidgets.QCheckBox("Resample to:")
    resample_check.setChecked(target_sample_rate is not None)
    resample_row.addWidget(resample_check)

    resample_combo = QtWidgets.QComboBox()
    resample_combo.setMinimumWidth(100)
    resample_combo.setEnabled(target_sample_rate is not None)

    # Populate with valid rates if original rate known
    if original_sample_rate:
        valid_rates = get_valid_sample_rates(original_sample_rate)
        for rate in valid_rates:
            resample_combo.addItem(f"{rate} Hz", rate)
        # Select target rate if specified
        if target_sample_rate:
            idx = resample_combo.findData(int(target_sample_rate))
            if idx >= 0:
                resample_combo.setCurrentIndex(idx)
    else:
        resample_combo.addItem("Load file first", None)

    resample_row.addWidget(resample_combo)
    resample_row.addStretch()

    resample_check.toggled.connect(resample_combo.setEnabled)
    layout.addRow("", resample_row)

    return group, lowcut_spin, highcut_spin, rereference_check, resample_check, resample_combo


def update_resample_combo(
    combo: QtWidgets.QComboBox,
    original_sample_rate: float,
    target_sample_rate: float | None = None,
):
    """Update resample combo with valid rates for given sample rate.

    Args:
        combo: The resample ComboBox to update
        original_sample_rate: Source file sample rate
        target_sample_rate: Rate to select (None = first/highest)
    """
    combo.clear()
    valid_rates = get_valid_sample_rates(original_sample_rate)
    for rate in valid_rates:
        combo.addItem(f"{rate} Hz", rate)

    if target_sample_rate:
        idx = combo.findData(int(target_sample_rate))
        if idx >= 0:
            combo.setCurrentIndex(idx)


class FileInfoWidget(QtWidgets.QWidget):
    """Widget to display file metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_xs"])

        self._path_label = QtWidgets.QLabel()
        self._path_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        self._info_label = QtWidgets.QLabel()
        self._info_label.setStyleSheet(WidgetStyles.label("small", color=STATUS_SUCCESS))
        layout.addWidget(self._info_label)

    def set_info(self, n_channels: int, sample_rate: float, n_samples: int):
        """Update displayed file info."""
        duration = n_samples / sample_rate if sample_rate > 0 else 0
        self._info_label.setText(
            f"Channels: {n_channels} · "
            f"Sample Rate: {sample_rate:.0f} Hz · "
            f"Duration: {duration:.1f} sec"
        )
        self._info_label.setStyleSheet(WidgetStyles.label("small", color=STATUS_SUCCESS))

    def set_path(self, path: str):
        """Set displayed file path."""
        self._path_label.setText(path)

    def clear(self):
        """Clear displayed info."""
        self._path_label.setText("")
        self._info_label.setText("")


def create_file_info_group(
    file_path: str | None = None,
    n_channels: int | None = None,
    sample_rate: float | None = None,
    n_samples: int | None = None,
) -> tuple[QtWidgets.QGroupBox, FileInfoWidget]:
    """Create File Info group box.

    Args:
        file_path: Path to display
        n_channels: Number of channels
        sample_rate: Sample rate in Hz
        n_samples: Number of samples

    Returns:
        (group_box, file_info_widget)
    """
    group = QtWidgets.QGroupBox("File Information")
    layout = QtWidgets.QVBoxLayout(group)

    widget = FileInfoWidget()
    if file_path:
        widget.set_path(file_path)
    if n_channels is not None and sample_rate is not None and n_samples is not None:
        widget.set_info(n_channels, sample_rate, n_samples)

    layout.addWidget(widget)
    return group, widget


class EventsSelectorWidget(QtWidgets.QWidget):
    """Widget for selecting events from a source file."""

    def __init__(self, file_path: str | None = None, events_json: str | None = None, parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self._total_events = 0
        self._existing_events: dict[str, int] = {}
        if events_json:
            try:
                self._existing_events = json.loads(events_json)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse events JSON: {e}")
        self._setup_ui()
        if file_path:
            self._load_events()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_xs"])

        # Table for events
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Use", "Code", "Class Name"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self._table.setColumnWidth(0, 40)
        self._table.setColumnWidth(1, 60)
        self._table.setMinimumHeight(120)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # Button row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)

        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self._on_select_all)
        btn_layout.addWidget(select_all_btn)

        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status label
        self._status_label = QtWidgets.QLabel("Load a file to detect events")
        self._status_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addWidget(self._status_label)

    def load_file(self, file_path: str, preselect_codes: set | None = None):
        """Load events from a file (can be called after construction).

        Args:
            file_path: Path to the data file
            preselect_codes: Optional set of event codes to pre-check
        """
        self._file_path = file_path
        if preselect_codes is not None:
            self._existing_events = {}  # Clear existing, use preselect instead
        self._load_events(preselect_codes)

    def _load_events(self, preselect_codes: set | None = None):
        """Load events from the source file."""
        if not self._file_path:
            self._status_label.setText("No file path available")
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            from dendrite.data.imports import load_file

            loaded = load_file(self._file_path)
            QtWidgets.QApplication.restoreOverrideCursor()

            if not loaded.events:
                self._status_label.setText("No events detected in file")
                return

            self._total_events = len(loaded.events)

            # Get unique event codes
            event_codes = sorted(set(e[1] for e in loaded.events))

            # Reverse event_id mapping: code -> name
            code_to_name = {}
            if loaded.event_id:
                code_to_name = {v: k for k, v in loaded.event_id.items()}

            # Determine which codes to pre-check
            if preselect_codes is not None:
                codes_to_check = preselect_codes
            else:
                codes_to_check = set(self._existing_events.values())

            self._table.setRowCount(len(event_codes))

            for row, code in enumerate(event_codes):
                # Checkbox for "Use"
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(code in codes_to_check)
                checkbox.stateChanged.connect(self._update_status)
                checkbox_widget = QtWidgets.QWidget()
                checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                self._table.setCellWidget(row, 0, checkbox_widget)

                # Event code (read-only)
                code_item = QtWidgets.QTableWidgetItem(str(code))
                code_item.setFlags(code_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, 1, code_item)

                # Class name - use existing name if available, else from file
                existing_name = None
                for name, c in self._existing_events.items():
                    if c == code:
                        existing_name = name
                        break
                event_name = existing_name or code_to_name.get(code, f"class_{code}")
                name_item = QtWidgets.QTableWidgetItem(event_name)
                self._table.setItem(row, 2, name_item)

            self._update_status()

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._status_label.setText(f"Failed to load: {e}")

    def _on_select_all(self):
        """Check all event checkboxes."""
        for row in range(self._table.rowCount()):
            checkbox_widget = self._table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)

    def _on_clear_all(self):
        """Uncheck all event checkboxes."""
        for row in range(self._table.rowCount()):
            checkbox_widget = self._table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)

    def _update_status(self):
        """Update status label with selection count."""
        selected = 0
        total = self._table.rowCount()
        for row in range(total):
            checkbox_widget = self._table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                if checkbox and checkbox.isChecked():
                    selected += 1

        if total > 0:
            self._status_label.setText(
                f"{selected} of {total} events selected ({self._total_events} total occurrences)"
            )
        else:
            self._status_label.setText("No events available")

    def get_events_json(self) -> str | None:
        """Get selected events as JSON string."""
        events = {}
        for row in range(self._table.rowCount()):
            # Check if row is enabled
            checkbox_widget = self._table.cellWidget(row, 0)
            if not checkbox_widget:
                continue
            checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
            if not checkbox or not checkbox.isChecked():
                continue

            # Get code and name
            code_item = self._table.item(row, 1)
            name_item = self._table.item(row, 2)
            if not code_item or not name_item:
                continue

            name = name_item.text().strip()
            if not name:
                continue

            try:
                code = int(code_item.text())
                events[name] = code
            except ValueError:
                continue

        return json.dumps(events) if events else None


def create_events_group(
    file_path: str | None = None, events_json: str | None = None
) -> tuple[QtWidgets.QGroupBox, EventsSelectorWidget]:
    """Create Events group box with selector widget.

    Args:
        file_path: Path to source file for loading available events
        events_json: Existing events as JSON string (for pre-selection)

    Returns:
        (group_box, events_selector)
    """
    group = QtWidgets.QGroupBox("Event Mappings")
    layout = QtWidgets.QVBoxLayout(group)

    selector = EventsSelectorWidget(file_path, events_json)
    layout.addWidget(selector)

    return group, selector
