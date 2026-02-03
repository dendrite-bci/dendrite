"""Dialog for selecting specific EEG channels for ERP display."""

from functools import partial

from PyQt6 import QtWidgets

from dendrite.gui.styles.design_tokens import BG_INPUT, BG_PANEL, BORDER, STATUS_ERROR
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

REGION_PRESETS: list[tuple[str, str]] = [
    ("Central", "C"),
    ("Frontal", "F"),
    ("Parietal", "P"),
    ("Occipital", "O"),
]


class ChannelSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting specific EEG channels for ERP display."""

    _BAD_CHANNEL_STYLE = f"""
        QCheckBox {{ color: {STATUS_ERROR}; spacing: {LAYOUT["padding_sm"]}px; }}
        QCheckBox::indicator {{
            width: 13px; height: 13px;
            border: 2px solid {STATUS_ERROR};
            background-color: {BG_INPUT};
            border-radius: 2px;
        }}
        QCheckBox::indicator:hover {{ border-color: {STATUS_ERROR}; background-color: {BG_PANEL}; }}
        QCheckBox::indicator:checked {{ background-color: {STATUS_ERROR}; border-color: {STATUS_ERROR}; }}
    """

    def __init__(
        self,
        channel_labels: list[str],
        selected_indices: set[int],
        parent=None,
        bad_channels: set[int] = frozenset(),
    ):
        """Initialize the channel selection dialog.

        Args:
            channel_labels: List of channel names (e.g., ["Fp1", "Fp2", "Cz", ...])
            selected_indices: Set of currently selected channel indices
            parent: Parent widget
            bad_channels: Set of channel indices flagged as BAD by signal quality analysis
        """
        super().__init__(parent)
        self._channel_labels = channel_labels or []
        self._selected_indices = set(selected_indices)
        self._bad_channels = set(bad_channels)
        self._checkboxes: list[QtWidgets.QCheckBox] = []
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Select Channels")
        self.setMinimumWidth(400)
        self.setStyleSheet(WidgetStyles.dialog())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        # Header with quick select buttons
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(LAYOUT["spacing"])

        self._count_label = QtWidgets.QLabel()
        self._count_label.setStyleSheet(WidgetStyles.label(variant="header"))
        header_layout.addWidget(self._count_label)

        header_layout.addStretch()

        for name, prefix in REGION_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(WidgetStyles.button(size="small"))
            matching = self._indices_for_prefix(prefix)
            btn.setEnabled(len(matching) > 0)
            btn.clicked.connect(partial(self._select_region, prefix))
            header_layout.addWidget(btn)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setStyleSheet(f"color: {BORDER};")
        header_layout.addWidget(separator)

        all_btn = QtWidgets.QPushButton("All")
        all_btn.setStyleSheet(WidgetStyles.button(size="small"))
        all_btn.clicked.connect(self._select_all)
        header_layout.addWidget(all_btn)

        none_btn = QtWidgets.QPushButton("None")
        none_btn.setStyleSheet(WidgetStyles.button(size="small"))
        none_btn.clicked.connect(self._select_none)
        header_layout.addWidget(none_btn)

        layout.addLayout(header_layout)

        # Scrollable checkbox grid
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(WidgetStyles.scrollarea)

        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setSpacing(LAYOUT["spacing_sm"])
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Determine columns based on channel count
        n_channels = len(self._channel_labels)
        if n_channels <= 16:
            n_cols = 4
        elif n_channels <= 32:
            n_cols = 5
        elif n_channels <= 64:
            n_cols = 6
        else:
            n_cols = 8

        for idx, label in enumerate(self._channel_labels):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(idx in self._selected_indices)
            if idx in self._bad_channels:
                checkbox.setStyleSheet(self._BAD_CHANNEL_STYLE)
            else:
                checkbox.setStyleSheet(WidgetStyles.checkbox)
            checkbox.stateChanged.connect(self._update_count)
            self._checkboxes.append(checkbox)

            row = idx // n_cols
            col = idx % n_cols
            grid_layout.addWidget(checkbox, row, col)

        scroll_area.setWidget(grid_widget)

        # Set reasonable max height to show multiple rows without excessive scrolling
        max_rows = 10
        row_height = 28
        scroll_area.setMaximumHeight(max_rows * row_height)

        layout.addWidget(scroll_area)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.setStyleSheet(WidgetStyles.dialog_buttonbox)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_count()

    def _update_count(self):
        """Update the selection count label."""
        selected = sum(1 for cb in self._checkboxes if cb.isChecked())
        total = len(self._checkboxes)
        self._count_label.setText(f"Selected: {selected} of {total}")

    def _select_all(self):
        """Select all channels."""
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self._update_count()

    def _select_none(self):
        """Deselect all channels."""
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self._update_count()

    def _indices_for_prefix(self, prefix: str) -> list[int]:
        """Return channel indices whose label starts with the given prefix."""
        upper = prefix.upper()
        return [
            idx for idx, label in enumerate(self._channel_labels) if label.upper().startswith(upper)
        ]

    def _select_region(self, prefix: str) -> None:
        """Replace current selection with channels matching a region prefix."""
        matching = set(self._indices_for_prefix(prefix))
        for idx, checkbox in enumerate(self._checkboxes):
            checkbox.blockSignals(True)
            checkbox.setChecked(idx in matching)
            checkbox.blockSignals(False)
        self._update_count()

    def get_selected_indices(self) -> set[int]:
        """Return the set of selected channel indices."""
        return {idx for idx, cb in enumerate(self._checkboxes) if cb.isChecked()}

    @staticmethod
    def get_selection(
        channel_labels: list[str],
        selected_indices: set[int],
        parent=None,
        bad_channels: set[int] = frozenset(),
    ) -> tuple[bool, set[int]]:
        """Static convenience method to show dialog and get selection.

        Args:
            channel_labels: List of channel names
            selected_indices: Currently selected indices
            parent: Parent widget
            bad_channels: Set of channel indices flagged as BAD

        Returns:
            Tuple of (accepted, selected_indices)
        """
        dialog = ChannelSelectionDialog(channel_labels, selected_indices, parent, bad_channels)
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            return True, dialog.get_selected_indices()
        return False, selected_indices
