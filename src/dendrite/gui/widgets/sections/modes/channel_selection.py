"""
Channel selection widget for mode configuration.

Provides compact channel selection with popover for detailed selection.
"""

from functools import partial

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import BG_INPUT, BG_PANEL, BORDER, STATUS_WARN, TEXT_LABEL
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

EEG_REGION_PRESETS: list[tuple[str, str]] = [
    ("Central", "C"),
    ("Frontal", "F"),
    ("Parietal", "P"),
    ("Occipital", "O"),
]

CHECKBOX_WIDTH = 90
MAX_COLS = 6


class ChannelSelectorPopover(QtWidgets.QDialog):
    """Popover dialog for channel selection."""

    def __init__(
        self,
        stream_data: dict,
        selections: dict,
        stream_per_modality: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.stream_data = stream_data
        self._selections = {k: list(v) for k, v in selections.items()}
        self._stream_per_modality = dict(stream_per_modality)
        self._current_modality = None
        self._current_stream_uid = None
        self.checkboxes = []
        self.channels = []
        self._preset_buttons: list[QtWidgets.QPushButton] = []
        self._current_n_cols = 0

        self.setWindowFlags(
            QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet(
            f"ChannelSelectorPopover {{ "
            f"background: {BG_PANEL}; "
            f"border: 1px solid {BORDER}; "
            f"border-radius: {LAYOUT['radius']}px; "
            f"}}"
        )
        self._setup_ui()
        self._rebuild_stream_combo()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(LAYOUT["spacing"])

        # Row 1: Modality | All | None | stretch | Stream (if multiple)
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(LAYOUT["spacing"])

        # Modality dropdown (most used, first)
        mod_group = QtWidgets.QHBoxLayout()
        mod_group.setSpacing(LAYOUT["spacing_xs"])
        self.mod_lbl = QtWidgets.QLabel("Modality")
        self.mod_lbl.setStyleSheet(WidgetStyles.label("small", weight="bold"))
        self.modality_combo = QtWidgets.QComboBox()
        self.modality_combo.setStyleSheet(WidgetStyles.combobox())
        self.modality_combo.setMinimumWidth(70)
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        mod_group.addWidget(self.mod_lbl)
        mod_group.addWidget(self.modality_combo)
        row1.addLayout(mod_group)

        # Quick select buttons (right after modality)
        all_btn = QtWidgets.QPushButton("All")
        all_btn.setStyleSheet(WidgetStyles.button(size="small", transparent=True))
        all_btn.clicked.connect(self._select_all)
        none_btn = QtWidgets.QPushButton("None")
        none_btn.setStyleSheet(WidgetStyles.button(size="small", transparent=True))
        none_btn.clicked.connect(self._select_none)
        row1.addWidget(all_btn)
        row1.addWidget(none_btn)

        row1.addStretch()

        # Stream dropdown (right side, less prominent)
        stream_group = QtWidgets.QHBoxLayout()
        stream_group.setSpacing(LAYOUT["spacing_xs"])
        self.stream_lbl = QtWidgets.QLabel("Stream")
        self.stream_lbl.setStyleSheet(WidgetStyles.label("small", weight="bold"))
        self.stream_combo = QtWidgets.QComboBox()
        self.stream_combo.setStyleSheet(WidgetStyles.combobox())
        self.stream_combo.setMinimumWidth(140)
        self.stream_combo.currentIndexChanged.connect(self._on_stream_changed)
        stream_group.addWidget(self.stream_lbl)
        stream_group.addWidget(self.stream_combo)
        row1.addLayout(stream_group)

        layout.addLayout(row1)

        # Row 2: Preset buttons (EEG only)
        self._preset_container = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(self._preset_container)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(LAYOUT["spacing_xs"])
        for name, prefix in EEG_REGION_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(WidgetStyles.button(size="small"))
            btn.clicked.connect(partial(self._select_region, prefix))
            preset_layout.addWidget(btn)
            self._preset_buttons.append(btn)
        preset_layout.addStretch()
        self._preset_container.setVisible(False)
        layout.addWidget(self._preset_container)

        # Channel grid (no scroll - expands to fit)
        self.grid_widget = QtWidgets.QWidget()
        self.grid_widget.setStyleSheet(
            f"QWidget {{ background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: {LAYOUT['radius']}px; }}"
            f" QCheckBox {{ border: none; background: transparent; }}"
        )
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_layout.setSpacing(6)

        layout.addWidget(self.grid_widget)

    def _rebuild_stream_combo(self):
        self.stream_combo.blockSignals(True)
        self.stream_combo.clear()

        for uid, info in self.stream_data.items():
            stream = info["stream"]
            self.stream_combo.addItem(f"{stream.name} ({stream.type})", uid)

        if self.stream_combo.count() > 0:
            self.stream_combo.setCurrentIndex(0)

        self.stream_combo.blockSignals(False)

        single_stream = self.stream_combo.count() <= 1
        self.stream_combo.setVisible(not single_stream)
        self.stream_lbl.setVisible(not single_stream)

        if self.stream_combo.count() > 0:
            self._on_stream_changed(0)

    def _on_stream_changed(self, _index: int):
        uid = self.stream_combo.currentData()
        if not uid or uid not in self.stream_data:
            return

        self.modality_combo.blockSignals(True)
        self.modality_combo.clear()

        modalities = self.stream_data[uid].get("modalities", {})
        for mod in sorted(modalities.keys()):
            self.modality_combo.addItem(mod.upper(), mod)

        if self.modality_combo.count() > 0:
            eeg_index = self.modality_combo.findData("eeg")
            self.modality_combo.setCurrentIndex(eeg_index if eeg_index >= 0 else 0)

        self.modality_combo.blockSignals(False)

        single_modality = self.modality_combo.count() <= 1
        self.modality_combo.setVisible(not single_modality)
        self.mod_lbl.setVisible(not single_modality)

        if self.modality_combo.count() > 0:
            self._on_modality_changed(0)

    def _on_modality_changed(self, _index: int):
        self._save_selections()

        uid = self.stream_combo.currentData()
        mod = self.modality_combo.currentData()
        if not uid or not mod:
            return

        self._current_modality = mod
        self._current_stream_uid = uid

        for cb in self.checkboxes:
            cb.setParent(None)
        self.checkboxes.clear()
        self.channels.clear()

        channels = self.stream_data.get(uid, {}).get("modalities", {}).get(mod, [])
        self.channels = channels
        selection = self._selections.get(mod, [])

        for i, ch in enumerate(channels):
            cb = QtWidgets.QCheckBox(ch["label"])
            cb.setStyleSheet(WidgetStyles.checkbox)
            cb.setChecked(i in selection)
            self.checkboxes.append(cb)

        n_cols = min(MAX_COLS, len(channels)) if channels else 1
        self._relayout_grid(n_cols)

        is_eeg = mod == "eeg"
        self._preset_container.setVisible(is_eeg)
        if is_eeg:
            self._update_preset_button_states()

        # Adjust size to fit content
        self.adjustSize()

    def _relayout_grid(self, n_cols: int) -> None:
        for col in range(self._current_n_cols):
            self.grid_layout.setColumnStretch(col, 0)

        for i, cb in enumerate(self.checkboxes):
            self.grid_layout.addWidget(cb, i // n_cols, i % n_cols)

        for col in range(n_cols):
            self.grid_layout.setColumnStretch(col, 1)

        self._current_n_cols = n_cols

    def _indices_for_prefix(self, prefix: str) -> list[int]:
        upper = prefix.upper()
        return [i for i, ch in enumerate(self.channels) if ch["label"].upper().startswith(upper)]

    def _select_region(self, prefix: str) -> None:
        matching = set(self._indices_for_prefix(prefix))
        for i, cb in enumerate(self.checkboxes):
            cb.blockSignals(True)
            cb.setChecked(i in matching)
            cb.blockSignals(False)

    def _update_preset_button_states(self) -> None:
        for btn, (_name, prefix) in zip(self._preset_buttons, EEG_REGION_PRESETS, strict=True):
            btn.setEnabled(len(self._indices_for_prefix(prefix)) > 0)

    def _select_all(self):
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)

    def _select_none(self):
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _save_selections(self):
        if self._current_modality and self.checkboxes:
            indices = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
            if indices:
                self._selections[self._current_modality] = indices
                if self._current_stream_uid:
                    self._stream_per_modality[self._current_modality] = self._current_stream_uid
            else:
                self._selections.pop(self._current_modality, None)
                self._stream_per_modality.pop(self._current_modality, None)

    def get_selections(self) -> dict[str, list[int]]:
        self._save_selections()
        return {k: list(v) for k, v in self._selections.items()}

    def get_stream_per_modality(self) -> dict[str, str]:
        self._save_selections()
        return dict(self._stream_per_modality)


class ChannelSelectionWidget(QtWidgets.QWidget):
    """Compact channel selection with popover for detailed selection."""

    config_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stream_data = {}
        self._selections = {}
        self._stream_per_modality = {}
        self._setup_ui()
        self._update_summary()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing"])

        label = QtWidgets.QLabel("Channels")
        label.setStyleSheet(WidgetStyles.label(weight="bold"))
        layout.addWidget(label)

        # Status label showing stream > modality | count
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        # Edit button (right after status)
        self.edit_button = QtWidgets.QPushButton("Edit")
        self.edit_button.setStyleSheet(WidgetStyles.button(size="small", transparent=True))
        self.edit_button.clicked.connect(self._show_popover)
        layout.addWidget(self.edit_button)

        layout.addStretch()

    def _show_popover(self):
        popover = ChannelSelectorPopover(
            self.stream_data,
            self._selections,
            self._stream_per_modality,
            self,
        )
        pos = self.edit_button.mapToGlobal(self.edit_button.rect().bottomLeft())
        popover.move(pos)
        popover.exec()

        # Get selections from popover
        self._selections = popover.get_selections()
        self._stream_per_modality = popover.get_stream_per_modality()
        self._update_summary()
        self.config_changed.emit()

    def _update_summary(self):
        if not self._selections:
            self.status_label.setText("Not configured")
            self.status_label.setStyleSheet(WidgetStyles.label(color=STATUS_WARN))
            return

        self.status_label.setStyleSheet(WidgetStyles.label(color=TEXT_LABEL))
        modality, indices = next(iter(self._selections.items()))
        stream_name = self._get_stream_name(modality)
        total = self._get_total_channels_for_modality(modality)
        self.status_label.setText(f"{stream_name} > {modality.upper()} | {len(indices)} of {total}")

    def _get_stream_name(self, modality: str) -> str:
        uid = self._stream_per_modality.get(modality)
        if uid and uid in self.stream_data:
            return self.stream_data[uid]["stream"].name
        return "Unknown"

    def _get_total_channels_for_modality(self, modality: str) -> int:
        uid = self._stream_per_modality.get(modality)
        if not uid or uid not in self.stream_data:
            return len(self._selections.get(modality, []))
        channels = self.stream_data[uid].get("modalities", {}).get(modality, [])
        return len(channels)

    def update_content(self, stream_data: dict | None = None, current_config: dict | None = None, **_kwargs):
        """Update widget with stream data and existing config."""
        self.stream_data = stream_data or {}
        self._selections = {k: list(v) for k, v in (current_config or {}).items()}

        # Rebuild stream_per_modality from first available stream
        if self.stream_data and self._selections:
            for uid, info in self.stream_data.items():
                modalities = info.get("modalities", {})
                for mod in self._selections.keys():
                    if mod in modalities:
                        self._stream_per_modality[mod] = uid
                        break

        self._update_summary()

    def get_config(self) -> dict[str, list[int]]:
        """Get channel selections per modality."""
        return {k: list(v) for k, v in self._selections.items()}

    def get_required_modalities(self) -> list[str]:
        """Get modalities that have selections."""
        return list(self._selections.keys())

    def get_stream_sources(self) -> dict[str, str]:
        """Get stream name per modality selection."""
        result = {}
        for modality, uid in self._stream_per_modality.items():
            if uid in self.stream_data:
                result[modality] = self.stream_data[uid]["stream"].name
        return result

    def get_modality_labels(self) -> dict[str, list[str]]:
        """Get channel labels per modality (for decoder validation)."""
        result = {}
        for modality, uid in self._stream_per_modality.items():
            if uid not in self.stream_data:
                continue
            channels = self.stream_data[uid].get("modalities", {}).get(modality, [])
            selected_indices = self._selections.get(modality, [])
            result[modality] = [channels[i]["label"] for i in selected_indices if i < len(channels)]
        return result
