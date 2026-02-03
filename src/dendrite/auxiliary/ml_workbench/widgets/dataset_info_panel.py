"""Dataset info panel widget for displaying dataset details."""

import json
from typing import TYPE_CHECKING

from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.datasets import StudyItem
from dendrite.gui.styles.design_tokens import (
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_DISABLED,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.utils.logger_central import get_logger

if TYPE_CHECKING:
    from dendrite.auxiliary.ml_workbench.datasets.moabb_loader import MOAABLoader

logger = get_logger(__name__)


class DatasetInfoPanel(QtWidgets.QWidget):
    """Panel showing dataset details and configuration."""

    selection_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_data = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"]
        )

        self._placeholder = QtWidgets.QLabel("Select a dataset")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(WidgetStyles.label("small", color=TEXT_DISABLED))
        layout.addWidget(self._placeholder)

        self._info_container = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(self._info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(LAYOUT["spacing_sm"])

        # Dataset name and type
        self._name_label = QtWidgets.QLabel()
        self._name_label.setStyleSheet(WidgetStyles.label("header"))
        info_layout.addWidget(self._name_label)

        self._type_label = QtWidgets.QLabel()
        info_layout.addWidget(self._type_label)

        # Stats container for key-value rows
        self._stats_container = QtWidgets.QWidget()
        self._stats_layout = QtWidgets.QVBoxLayout(self._stats_container)
        self._stats_layout.setContentsMargins(0, 0, 0, 0)
        self._stats_layout.setSpacing(2)
        info_layout.addWidget(self._stats_container)

        # Classes section (header + indented list)
        self._classes_header = QtWidgets.QLabel()
        self._classes_header.setStyleSheet(WidgetStyles.label("small"))
        info_layout.addWidget(self._classes_header)

        self._classes_container = QtWidgets.QWidget()
        self._classes_layout = QtWidgets.QVBoxLayout(self._classes_container)
        self._classes_layout.setContentsMargins(LAYOUT["spacing_md"], 0, 0, 0)  # Indent
        self._classes_layout.setSpacing(2)
        info_layout.addWidget(self._classes_container)

        info_layout.addSpacing(LAYOUT["spacing_md"])

        # Subject selection
        self._subject_row = QtWidgets.QWidget()
        subject_layout = QtWidgets.QHBoxLayout(self._subject_row)
        subject_layout.setContentsMargins(0, 0, 0, 0)
        subject_layout.setSpacing(LAYOUT["spacing_sm"])
        subj_label = QtWidgets.QLabel("Subject:")
        subj_label.setStyleSheet(WidgetStyles.label("small"))
        subject_layout.addWidget(subj_label)
        self._subject_combo = QtWidgets.QComboBox()
        self._subject_combo.setStyleSheet(WidgetStyles.combobox())
        self._subject_combo.currentIndexChanged.connect(self._on_selection_changed)
        subject_layout.addWidget(self._subject_combo, 1)
        info_layout.addWidget(self._subject_row)

        # Session/run info (lazy-loaded when subject selected)
        self._sessions_label = QtWidgets.QLabel()
        self._sessions_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        self._sessions_label.setWordWrap(True)
        info_layout.addWidget(self._sessions_label)

        # Preprocessing (read-only)
        self._preproc_label = QtWidgets.QLabel()
        self._preproc_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        info_layout.addWidget(self._preproc_label)

        # Edit button (custom datasets only)
        self._edit_btn = QtWidgets.QPushButton("Edit...")
        self._edit_btn.setStyleSheet(WidgetStyles.button(transparent=True))
        self._edit_btn.clicked.connect(self._on_edit_clicked)
        self._edit_btn.setVisible(False)
        info_layout.addWidget(self._edit_btn)

        info_layout.addStretch()
        layout.addWidget(self._info_container)
        self._info_container.setVisible(False)

    def _on_selection_changed(self):
        if self._current_data:
            # Update session info for MOABB when subject changes
            if self._current_data.get("type") == "moabb":
                subject_id = self._subject_combo.currentData()
                if subject_id is not None:
                    study = self._current_data.get("study")
                    if study:
                        self._update_session_info(study.loader, subject_id)
                else:
                    self._sessions_label.setText("")
            self.selection_changed.emit()

    def _clear_stats_layout(self):
        """Remove all widgets from the stats layout."""
        while self._stats_layout.count():
            item = self._stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Clear nested layout
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

    def _add_info_row(self, label: str, value: str, tooltip: str = ""):
        """Add a label: value row to the stats container."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(LAYOUT["spacing_sm"])

        lbl = QtWidgets.QLabel(f"{label}:")
        lbl.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        lbl.setMinimumWidth(70)
        row.addWidget(lbl)

        val = QtWidgets.QLabel(value)
        val.setStyleSheet(WidgetStyles.label("small"))  # TEXT_LABEL default
        if tooltip:
            val.setToolTip(tooltip)
        row.addWidget(val, 1)

        self._stats_layout.addLayout(row)

    def _clear_classes_layout(self):
        """Remove all widgets from the classes layout."""
        while self._classes_layout.count():
            item = self._classes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

    def _set_classes(self, events: dict[str, int] | None):
        """Display classes with their event codes in a vertical list."""
        self._clear_classes_layout()

        if not events:
            self._classes_header.setText("")
            return

        self._classes_header.setText(f"Classes ({len(events)}):")

        for name, code in events.items():
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(LAYOUT["spacing_sm"])

            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(WidgetStyles.label("small"))
            row.addWidget(name_lbl)

            row.addStretch()

            code_lbl = QtWidgets.QLabel(str(code))
            code_lbl.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
            row.addWidget(code_lbl)

            self._classes_layout.addLayout(row)

    def _update_session_info(self, loader: "MOAABLoader", subject_id: int):
        """Fetch and display session/run structure for selected subject."""
        try:
            sessions = loader.get_sessions(subject_id)
            if sessions:
                session_info = []
                for sess in sessions[:3]:  # Limit to first 3 for display
                    runs = loader.get_runs(subject_id, sess)
                    session_info.append(f"{sess}: {len(runs)} runs")
                self._sessions_label.setText("Sessions: " + " · ".join(session_info))
            else:
                self._sessions_label.setText("Sessions: —")
        except Exception as e:
            logger.debug(f"Could not load session info: {e}")
            self._sessions_label.setText("Sessions: —")

    def show_moabb_study(self, study: StudyItem):
        """Display MOABB study information."""
        self._current_data = {"type": "moabb", "study": study}
        self._placeholder.setVisible(False)
        self._info_container.setVisible(True)
        self._subject_row.setVisible(True)
        self._edit_btn.setVisible(False)
        self._preproc_label.setText("Preprocessing: 0.5-50 Hz (default)")
        self._sessions_label.setText("")

        config = study.config
        self._name_label.setText(config.name)

        # Show paradigm and class count in type label
        paradigm = config.moabb_paradigm or "Unknown"
        n_classes = len(config.events) if config.events else 0
        cap = study.capability
        if cap == StudyItem.CAPABILITY_FULL:
            cap_color = STATUS_SUCCESS
        elif cap == StudyItem.CAPABILITY_EPOCHS:
            cap_color = STATUS_WARNING_ALT
        else:
            cap_color = TEXT_DISABLED
        self._type_label.setText(
            f"<span style='color:{cap_color}'>● {paradigm}</span> · {n_classes} classes"
        )

        # Stats as key-value rows
        self._clear_stats_layout()
        self._add_info_row("Subjects", str(len(config.subjects)))

        # Channel count
        if config.channels:
            self._add_info_row("Channels", str(len(config.channels)))
        else:
            self._add_info_row("Channels", "all EEG")

        srate = f"{config.sample_rate:.0f} Hz" if config.sample_rate else "—"
        self._add_info_row("Sampling", srate)
        epoch_len = config.epoch_tmax - config.epoch_tmin
        self._add_info_row(
            "Epoch", f"{config.epoch_tmin} to {config.epoch_tmax} sec ({epoch_len:.1f} sec)"
        )

        # Show classes with event codes
        self._set_classes(config.events)

        self._subject_combo.clear()
        self._subject_combo.addItem("All", None)
        for subj in config.subjects:
            self._subject_combo.addItem(f"Subject {subj}", subj)

    def show_dataset(self, dataset: dict):
        """Display dataset information from database."""
        self._current_data = {"type": "dataset", "data": dataset}
        self._placeholder.setVisible(False)
        self._info_container.setVisible(True)
        self._subject_row.setVisible(False)
        self._edit_btn.setVisible(True)
        self._sessions_label.setText("")

        name = dataset.get("name", "Unknown")
        file_path = dataset.get("file_path", "")
        srate = dataset.get("sampling_rate")

        self._name_label.setText(name)

        # Show paradigm and class count in type label
        paradigm = dataset.get("paradigm") or "Unknown"
        events_json = dataset.get("events_json")
        n_classes = 0
        events = {}
        if events_json:
            try:
                events = json.loads(events_json)
                n_classes = len(events)
            except Exception as e:
                logger.debug(f"Could not parse events JSON: {e}")

        self._type_label.setText(
            f"<span style='color:{STATUS_SUCCESS}'>● {paradigm}</span> · {n_classes} classes"
        )

        # Stats as key-value rows
        self._clear_stats_layout()

        modality = dataset.get("modality") or "—"
        n_channels = dataset.get("n_channels")
        if n_channels:
            self._add_info_row("Modality", f"{modality} · {n_channels} ch")
        else:
            self._add_info_row("Modality", modality)

        # Sampling rate with optional resample indicator
        target_rate = dataset.get("target_sample_rate")
        if srate and target_rate and target_rate != srate:
            srate_str = f"{srate:.0f} -> {target_rate:.0f} Hz"
        elif srate:
            srate_str = f"{srate:.0f} Hz"
        else:
            srate_str = "unknown"
        self._add_info_row("Sampling", srate_str)

        epoch_tmin = dataset.get("epoch_tmin", -0.2)
        epoch_tmax = dataset.get("epoch_tmax", 0.8)
        epoch_len = epoch_tmax - epoch_tmin
        self._add_info_row("Epoch", f"{epoch_tmin} to {epoch_tmax} sec ({epoch_len:.1f} sec)")

        # File path with tooltip for full path
        if file_path:
            truncated_path = _truncate_path(file_path)
            self._add_info_row("File", truncated_path, tooltip=file_path)

        # Add description if present
        description = dataset.get("description")
        if description:
            self._add_info_row("Description", description)

        self._set_classes(events)

        lowcut = dataset.get("preproc_lowcut", 0.5)
        highcut = dataset.get("preproc_highcut", 50.0)
        car = dataset.get("preproc_rereference", 0)
        car_str = " + CAR" if car else ""
        self._preproc_label.setText(f"Preprocessing: {lowcut}-{highcut} Hz{car_str}")

    def clear(self):
        self._current_data = None
        self._placeholder.setVisible(True)
        self._info_container.setVisible(False)
        self._edit_btn.setVisible(False)

    def get_selected_subject(self) -> int | None:
        if self._current_data:
            data_type = self._current_data.get("type")
            if data_type == "moabb":
                return self._subject_combo.currentData()
        return None

    def get_preproc_lowcut(self) -> float:
        """Get lowcut from stored data or default."""
        if self._current_data and self._current_data.get("type") == "dataset":
            return self._current_data["data"].get("preproc_lowcut", 0.5)
        return 0.5

    def get_preproc_highcut(self) -> float:
        """Get highcut from stored data or default."""
        if self._current_data and self._current_data.get("type") == "dataset":
            return self._current_data["data"].get("preproc_highcut", 50.0)
        return 50.0

    def get_preproc_rereference(self) -> bool:
        """Get rereference from stored data or default."""
        if self._current_data and self._current_data.get("type") == "dataset":
            return bool(self._current_data["data"].get("preproc_rereference", 0))
        return False

    def _on_edit_clicked(self):
        """Open edit dialog for dataset."""
        if not self._current_data or self._current_data.get("type") != "dataset":
            return

        from dendrite.auxiliary.ml_workbench.dialogs.dataset_dialog import DatasetDialog

        dataset = self._current_data["data"]
        dialog = DatasetDialog(mode="edit", dataset=dataset, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Signal parent to refresh
            self.selection_changed.emit()


def _truncate_path(file_path: str, max_parts: int = 3) -> str:
    """Truncate path to last N components with ellipsis."""
    from pathlib import Path

    parts = Path(file_path).parts
    if len(parts) <= max_parts:
        return file_path
    return ".../" + "/".join(parts[-max_parts:])
