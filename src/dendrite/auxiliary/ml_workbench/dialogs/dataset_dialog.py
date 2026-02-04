"""Unified dialog for adding and editing custom FIF/SET datasets."""

from pathlib import Path
from typing import Literal

from PyQt6 import QtCore, QtWidgets

from dendrite.data.imports import get_file_filter, is_supported_format, load_file
from dendrite.data.storage.database import Database, DatasetRepository, StudyRepository
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.utils.logger_central import get_logger

from ._form_widgets import (
    FileInfoWidget,
    create_epoch_group,
    create_events_group,
    create_file_info_group,
    create_preproc_group,
    update_resample_combo,
)

logger = get_logger(__name__)

PARADIGM_OPTIONS = ["", "Motor Imagery", "P300", "SSVEP", "ERP", "Resting State", "Other"]
MODALITY_OPTIONS = ["EEG", "EMG", "ECoG", "MEG", "fNIRS", "Other"]


class DatasetDialog(QtWidgets.QDialog):
    """Unified dialog for adding or editing a custom dataset.

    Args:
        mode: 'add' for new datasets, 'edit' for existing datasets
        dataset: Existing dataset dict (required for edit mode)
        parent: Parent widget
    """

    def __init__(
        self,
        mode: Literal["add", "edit"] = "add",
        dataset: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._mode = mode
        self._dataset = dataset or {}
        self._file_path: str | None = self._dataset.get("file_path") if mode == "edit" else None
        self._file_info: dict | None = None
        self._setup_ui()

    def _setup_ui(self):
        if self._mode == "add":
            self.setWindowTitle("Add Custom Dataset")
        else:
            self.setWindowTitle(f"Edit Dataset: {self._dataset.get('name', 'Unknown')}")

        self.setMinimumWidth(500)
        self.setStyleSheet(WidgetStyles.dialog())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        if self._mode == "add":
            self._setup_add_file_section(layout)
            self._setup_config_section(layout)
        else:
            self._setup_edit_file_section(layout)
            self._setup_metadata_section(layout)

        # Epoch settings
        epoch_kwargs = {}
        if self._mode == "edit":
            epoch_kwargs["tmin"] = self._dataset.get("epoch_tmin", -0.2)
            epoch_kwargs["tmax"] = self._dataset.get("epoch_tmax", 0.8)
        epoch_group, self._tmin_spin, self._tmax_spin = create_epoch_group(**epoch_kwargs)
        layout.addWidget(epoch_group)

        # Preprocessing settings
        preproc_kwargs = {}
        if self._mode == "edit":
            sample_rate = self._dataset.get("sampling_rate")
            preproc_kwargs = {
                "lowcut": self._dataset.get("preproc_lowcut", 0.5),
                "highcut": self._dataset.get("preproc_highcut", 50.0),
                "rereference": bool(self._dataset.get("preproc_rereference", 0)),
                "original_sample_rate": sample_rate,
                "target_sample_rate": self._dataset.get("target_sample_rate"),
            }
        (
            preproc_group,
            self._lowcut_spin,
            self._highcut_spin,
            self._rereference_check,
            self._resample_check,
            self._resample_combo,
        ) = create_preproc_group(**preproc_kwargs)
        layout.addWidget(preproc_group)

        # Events selector
        events_kwargs = {}
        if self._mode == "edit":
            events_kwargs = {
                "file_path": self._dataset.get("file_path"),
                "events_json": self._dataset.get("events_json"),
            }
        events_group, self._events_selector = create_events_group(**events_kwargs)
        layout.addWidget(events_group)

        layout.addStretch()

        # Dialog buttons
        if self._mode == "add":
            buttons = (
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
        else:
            buttons = (
                QtWidgets.QDialogButtonBox.StandardButton.Save
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
        button_box = QtWidgets.QDialogButtonBox(buttons)
        button_box.setStyleSheet(WidgetStyles.dialog_buttonbox)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # -- Add mode sections --

    def _setup_add_file_section(self, layout: QtWidgets.QVBoxLayout):
        """File selection group for add mode."""
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        file_row = QtWidgets.QHBoxLayout()
        self._file_edit = QtWidgets.QLineEdit()
        self._file_edit.setPlaceholderText("Select a .fif or .set file...")
        self._file_edit.setReadOnly(True)
        file_row.addWidget(self._file_edit, 1)

        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.setStyleSheet(WidgetStyles.button())
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)
        file_layout.addLayout(file_row)

        self._file_info_widget = FileInfoWidget()
        file_layout.addWidget(self._file_info_widget)

        layout.addWidget(file_group)

    def _setup_config_section(self, layout: QtWidgets.QVBoxLayout):
        """Dataset configuration group for add mode (name, description, study, paradigm, modality)."""
        config_group = QtWidgets.QGroupBox("Dataset Configuration")
        config_layout = QtWidgets.QFormLayout(config_group)
        config_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("Enter a unique dataset name")
        config_layout.addRow("Name:", self._name_edit)

        self._desc_edit = QtWidgets.QLineEdit()
        self._desc_edit.setPlaceholderText("Optional description")
        config_layout.addRow("Description:", self._desc_edit)

        self._study_combo = QtWidgets.QComboBox()
        self._study_combo.setEditable(True)
        self._study_combo.setStyleSheet(WidgetStyles.combobox())
        config_layout.addRow("Study:", self._study_combo)

        self._paradigm_combo = QtWidgets.QComboBox()
        self._paradigm_combo.setStyleSheet(WidgetStyles.combobox())
        self._paradigm_combo.addItems(PARADIGM_OPTIONS)
        config_layout.addRow("Paradigm:", self._paradigm_combo)

        self._modality_combo = QtWidgets.QComboBox()
        self._modality_combo.setStyleSheet(WidgetStyles.combobox())
        self._modality_combo.addItems(MODALITY_OPTIONS)
        config_layout.addRow("Modality:", self._modality_combo)

        layout.addWidget(config_group)
        self._populate_studies()

    # -- Edit mode sections --

    def _setup_edit_file_section(self, layout: QtWidgets.QVBoxLayout):
        """Read-only file info group for edit mode."""
        file_path = self._dataset.get("file_path", "")
        sample_rate = self._dataset.get("sampling_rate")
        file_group, self._file_info_widget = create_file_info_group(file_path=file_path)
        if sample_rate:
            self._file_info_widget._info_label.setText(f"Sample Rate: {sample_rate:.0f} Hz")
        layout.addWidget(file_group)

    def _setup_metadata_section(self, layout: QtWidgets.QVBoxLayout):
        """Paradigm/modality metadata group for edit mode."""
        meta_group = QtWidgets.QGroupBox("Metadata")
        meta_layout = QtWidgets.QFormLayout(meta_group)
        meta_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._paradigm_combo = QtWidgets.QComboBox()
        self._paradigm_combo.setStyleSheet(WidgetStyles.combobox())
        self._paradigm_combo.addItems(PARADIGM_OPTIONS)
        current_paradigm = self._dataset.get("paradigm", "")
        if current_paradigm in PARADIGM_OPTIONS:
            self._paradigm_combo.setCurrentText(current_paradigm)
        meta_layout.addRow("Paradigm:", self._paradigm_combo)

        self._modality_combo = QtWidgets.QComboBox()
        self._modality_combo.setStyleSheet(WidgetStyles.combobox())
        self._modality_combo.addItems(MODALITY_OPTIONS)
        current_modality = self._dataset.get("modality", "EEG")
        if current_modality in MODALITY_OPTIONS:
            self._modality_combo.setCurrentText(current_modality)
        meta_layout.addRow("Modality:", self._modality_combo)

        layout.addWidget(meta_group)

    # -- Add mode helpers --

    def _populate_studies(self):
        """Populate study dropdown from database."""
        self._study_combo.clear()
        self._study_combo.addItem("")
        try:
            db = Database()
            db.init_db()
            repo = StudyRepository(db)
            for study in repo.get_all_studies():
                name = study.get("study_name", "")
                if name:
                    self._study_combo.addItem(name)
        except Exception as e:
            logger.warning(f"Could not load studies for dropdown: {e}")

    def _browse_file(self):
        """Open file browser for FIF/SET files."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Dataset File", "", get_file_filter()
        )
        if file_path:
            self._load_file(file_path)

    def _load_file(self, file_path: str):
        """Load file and update UI."""
        if not is_supported_format(file_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Format",
                f"File format is not supported.\nSupported: .fif, .h5, .hdf5",
            )
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            loaded = load_file(file_path)
            QtWidgets.QApplication.restoreOverrideCursor()

            self._file_path = file_path
            self._file_info = {
                "n_samples": loaded.data.shape[0],
                "n_channels": len(loaded.channel_names),
                "sample_rate": loaded.sample_rate,
            }

            self._file_edit.setText(file_path)
            self._file_info_widget.set_info(
                n_channels=self._file_info["n_channels"],
                sample_rate=self._file_info["sample_rate"],
                n_samples=self._file_info["n_samples"],
            )

            self._events_selector.load_file(file_path, preselect_codes=set())
            update_resample_combo(self._resample_combo, self._file_info["sample_rate"])

            if not self._name_edit.text():
                self._name_edit.setText(Path(file_path).stem)

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            logger.exception(f"Failed to load file: {file_path}")
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")

    # -- Accept handler --

    def _on_accept(self):
        """Validate and save/update the dataset."""
        if self._mode == "add":
            self._save_new_dataset()
        else:
            self._update_existing_dataset()

    def _save_new_dataset(self):
        """Validate and save a new dataset."""
        if not self._file_path:
            QtWidgets.QMessageBox.warning(self, "Missing File", "Please select a file first.")
            return

        name = self._name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing Name", "Please enter a dataset name.")
            return

        db = Database()
        db.init_db()
        repo = DatasetRepository(db)

        existing = repo.get_by_name(name)
        if existing:
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A dataset named '{name}' already exists.\nPlease choose a different name.",
            )
            return

        events_json = self._events_selector.get_events_json()

        study_id = None
        study_name = self._study_combo.currentText().strip()
        if study_name:
            db = Database()
            db.init_db()
            study_repo = StudyRepository(db)
            study = study_repo.get_or_create(study_name)
            study_id = study.get("study_id") if study else None

        target_rate = (
            self._resample_combo.currentData() if self._resample_check.isChecked() else None
        )
        paradigm = self._paradigm_combo.currentText().strip() or None
        modality = self._modality_combo.currentText().strip() or None
        try:
            dataset_id = repo.add_dataset(
                name=name,
                file_path=self._file_path,
                study_id=study_id,
                events_json=events_json,
                epoch_tmin=self._tmin_spin.value(),
                epoch_tmax=self._tmax_spin.value(),
                sampling_rate=self._file_info.get("sample_rate") if self._file_info else None,
                target_sample_rate=target_rate,
                preproc_lowcut=self._lowcut_spin.value(),
                preproc_highcut=self._highcut_spin.value(),
                preproc_rereference=self._rereference_check.isChecked(),
                description=self._desc_edit.text().strip() or None,
                paradigm=paradigm,
                modality=modality,
            )

            if dataset_id:
                logger.info(f"Added custom dataset: {name} (id={dataset_id})")
                self.accept()
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Save Error", "Failed to save dataset to database."
                )
        except Exception as e:
            logger.exception("Failed to save dataset")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save dataset:\n{e}")

    def _update_existing_dataset(self):
        """Save changes to an existing dataset."""
        dataset_id = self._dataset.get("dataset_id")
        if not dataset_id:
            QtWidgets.QMessageBox.critical(self, "Error", "Dataset ID not found")
            return

        db = Database()
        db.init_db()
        repo = DatasetRepository(db)

        target_rate = (
            self._resample_combo.currentData() if self._resample_check.isChecked() else None
        )
        paradigm = self._paradigm_combo.currentText().strip() or None
        modality = self._modality_combo.currentText().strip() or None
        success = repo.update_dataset(
            dataset_id,
            epoch_tmin=self._tmin_spin.value(),
            epoch_tmax=self._tmax_spin.value(),
            preproc_lowcut=self._lowcut_spin.value(),
            preproc_highcut=self._highcut_spin.value(),
            preproc_rereference=int(self._rereference_check.isChecked()),
            target_sample_rate=target_rate,
            events_json=self._events_selector.get_events_json(),
            paradigm=paradigm,
            modality=modality,
        )

        if success:
            logger.info(f"Updated dataset {dataset_id}")
            self.accept()
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to update dataset")
