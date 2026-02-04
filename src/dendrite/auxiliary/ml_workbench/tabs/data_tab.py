"""Data tab - Dataset selection for training and evaluation."""

import json

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.data import DatasetConfig, SingleFileLoader, StudyItem
from dendrite.auxiliary.ml_workbench.utils import setup_worker_thread
from dendrite.auxiliary.ml_workbench.widgets import DatasetInfoPanel
from dendrite.gui.styles.design_tokens import (
    BG_ELEVATED,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_DISABLED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import PillNavigation
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class _DiscoveryWorker(QtCore.QObject):
    """Worker for discovering datasets in a background thread."""

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        """Discover all dataset sources."""
        try:
            from dendrite.data import discover_moabb_datasets

            moabb_configs = discover_moabb_datasets()

            datasets = []
            try:
                from dendrite.data.storage.database import Database, DatasetRepository

                db = Database()
                db.init_db()
                dataset_repo = DatasetRepository(db)
                datasets = dataset_repo.get_all_datasets()
            except Exception as e:
                logger.warning(f"Could not load datasets: {e}")

            self.finished.emit((moabb_configs, datasets))
        except Exception as e:
            logger.exception("_DiscoveryWorker failed")
            self.error.emit(str(e))
            self.finished.emit(None)


class _DetailsWorker(QtCore.QThread):
    """Worker thread for loading MOABB dataset details in background."""

    finished_with_config = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        """Load dataset details (subjects, events, etc.)."""
        try:
            from dendrite.data import load_moabb_dataset_details

            load_moabb_dataset_details(self.config)
            self.finished_with_config.emit(self.config)
        except Exception as e:
            logger.warning(f"Failed to load dataset details: {e}")
            self.error.emit(str(e))


# Item type for tree widget
ITEM_TYPE_STUDY = 1

_TREE_STYLE = WidgetStyles.tree_widget()


class MOABBTreeWidget(QtWidgets.QTreeWidget):
    """Tree widget for MOABB datasets (flat list, no categories)."""

    dataset_selected = QtCore.pyqtSignal(object)  # StudyItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self._study_items: dict[str, StudyItem] = {}

        self.setHeaderHidden(True)
        self.setIndentation(0)
        self.setStyleSheet(_TREE_STYLE)
        self.itemClicked.connect(self._on_item_clicked)

    def add_study(self, study: StudyItem):
        """Add a MOABB study to the tree."""
        item = QtWidgets.QTreeWidgetItem(self)

        paradigm = study.config.moabb_paradigm or "Unknown"
        n_subjects = len(study.config.subjects)
        text = f"{study.name} [{paradigm}] {n_subjects} subjects"

        item.setText(0, text)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ITEM_TYPE_STUDY)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, study.name)

        cap = study.capability
        if cap == StudyItem.CAPABILITY_FULL:
            item.setForeground(0, QtGui.QColor(STATUS_SUCCESS))
        elif cap == StudyItem.CAPABILITY_EPOCHS:
            item.setForeground(0, QtGui.QColor(STATUS_WARNING_ALT))
        else:
            item.setForeground(0, QtGui.QColor(TEXT_DISABLED))

        self._study_items[study.name] = study

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        name = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
        if name and name in self._study_items:
            self.dataset_selected.emit(self._study_items[name])

    def clear_all(self):
        """Clear all items."""
        self.clear()
        self._study_items.clear()

    def count(self) -> int:
        return self.topLevelItemCount()


class InternalTreeWidget(QtWidgets.QTreeWidget):
    """Tree widget for datasets (flat list)."""

    dataset_selected = QtCore.pyqtSignal(object)  # dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets: dict[str, dict] = {}

        self.setHeaderHidden(True)
        self.setIndentation(0)
        self.setStyleSheet(_TREE_STYLE)
        self.itemClicked.connect(self._on_item_clicked)

    def add_dataset(self, dataset: dict):
        """Add a dataset to the list."""
        name = dataset.get("name", "Unknown")
        study_name = dataset.get("study_name")

        item = QtWidgets.QTreeWidgetItem(self)
        if study_name:
            item.setText(0, f"{name} [{study_name}]")
        else:
            item.setText(0, name)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, name)
        item.setForeground(0, QtGui.QColor(STATUS_SUCCESS))

        self._datasets[name] = dataset

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if name and name in self._datasets:
            self.dataset_selected.emit({"type": "dataset", "data": self._datasets[name]})

    def clear_all(self):
        """Clear all items."""
        self.clear()
        self._datasets.clear()

    def count(self) -> int:
        return self.topLevelItemCount()


class DataTab(QtWidgets.QWidget):
    """Dataset selection tab - select a dataset for use in other tabs."""

    study_changed = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_data = None
        self._discovery_worker = None
        self._discovery_thread = None
        self._setup_ui()
        self._scan_datasets()

    def _setup_ui(self):
        # Main horizontal layout (with margins like other tabs)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        main_layout.setSpacing(LAYOUT["spacing_lg"])

        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(560)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(
            LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"]
        )
        left_layout.setSpacing(LAYOUT["spacing_md"])

        # PillNavigation for MOABB / Internal (highlight matches trainer bg = invisible)
        self._source_nav = PillNavigation(
            tabs=[("moabb", "MOABB"), ("internal", "Internal")], size="medium"
        )
        self._source_nav._highlight.setStyleSheet(
            f"background-color: {BG_ELEVATED}; border-radius: {LAYOUT['radius']}px;"
        )
        self._source_nav.section_changed.connect(self._on_source_changed)
        left_layout.addWidget(self._source_nav)

        # Stacked widget for tree panels
        self._panel_stack = QtWidgets.QStackedWidget()

        self._moabb_tree = MOABBTreeWidget()
        self._moabb_tree.dataset_selected.connect(self._on_dataset_selected)
        self._panel_stack.addWidget(self._moabb_tree)

        self._internal_tree = InternalTreeWidget()
        self._internal_tree.dataset_selected.connect(self._on_dataset_selected)
        self._panel_stack.addWidget(self._internal_tree)

        left_layout.addWidget(self._panel_stack, 1)

        # Status label
        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_DISABLED))
        left_layout.addWidget(self._status_label)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(LAYOUT["spacing_sm"])

        self._add_btn = QtWidgets.QPushButton("+ Add")
        self._add_btn.setStyleSheet(WidgetStyles.button())
        self._add_btn.clicked.connect(self._on_add_dataset)
        btn_layout.addWidget(self._add_btn)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setStyleSheet(WidgetStyles.button(transparent=True))
        self._refresh_btn.clicked.connect(self._scan_datasets)
        btn_layout.addWidget(self._refresh_btn)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        main_layout.addWidget(left_panel)

        self._info_panel = DatasetInfoPanel()
        self._info_panel.selection_changed.connect(self._emit_current_dataset)
        main_layout.addWidget(self._info_panel, stretch=1)

    def _on_source_changed(self, source: str):
        """Handle dataset source tab change."""
        self._panel_stack.setCurrentIndex(0 if source == "moabb" else 1)

    def _scan_datasets(self):
        """Scan and populate the dataset trees asynchronously."""
        self._moabb_tree.clear_all()
        self._internal_tree.clear_all()
        self._info_panel.clear()
        self._selected_data = None
        self._status_label.setText("Discovering datasets...")
        self._refresh_btn.setEnabled(False)
        self._add_btn.setEnabled(False)

        # Launch discovery in background thread
        self._discovery_thread = QtCore.QThread()
        self._discovery_worker = _DiscoveryWorker()
        setup_worker_thread(
            self._discovery_worker,
            self._discovery_thread,
            on_finished=self._on_discovery_finished,
            on_error=lambda e: self._status_label.setText(f"Error: {e}"),
        )
        self._discovery_thread.start()

    def _on_discovery_finished(self, result):
        """Handle discovery completion."""
        self._refresh_btn.setEnabled(True)
        self._add_btn.setEnabled(True)

        # Handle error case (result is None on failure)
        if result is None:
            return

        moabb_configs, datasets = result

        # Add MOABB datasets to MOABB tab
        for config in moabb_configs:
            study = StudyItem(config, is_preset=True, is_moabb=True)
            self._moabb_tree.add_study(study)

        # Add datasets to Internal tab
        for dataset in datasets:
            self._internal_tree.add_dataset(dataset)

        # Update status with counts
        n_moabb = self._moabb_tree.count()
        n_datasets = self._internal_tree.count()
        self._status_label.setText(f"{n_moabb} MOABB · {n_datasets} datasets")

    def _on_dataset_selected(self, data):
        """Handle dataset selection."""
        self._selected_data = data

        if isinstance(data, StudyItem):
            # MOABB study - load details if not already loaded (BIDS datasets are lazy)
            if not data.config.subjects:
                self._status_label.setText(f"Loading {data.name}...")
                self._load_moabb_details_async(data)
                return
            self._show_moabb_study(data)

        elif isinstance(data, dict) and data.get("type") == "dataset":
            # Dataset from database
            dataset = data["data"]
            self._info_panel.show_dataset(dataset)
            name = dataset.get("name", "Unknown")
            study_name = dataset.get("study_name")
            if study_name:
                self._status_label.setText(f"Selected: {name} [{study_name}]")
            else:
                self._status_label.setText(f"Selected: {name}")

        self._emit_current_dataset()

    def _load_moabb_details_async(self, data: StudyItem):
        """Load MOABB dataset details in background thread."""
        self._details_worker = _DetailsWorker(data.config, parent=self)
        self._details_worker.finished_with_config.connect(
            lambda cfg: self._on_moabb_details_loaded(data)
        )
        self._details_worker.error.connect(
            lambda e: self._status_label.setText(f"Failed to load: {e}")
        )
        self._details_worker.start()

    def _on_moabb_details_loaded(self, data: StudyItem):
        """Handle completion of MOABB details loading."""
        self._show_moabb_study(data)
        self._emit_current_dataset()

    def _show_moabb_study(self, data: StudyItem):
        """Display MOABB study info after details are loaded."""
        self._info_panel.show_moabb_study(data)
        if data.capability == StudyItem.CAPABILITY_UNAVAILABLE:
            self._status_label.setText(f"Selected: {data.name} [Unavailable]")
        else:
            self._status_label.setText(f"Selected: {data.name} [MOABB]")

    def _emit_current_dataset(self):
        """Emit current dataset data to other tabs."""
        data = self._selected_data
        if not data:
            return

        if isinstance(data, StudyItem):
            if data.capability == StudyItem.CAPABILITY_UNAVAILABLE:
                return
            study_data = {
                "type": "moabb",
                "config": data.config,
                "loader": data.loader,
                "capability": data.capability,
                "selected_subject": self._info_panel.get_selected_subject(),
            }
            self.study_changed.emit(study_data)

        elif isinstance(data, dict) and data.get("type") == "dataset":
            self._emit_dataset(data["data"])

    def _emit_dataset(self, dataset_info: dict):
        """Emit dataset data from database."""
        name = dataset_info.get("name", "dataset")

        # Parse events from JSON
        events = {}
        if dataset_info.get("events_json"):
            try:
                events = json.loads(dataset_info["events_json"])
            except Exception as e:
                logger.warning(f"Could not parse events JSON for {name}: {e}")

        config = DatasetConfig(
            name=name,
            source_type="dataset",
            events=events,
            epoch_tmin=dataset_info.get("epoch_tmin", -0.2),
            epoch_tmax=dataset_info.get("epoch_tmax", 0.8),
            sample_rate=dataset_info.get("sampling_rate", 250.0),
            subjects=[1],
            preproc_lowcut=self._info_panel.get_preproc_lowcut(),
            preproc_highcut=self._info_panel.get_preproc_highcut(),
            preproc_rereference=self._info_panel.get_preproc_rereference(),
        )
        loader = SingleFileLoader.from_dataset_info(config, dataset_info)

        self.study_changed.emit(
            {
                "type": "dataset",
                "config": config,
                "loader": loader,
                "capability": "full",
                "selected_subject": 1,
            }
        )

    def _on_add_dataset(self):
        """Open dialog to add a custom dataset."""
        from dendrite.auxiliary.ml_workbench.dialogs.dataset_dialog import DatasetDialog

        dialog = DatasetDialog(mode="add", parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._scan_datasets()
