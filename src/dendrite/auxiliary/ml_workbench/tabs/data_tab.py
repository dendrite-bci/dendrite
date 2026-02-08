"""Data tab - Dataset selection for training and evaluation."""

import json

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.data import DatasetConfig, FIFLoader, MOAABLoader
from dendrite.auxiliary.ml_workbench.utils import setup_worker_thread
from dendrite.auxiliary.ml_workbench.widgets import DatasetInfoPanel
from dendrite.gui.styles.design_tokens import (
    BG_ELEVATED,
    STATUS_SUCCESS,
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


class _DatasetTreeWidget(QtWidgets.QTreeWidget):
    """Base tree widget for dataset lists."""

    dataset_selected = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, any] = {}
        self.setHeaderHidden(True)
        self.setIndentation(0)
        self.setStyleSheet(_TREE_STYLE)
        self.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if name and name in self._items:
            self.dataset_selected.emit(self._get_selection_data(name))

    def _get_selection_data(self, name: str):
        """Override to customize emitted data."""
        return self._items[name]

    def clear_all(self):
        self.clear()
        self._items.clear()

    def count(self) -> int:
        return self.topLevelItemCount()


class MOABBTreeWidget(_DatasetTreeWidget):
    """Tree widget for MOABB datasets (flat list, no categories)."""

    def add_study(self, config: DatasetConfig):
        """Add a MOABB dataset to the tree."""
        item = QtWidgets.QTreeWidgetItem(self)

        paradigm = config.moabb_paradigm or "Unknown"
        n_subjects = len(config.subjects)
        text = f"{config.name} [{paradigm}] {n_subjects} subjects"

        item.setText(0, text)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, config.name)
        item.setForeground(0, QtGui.QColor(STATUS_SUCCESS))

        self._items[config.name] = config


class InternalTreeWidget(_DatasetTreeWidget):
    """Tree widget for datasets (flat list)."""

    def add_dataset(self, dataset: dict):
        """Add a dataset to the list."""
        name = dataset.get("name", "Unknown")
        study_name = dataset.get("study_name")

        item = QtWidgets.QTreeWidgetItem(self)
        suffix = f" [{study_name}]" if study_name else ""
        item.setText(0, f"{name}{suffix}")
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, name)
        item.setForeground(0, QtGui.QColor(STATUS_SUCCESS))

        self._items[name] = dataset

    def _get_selection_data(self, name: str):
        return {"type": "dataset", "data": self._items[name]}


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
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        main_layout.setSpacing(LAYOUT["spacing_lg"])

        main_layout.addWidget(self._create_browser_panel())

        self._info_panel = DatasetInfoPanel()
        self._info_panel.selection_changed.connect(self._emit_current_dataset)
        main_layout.addWidget(self._info_panel, stretch=1)

    def _create_browser_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(560)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(
            LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"], LAYOUT["spacing_md"]
        )
        layout.setSpacing(LAYOUT["spacing_md"])

        self._source_nav = PillNavigation(
            tabs=[("moabb", "MOABB"), ("internal", "Internal")], size="medium"
        )
        self._source_nav._highlight.setStyleSheet(
            f"background-color: {BG_ELEVATED}; border-radius: {LAYOUT['radius']}px;"
        )
        self._source_nav.section_changed.connect(self._on_source_changed)
        layout.addWidget(self._source_nav)

        self._panel_stack = QtWidgets.QStackedWidget()

        self._moabb_tree = MOABBTreeWidget()
        self._moabb_tree.dataset_selected.connect(self._on_dataset_selected)
        self._panel_stack.addWidget(self._moabb_tree)

        self._internal_tree = InternalTreeWidget()
        self._internal_tree.dataset_selected.connect(self._on_dataset_selected)
        self._panel_stack.addWidget(self._internal_tree)

        layout.addWidget(self._panel_stack, 1)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_DISABLED))
        layout.addWidget(self._status_label)

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
        layout.addLayout(btn_layout)

        return panel

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
            self._moabb_tree.add_study(config)

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

        if isinstance(data, DatasetConfig):
            # MOABB config - load details if not already loaded (BIDS datasets are lazy)
            if not data.subjects:
                self._status_label.setText(f"Loading {data.name}...")
                self._load_moabb_details_async(data)
                return
            self._show_moabb_config(data)

        elif isinstance(data, dict) and data.get("type") == "dataset":
            # Dataset from database
            dataset = data["data"]
            self._info_panel.show_dataset(dataset)
            name = dataset.get("name", "Unknown")
            study_name = dataset.get("study_name")
            suffix = f" [{study_name}]" if study_name else ""
            self._status_label.setText(f"Selected: {name}{suffix}")

        self._emit_current_dataset()

    def _load_moabb_details_async(self, config: DatasetConfig):
        """Load MOABB dataset details in background thread."""
        self._details_worker = _DetailsWorker(config, parent=self)
        self._details_worker.finished_with_config.connect(
            lambda cfg: self._on_moabb_details_loaded(config)
        )
        self._details_worker.error.connect(
            lambda e: self._status_label.setText(f"Failed to load: {e}")
        )
        self._details_worker.start()

    def _on_moabb_details_loaded(self, config: DatasetConfig):
        """Handle completion of MOABB details loading."""
        self._show_moabb_config(config)
        self._emit_current_dataset()

    def _show_moabb_config(self, config: DatasetConfig):
        """Display MOABB config info after details are loaded."""
        self._info_panel.show_moabb_config(config)
        self._status_label.setText(f"Selected: {config.name} [MOABB]")

    def _emit_current_dataset(self):
        """Emit current dataset data to other tabs."""
        data = self._selected_data
        if not data:
            return

        if isinstance(data, DatasetConfig):
            study_data = {
                "type": "moabb",
                "config": data,
                "loader": MOAABLoader(data),
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
        loader = FIFLoader.from_dataset_info(config, dataset_info)

        self.study_changed.emit(
            {
                "type": "dataset",
                "config": config,
                "loader": loader,
                "selected_subject": 1,
            }
        )

    def _on_add_dataset(self):
        """Open dialog to add a custom dataset."""
        from dendrite.auxiliary.ml_workbench.dialogs.dataset_dialog import DatasetDialog

        dialog = DatasetDialog(mode="add", parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._scan_datasets()
