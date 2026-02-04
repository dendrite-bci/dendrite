"""Dataset browser panels for offline streaming."""

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    ACCENT,
    ACCENT_HOVER,
    BG_INPUT,
    BG_MAIN,
    BG_PANEL,
    BORDER,
    BORDER_INPUT,
    FONT_SIZE,
    PADDING,
    RADIUS,
    SPACING,
    TEXT_DISABLED,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import WidgetStyles


class MOAABDiscoveryWorker(QtCore.QThread):
    """Background worker to discover MOABB datasets without blocking the UI."""

    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def run(self):
        try:
            from dendrite.data import discover_moabb_datasets

            configs = discover_moabb_datasets()
            self.finished.emit(configs)
        except Exception as e:
            self.error.emit(str(e))


class MOAABPresetPanel(QtWidgets.QWidget):
    """Sidebar listing available MOABB dataset presets."""

    preset_selected = QtCore.pyqtSignal(str, object)  # (preset_name, DatasetConfig)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self.setup_ui()
        self.apply_styles()
        self._start_discovery()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(SPACING["sm"], 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["sm"])

        # Loading indicator (shown while discovering datasets)
        self._loading_label = QtWidgets.QLabel("Loading MOABB datasets...")
        self._loading_label.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['md']}px;")
        self._loading_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)

        # Preset list (hidden until loaded)
        self.preset_list = QtWidgets.QListWidget()
        self.preset_list.itemDoubleClicked.connect(self._on_item_clicked)
        self.preset_list.hide()
        layout.addWidget(self.preset_list)

        # Help text
        self._help_label = QtWidgets.QLabel("Double-click to add stream")
        self._help_label.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        self._help_label.hide()
        layout.addWidget(self._help_label)

    def _start_discovery(self):
        """Start background thread to discover MOABB datasets."""
        self._worker = MOAABDiscoveryWorker()
        self._worker.finished.connect(self._on_discovery_finished)
        self._worker.error.connect(self._on_discovery_error)
        self._worker.start()

    def _on_discovery_finished(self, configs: list):
        """Populate list when discovery completes."""
        self._loading_label.hide()
        self.preset_list.show()
        self._help_label.show()

        for config in configs:
            item = QtWidgets.QListWidgetItem()
            display_name = config.name.replace("_", " ")
            n_subjects = len(config.subjects) if config.subjects else "?"
            paradigm = config.moabb_paradigm or "Unknown"
            item.setText(f"{display_name}\n{paradigm} | {n_subjects} subjects")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, (config.name, config))
            self.preset_list.addItem(item)

    def _on_discovery_error(self, error_msg: str):
        """Show error message if discovery fails."""
        self._loading_label.setText(f"Failed to load: {error_msg}")

    def apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {BG_PANEL};
            }}
            QListWidget {{
                background: {BG_MAIN};
                border: 1px solid {BORDER_INPUT};
                border-radius: {RADIUS["md"]}px;
                color: {TEXT_MAIN};
                font-size: {FONT_SIZE["md"]}px;
            }}
            QListWidget::item {{
                padding: {PADDING["md"]}px;
                border-bottom: 1px solid {BORDER};
            }}
            QListWidget::item:selected {{
                background: {ACCENT};
                color: {TEXT_MAIN};
            }}
            QListWidget::item:hover {{
                background: {BG_INPUT};
            }}
            QListWidget::item:selected:hover {{
                background: {ACCENT_HOVER};
                color: {TEXT_MAIN};
            }}
        """)

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem):
        preset_name, config = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.preset_selected.emit(preset_name, config)


class InternalDatasetsPanel(QtWidgets.QWidget):
    """Panel listing internal BIDS-exported studies and custom datasets from database."""

    recording_selected = QtCore.pyqtSignal(str)  # fif_path

    # Item type constants
    ITEM_TYPE_CATEGORY = 0
    ITEM_TYPE_FILE = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(SPACING["sm"], 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["sm"])

        # Tree widget for hierarchical display
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.itemDoubleClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

        # Header item (non-selectable)
        font = self.tree.font()
        font.setBold(True)

        self._datasets_header = QtWidgets.QTreeWidgetItem(self.tree, ["Datasets"])
        self._datasets_header.setData(0, QtCore.Qt.ItemDataRole.UserRole, self.ITEM_TYPE_CATEGORY)
        self._datasets_header.setExpanded(True)
        self._datasets_header.setFont(0, font)
        self._datasets_header.setFlags(
            self._datasets_header.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
        )

        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        refresh_btn.setStyleSheet(WidgetStyles.button())
        layout.addWidget(refresh_btn)

        # Help text
        help_label = QtWidgets.QLabel("Double-click a recording to stream")
        help_label.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        layout.addWidget(help_label)

        # Initial load
        self.refresh()

    def apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {BG_PANEL};
            }}
            QTreeWidget {{
                background: {BG_MAIN};
                border: 1px solid {BORDER_INPUT};
                border-radius: {RADIUS["md"]}px;
                color: {TEXT_MAIN};
                font-size: {FONT_SIZE["md"]}px;
            }}
            QTreeWidget::item {{
                padding: {PADDING["sm"]}px;
            }}
            QTreeWidget::item:selected {{
                background: {ACCENT};
                color: {TEXT_MAIN};
            }}
            QTreeWidget::item:hover {{
                background: {BG_INPUT};
            }}
            QTreeWidget::item:selected:hover {{
                background: {ACCENT_HOVER};
                color: {TEXT_MAIN};
            }}
        """)

    def refresh(self):
        """Refresh the tree with datasets from database."""
        # Clear existing items
        self._clear_category(self._datasets_header)

        try:
            from dendrite.data.storage.database import Database, DatasetRepository

            db = Database()
            db.init_db()

            dataset_repo = DatasetRepository(db)
            datasets = dataset_repo.get_all_datasets()
            self._populate_datasets(datasets)

            # Update header with count
            n_datasets = self._datasets_header.childCount()
            self._datasets_header.setText(0, f"Datasets ({n_datasets})")

        except Exception as e:
            error_item = QtWidgets.QTreeWidgetItem(self._datasets_header, [f"Error: {e}"])
            error_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)

    def _clear_category(self, category: QtWidgets.QTreeWidgetItem):
        """Remove all children from a category item."""
        for i in range(category.childCount() - 1, -1, -1):
            category.removeChild(category.child(i))

    def _populate_datasets(self, datasets: list):
        """Populate datasets list."""
        if not datasets:
            return

        for dataset in datasets:
            name = dataset.get("name", "Unknown")
            file_path = dataset.get("file_path", "")
            study_name = dataset.get("study_name")

            if not file_path or not Path(file_path).exists():
                continue

            # Show study name if linked
            if study_name:
                display = f"{name} [{study_name}]"
            else:
                display = name

            item = QtWidgets.QTreeWidgetItem(self._datasets_header, [display])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, self.ITEM_TYPE_FILE)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, file_path)
            item.setToolTip(0, file_path)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        item_type = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if item_type == self.ITEM_TYPE_FILE:
            file_path = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
            if file_path:
                self.recording_selected.emit(file_path)
