#!/usr/bin/env python3
"""
DB Explorer GUI Tool

Split-panel database browser for recordings and decoders.
"""

import logging
import sys

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import BG_MAIN, BG_PANEL, SPACING, TEXT_MUTED
from dendrite.gui.styles.widget_styles import WidgetStyles, apply_app_styles
from dendrite.gui.utils import set_app_icon
from dendrite.gui.widgets.common.pill_navigation import PillNavigation

# Import database functions
try:
    from dendrite.data.storage.database import (
        Database,
        DatasetRepository,
        DecoderRepository,
        RecordingRepository,
        StudyRepository,
    )

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logging.warning("dendrite.database module not available. Database features will be disabled.")

# Import widgets
from dendrite.auxiliary.database.widgets import (
    AddRecordingPanel,
    RecordDetailsPanel,
    RecordListItem,
)

# Simple logging setup for standalone app
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DBExplorer")


class DBExplorer(QtWidgets.QMainWindow):
    """Split-panel database explorer for recordings and decoders."""

    ITEMS_PER_BATCH = 15  # Items to load per batch for lazy loading

    def __init__(self):
        super().__init__()
        self.current_mode = "recordings"  # "recordings" or "studies"
        self.search_term = ""
        self.selected_record: dict | None = None
        self.selected_record_type: str | None = None

        # Record items by UID for selection tracking
        self.record_items: dict[str, RecordListItem] = {}

        # Search timer for debouncing
        self.search_timer = QtCore.QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._refresh_list)

        # Lazy loading state
        self.pending_records = []
        self.lazy_load_timer = QtCore.QTimer()
        self.lazy_load_timer.setSingleShot(True)
        self.lazy_load_timer.timeout.connect(self._load_next_batch)

        # Initialize database if available
        if DB_AVAILABLE:
            try:
                self.db = Database()
                self.db.init_db()
                self.study_repo = StudyRepository(self.db)
                self.repo = RecordingRepository(self.db)
                self.decoder_repo = DecoderRepository(self.db)
                self.dataset_repo = DatasetRepository(self.db)
                logger.info(f"Database initialized at: {self.db.db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                QtWidgets.QMessageBox.critical(
                    None, "Database Error", f"Failed to initialize database: {e}"
                )
                sys.exit(1)
        else:
            QtWidgets.QMessageBox.critical(
                None,
                "Database Error",
                "Database module not available. Please check dendrite.database installation.",
            )
            sys.exit(1)

        self._setup_ui()
        self._apply_styles()
        self._refresh_list()
        logger.info("DB Explorer initialized")

    def _setup_ui(self):
        self.setWindowTitle("Database Explorer")
        self.setGeometry(100, 100, 1000, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("left_panel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        left_layout.setSpacing(SPACING["md"])

        # Mode tabs (Recordings / Studies)
        self.mode_nav = PillNavigation(
            tabs=[("recordings", "Recordings"), ("studies", "Studies")], size="medium"
        )
        self.mode_nav.section_changed.connect(self._on_mode_changed)
        left_layout.addWidget(self.mode_nav)

        # Search input
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setStyleSheet(WidgetStyles.input())
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self._on_search_changed)
        left_layout.addWidget(self.search_input)

        # Stacked widget for list views
        self.list_stack = QtWidgets.QStackedWidget()

        # Recording list (index 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background: {BG_MAIN}; border: none;")

        self.list_widget = QtWidgets.QWidget()
        self.list_widget.setStyleSheet(f"background: {BG_MAIN};")
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(2)
        self.list_layout.addStretch()
        scroll.setWidget(self.list_widget)
        self.list_stack.addWidget(scroll)

        # Studies tree (index 1)
        self.studies_tree = QtWidgets.QTreeWidget()
        self.studies_tree.setHeaderHidden(True)
        self.studies_tree.setIndentation(16)
        self.studies_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.studies_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {BG_MAIN};
                border: none;
            }}
            QTreeWidget::item {{
                padding: 4px;
            }}
            QTreeWidget::item:hover {{
                background: #1e1e1e;
            }}
            QTreeWidget::item:selected {{
                background: #252525;
            }}
        """)
        self.list_stack.addWidget(self.studies_tree)

        left_layout.addWidget(self.list_stack, stretch=1)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        left_layout.addWidget(self.status_label)

        # Add Recording button
        self.add_btn = QtWidgets.QPushButton("+ Add Recording")
        self.add_btn.setStyleSheet(WidgetStyles.button())
        self.add_btn.clicked.connect(self._show_add_panel)
        left_layout.addWidget(self.add_btn)

        left_panel.setFixedWidth(300)
        main_layout.addWidget(left_panel)

        self.right_stack = QtWidgets.QStackedWidget()

        # Details panel (index 0)
        self.details_panel = RecordDetailsPanel()
        self.details_panel.record_deleted.connect(self._on_record_deleted)
        self.right_stack.addWidget(self.details_panel)

        # Add panel (index 1)
        self.add_panel = AddRecordingPanel()
        self.add_panel.recording_added.connect(self._on_recording_added)
        self.add_panel.cancelled.connect(self._show_details_panel)
        self.right_stack.addWidget(self.add_panel)

        main_layout.addWidget(self.right_stack, stretch=1)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {BG_MAIN};
            }}
            #left_panel {{
                background: {BG_PANEL};
            }}
        """)

    def _on_mode_changed(self, mode: str):
        """Handle mode tab change."""
        self.current_mode = mode
        self.selected_record = None
        self.selected_record_type = None
        self.details_panel.clear()
        self._show_details_panel()

        # Update add button visibility (only for recordings)
        self.add_btn.setVisible(mode == "recordings")

        # Switch list view
        self.list_stack.setCurrentIndex(0 if mode == "recordings" else 1)

        self._refresh_list()

    def _on_search_changed(self, text: str):
        """Handle search input change with debouncing."""
        self.search_term = text
        self.search_timer.stop()
        self.search_timer.start(300)

    def _refresh_list(self):
        """Refresh the record list based on current mode and search."""
        # Stop any ongoing lazy loading
        self.lazy_load_timer.stop()

        self.status_label.setText("Loading...")

        if self.current_mode == "studies":
            self._refresh_studies_tree()
            return

        # Clear existing items (recordings mode)
        for i in reversed(range(self.list_layout.count() - 1)):
            widget = self.list_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.record_items.clear()

        try:
            if self.search_term:
                records = self.repo.search_recordings(self.search_term)
            else:
                records = self.repo.get_all_recordings()

            # Setup lazy loading
            self.pending_records = records

            if self.pending_records:
                self._load_next_batch()
            else:
                self.status_label.setText("No recordings found")

            logger.info(f"Started loading {len(records)} recordings")

        except Exception as e:
            self.status_label.setText(f"Error: {e!s}")
            logger.error(f"Error refreshing recordings: {e}")

    def _refresh_studies_tree(self):
        """Refresh the studies tree view."""
        self.studies_tree.clear()

        try:
            studies = self.study_repo.get_all_studies()

            for study in studies:
                study_id = study["study_id"]
                study_name = study["study_name"]

                # Skip if doesn't match search
                if self.search_term and self.search_term.lower() not in study_name.lower():
                    continue

                # Create study item
                study_item = QtWidgets.QTreeWidgetItem(self.studies_tree)
                study_item.setText(0, study_name)
                study_item.setIcon(
                    0, self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
                )
                study_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("study", study))
                study_item.setExpanded(True)

                # Get nested items
                recordings = self.repo.get_recordings_by_study(study_id)
                decoders = self.decoder_repo.get_decoders_by_study(study_id)
                datasets = self.dataset_repo.get_datasets_by_study(study_id)

                # Recordings section
                if recordings:
                    rec_header = QtWidgets.QTreeWidgetItem(study_item)
                    rec_header.setText(0, f"Recordings ({len(recordings)})")
                    rec_header.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("header", None))
                    for rec in recordings:
                        rec_item = QtWidgets.QTreeWidgetItem(rec_header)
                        sub = rec.get("subject_id", "")
                        ses = rec.get("session_id", "")
                        label = f"{rec['recording_name']}"
                        if sub or ses:
                            label += f" (sub-{sub}/ses-{ses})"
                        rec_item.setText(0, label)
                        rec_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("recording", rec))

                # Decoders section
                if decoders:
                    dec_header = QtWidgets.QTreeWidgetItem(study_item)
                    dec_header.setText(0, f"Decoders ({len(decoders)})")
                    dec_header.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("header", None))
                    for dec in decoders:
                        dec_item = QtWidgets.QTreeWidgetItem(dec_header)
                        dec_item.setText(
                            0, f"{dec['decoder_name']} ({dec.get('model_type', 'unknown')})"
                        )
                        dec_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("decoder", dec))

                # Datasets section
                if datasets:
                    ds_header = QtWidgets.QTreeWidgetItem(study_item)
                    ds_header.setText(0, f"Datasets ({len(datasets)})")
                    ds_header.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("header", None))
                    for ds in datasets:
                        ds_item = QtWidgets.QTreeWidgetItem(ds_header)
                        ds_item.setText(0, ds["name"])
                        ds_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("dataset", ds))

            self.status_label.setText(f"{len(studies)} studies")

        except Exception as e:
            self.status_label.setText(f"Error: {e!s}")
            logger.error(f"Error refreshing studies: {e}")

    def _on_tree_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        """Handle tree item click."""
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return

        item_type, record = data

        if item_type == "header":
            return  # Don't select headers

        self.selected_record = record
        self.selected_record_type = (
            item_type.capitalize()
        )  # "Study", "Recording", "Decoder", "Dataset"

        self._show_details_panel()
        self.details_panel.load_record(record, self.selected_record_type)

    def _load_next_batch(self):
        """Load the next batch of items for lazy loading."""
        if not self.pending_records:
            return

        batch_size = min(self.ITEMS_PER_BATCH, len(self.pending_records))
        batch = self.pending_records[:batch_size]
        self.pending_records = self.pending_records[batch_size:]

        record_type = "Recording" if self.current_mode == "recordings" else "Decoder"

        for record in batch:
            # Get unique ID for tracking
            if record_type == "Recording":
                uid = str(record.get("recording_id", id(record)))
            else:
                uid = str(record.get("decoder_id", id(record)))

            item = RecordListItem(record, record_type)
            item.clicked.connect(self._on_record_clicked)
            self.record_items[uid] = item
            self.list_layout.insertWidget(self.list_layout.count() - 1, item)

        QtWidgets.QApplication.processEvents()

        total_loaded = len(self.record_items)
        remaining = len(self.pending_records)

        if remaining > 0:
            self.status_label.setText(f"Loading {total_loaded}/{total_loaded + remaining}...")
            self.lazy_load_timer.start(30)
        else:
            self.status_label.setText(f"{total_loaded} {self.current_mode}")

    def _on_record_clicked(self, record: dict, record_type: str):
        """Handle record item click."""
        # Update selection state
        if record_type == "Recording":
            uid = str(record.get("recording_id", id(record)))
        else:
            uid = str(record.get("decoder_id", id(record)))

        # Deselect previous
        for item_uid, item in self.record_items.items():
            item.set_selected(item_uid == uid)

        self.selected_record = record
        self.selected_record_type = record_type

        # Show details
        self._show_details_panel()
        self.details_panel.load_record(record, record_type)

    def _show_details_panel(self):
        """Show the details panel."""
        self.right_stack.setCurrentIndex(0)

    def _show_add_panel(self):
        """Show the add recording panel."""
        self.add_panel.reset()
        self.right_stack.setCurrentIndex(1)

    def _on_record_deleted(self):
        """Handle record deletion."""
        self.selected_record = None
        self.selected_record_type = None
        self._refresh_list()

    def _on_recording_added(self, recording_data: dict):
        """Handle new recording added."""
        try:
            # Validate required fields
            if not recording_data.get("recording_name"):
                QtWidgets.QMessageBox.warning(
                    self, "Missing Information", "Recording name is required to add a recording."
                )
                return

            # Get or create the study
            study = self.study_repo.get_or_create(recording_data["study_name"])
            study_id = study["study_id"]

            recording_id = self.repo.add_recording(
                study_id=study_id,
                recording_name=recording_data["recording_name"],
                session_timestamp=recording_data["session_timestamp"],
                hdf5_file_path=recording_data["hdf5_file_path"],
                subject_id=recording_data.get("subject_id", ""),
                session_id=recording_data.get("session_id", ""),
            )

            if recording_id:
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Recording added successfully with ID: {recording_id}"
                )
                # Switch to recordings mode and refresh
                self.mode_nav.set_current_tab("recordings")
                self._on_mode_changed("recordings")
                self._show_details_panel()
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Recording may already exist in database."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Database Error", f"Failed to add recording to database:\n{e!s}"
            )
            logger.error(f"Error adding recording: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Database Explorer")

    # Set application icon
    set_app_icon(app, "icons/database.svg")

    # Apply global app styles
    apply_app_styles(app)

    window = DBExplorer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
