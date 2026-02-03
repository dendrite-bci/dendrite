"""
Inline decoder browser widget for selecting decoders from the database.
"""

from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.data.storage.database import Database, DecoderRepository, StudyRepository
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_INPUT,
    BG_PANEL,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

from .decoder_formatters import (
    relative_timestamp,
    type_badge_style,
)


class CompactDecoderRow(QtWidgets.QFrame):
    """Card-style decoder item for inline browser."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, decoder_data: dict[str, Any], parent=None):
        super().__init__(parent)
        self.decoder_data = decoder_data
        self._selected = False
        self._setup_ui()

    def _setup_ui(self):
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Selection checkmark (hidden by default)
        self.check_label = QtWidgets.QLabel("✓")
        self.check_label.setStyleSheet(WidgetStyles.label("small", color=ACCENT))
        self.check_label.setFixedWidth(14)
        self.check_label.setVisible(False)
        layout.addWidget(self.check_label)

        # Decoder name
        name = self.decoder_data.get("decoder_name", "Unknown")
        name_label = QtWidgets.QLabel(name)
        name_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MAIN))
        layout.addWidget(name_label)

        # Model type badge (small)
        model_type = self.decoder_data.get("model_type", "Unknown")
        type_badge = QtWidgets.QLabel(model_type)
        type_badge.setStyleSheet(type_badge_style("tiny"))
        layout.addWidget(type_badge)

        layout.addStretch()

        # Accuracy %
        cv_acc = self.decoder_data.get("cv_mean_accuracy")
        val_acc = self.decoder_data.get("validation_accuracy")
        accuracy = cv_acc if cv_acc is not None else val_acc

        acc_text = f"{accuracy:.0%}" if accuracy is not None else "—"
        acc_label = QtWidgets.QLabel(acc_text)
        acc_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addWidget(acc_label)

        # Relative timestamp
        created = self.decoder_data.get("created_at", "")
        timestamp_text = relative_timestamp(created)
        if timestamp_text:
            time_label = QtWidgets.QLabel(timestamp_text)
            time_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_MUTED))
            time_label.setMinimumWidth(60)
            time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            layout.addWidget(time_label)

        self._update_style()

    def _update_style(self):
        self.check_label.setVisible(self._selected)
        if self._selected:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {BG_INPUT};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background: transparent;
                }}
                QFrame:hover {{
                    background: {BG_PANEL};
                }}
            """)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class InlineDecoderBrowser(QtWidgets.QWidget):
    """Embeddable decoder browser for inline use in tabs.

    Compact vertical layout with inline filters and responsive height.
    """

    decoder_selected = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, default_study_name: str | None = None, db: Database = None):
        super().__init__(parent)
        self._db = db
        self.default_study_name = default_study_name
        self.decoders: list[dict[str, Any]] = []
        self.filtered_decoders: list[dict[str, Any]] = []
        self.selected_decoder: dict[str, Any] | None = None
        self.list_items: list[CompactDecoderRow] = []

        self._setup_ui()
        self._load_decoders()

    def _setup_ui(self):
        """Set up compact vertical layout with inline filters."""
        self.setMinimumHeight(100)
        self.setMaximumHeight(240)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, LAYOUT["spacing_xs"], 0, 0)
        layout.setSpacing(LAYOUT["spacing_xs"])

        # Filter row (single horizontal line)
        filter_row = self._create_filter_row()
        layout.addWidget(filter_row)

        # Scrollable list
        list_scroll = QtWidgets.QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.list_widget = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(2)
        self.list_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        list_scroll.setWidget(self.list_widget)
        layout.addWidget(list_scroll, 1)

    def _create_filter_row(self) -> QtWidgets.QWidget:
        """Create minimal filter row with dropdowns."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_xs"])

        # Study filter
        self.study_filter = QtWidgets.QComboBox()
        self.study_filter.addItem("All Studies", None)
        self.study_filter.setStyleSheet(WidgetStyles.combobox())
        self.study_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.study_filter)

        # Model filter
        self.model_filter = QtWidgets.QComboBox()
        self.model_filter.addItem("All Models", None)
        self.model_filter.setStyleSheet(WidgetStyles.combobox())
        self.model_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.model_filter)

        layout.addStretch()

        # Status count
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_MUTED))
        layout.addWidget(self.status_label)

        # Refresh button
        refresh_btn = QtWidgets.QPushButton("↻")
        refresh_btn.setFixedSize(20, 20)
        refresh_btn.setStyleSheet(WidgetStyles.button(size="small"))
        refresh_btn.setToolTip("Refresh")
        refresh_btn.clicked.connect(self._load_decoders)
        layout.addWidget(refresh_btn)

        return row

    def _load_decoders(self):
        """Load decoders from the database."""
        self.status_label.setText("Loading...")
        self._set_controls_enabled(False)
        QtWidgets.QApplication.processEvents()

        try:
            if self._db is None:
                self._db = Database()
                self._db.init_db()

            repo = DecoderRepository(self._db)
            study_repo = StudyRepository(self._db)

            self.decoders = repo.get_all_decoders()
            studies = study_repo.get_all_studies()

            self._populate_study_filter(studies)
            self._populate_model_filter()
            self._apply_filters()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.decoders = []
            self._update_list([])
        finally:
            self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable filter controls during loading."""
        self.study_filter.setEnabled(enabled)
        self.model_filter.setEnabled(enabled)

    def _populate_study_filter(self, studies: list[dict[str, Any]]):
        """Populate study filter dropdown."""
        self.study_filter.blockSignals(True)
        current = self.study_filter.currentData()

        self.study_filter.clear()
        self.study_filter.addItem("All Studies", None)

        for study in studies:
            name = study.get("study_name", "")
            study_id = study.get("study_id")
            if name and study_id:
                self.study_filter.addItem(name, study_id)

        if self.default_study_name:
            idx = self.study_filter.findText(self.default_study_name)
            if idx >= 0:
                self.study_filter.setCurrentIndex(idx)
        elif current:
            idx = self.study_filter.findData(current)
            if idx >= 0:
                self.study_filter.setCurrentIndex(idx)

        self.study_filter.blockSignals(False)

    def _populate_model_filter(self):
        """Populate model type filter dropdown."""
        self.model_filter.blockSignals(True)
        current = self.model_filter.currentData()

        self.model_filter.clear()
        self.model_filter.addItem("All Models", None)

        model_types = set()
        for decoder in self.decoders:
            mt = decoder.get("model_type")
            if mt:
                model_types.add(mt)

        for mt in sorted(model_types):
            self.model_filter.addItem(mt, mt)

        if current:
            idx = self.model_filter.findData(current)
            if idx >= 0:
                self.model_filter.setCurrentIndex(idx)

        self.model_filter.blockSignals(False)

    def _apply_filters(self):
        """Apply filters and sorting to decoder list."""
        study_id = self.study_filter.currentData()
        model_type = self.model_filter.currentData()

        filtered = []
        for decoder in self.decoders:
            if study_id and decoder.get("study_id") != study_id:
                continue

            if model_type and decoder.get("model_type") != model_type:
                continue

            filtered.append(decoder)

        # Sort by date (newest first)
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        self.filtered_decoders = filtered
        self._update_list(filtered)
        self.status_label.setText(f"Showing {len(filtered)}/{len(self.decoders)}")

    def _has_active_filters(self) -> bool:
        """Check if any filters are currently active."""
        study_filter_active = self.study_filter.currentData() is not None
        model_filter_active = self.model_filter.currentData() is not None
        return study_filter_active or model_filter_active

    def _update_list(self, decoders: list[dict[str, Any]]):
        """Update the list display with filtered decoders."""
        # Store previous selection ID before clearing
        previous_selection_id = None
        if self.selected_decoder:
            previous_selection_id = self.selected_decoder.get("decoder_id")

        for item in self.list_items:
            item.deleteLater()
        self.list_items.clear()

        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.selected_decoder = None

        if not decoders:
            self._show_empty_state()
            return

        for decoder in decoders:
            item = CompactDecoderRow(decoder)
            item.clicked.connect(lambda d=decoder, i=item: self._on_item_clicked(d, i))
            self.list_items.append(item)
            self.list_layout.addWidget(item)

        # Restore previous selection if decoder still in list
        if previous_selection_id is not None:
            self.select_decoder_by_id(previous_selection_id)

    def _show_empty_state(self):
        """Show appropriate empty state message."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, LAYOUT["spacing"], 0, 0)
        layout.setSpacing(LAYOUT["spacing_xs"])
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        if self._has_active_filters():
            title = "No decoders match filters"
            hint = "Try clearing filters"
        elif not self.decoders:
            title = "No decoders in database"
            hint = "Train in Sync mode or use File source"
        else:
            title = "No decoders"
            hint = ""

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addWidget(title_label)

        if hint:
            hint_label = QtWidgets.QLabel(hint)
            hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            hint_label.setStyleSheet(WidgetStyles.label("tiny", color=TEXT_MUTED))
            layout.addWidget(hint_label)

        self.list_layout.addWidget(container)

    def _on_item_clicked(self, decoder: dict[str, Any], item: CompactDecoderRow):
        """Handle list item selection."""
        for i in self.list_items:
            i.set_selected(False)

        item.set_selected(True)
        self.selected_decoder = decoder
        self.decoder_selected.emit(decoder)

    def get_selected_decoder(self) -> dict[str, Any] | None:
        """Get the selected decoder data."""
        return self.selected_decoder

    def get_selected_decoder_path(self) -> str | None:
        """Get the path of the selected decoder."""
        if self.selected_decoder:
            return self.selected_decoder.get("decoder_path")
        return None

    def select_decoder_by_id(self, decoder_id: int) -> bool:
        """Programmatically select a decoder by its database ID.

        Args:
            decoder_id: The database ID of the decoder to select.

        Returns:
            True if decoder was found and selected, False otherwise.
        """
        for item in self.list_items:
            if item.decoder_data.get("decoder_id") == decoder_id:
                self._on_item_clicked(item.decoder_data, item)
                return True
        return False

    def set_default_study(self, study_name: str | None):
        """Update the default study filter."""
        self.default_study_name = study_name
        if study_name:
            idx = self.study_filter.findText(study_name)
            if idx >= 0:
                self.study_filter.setCurrentIndex(idx)

    def refresh(self):
        """Refresh the decoder list from database."""
        self._load_decoders()
