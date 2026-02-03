"""
Stream Setup Dialog

Dialog for stream discovery, selection, and configuration
in a split-panel layout.
"""

import logging

from PyQt6 import QtCore, QtWidgets

from dendrite.constants import LSL_DISCOVERY_TIMEOUT
from dendrite.data.stream_schemas import StreamMetadata
from dendrite.gui.styles.design_tokens import ACCENT, STATUS_OK, STATUS_WARN, TEXT_MUTED
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.pill_navigation import PillNavigation
from dendrite.gui.widgets.common.toggle_pill import TogglePillWidget

from .components import (
    CHANNEL_TYPES,
    ChannelTableModel,
    clear_layout,
    configure_channel_table,
    evaluate_metadata_issues,
    setup_channel_delegates,
)
from .preflight import discover_all_lsl_streams

# Form field definitions for acquisition hardware metadata
ACQUISITION_INFO_FIELDS = [
    ("manufacturer", "Manufacturer:", "e.g., BrainVision, Neuroscan, OpenBCI"),
    ("model", "Model:", "e.g., actiCHamp, SynAmps2, Cyton"),
    ("serial_number", "Serial Number:", "Device serial number (if available)"),
    ("hardware_config", "Hardware Config:", "e.g., 64-channel active electrodes, wireless"),
    ("software_version", "Software Version:", "Recording software version"),
]

# Issue message formatters for metadata issues display
ISSUE_MESSAGES = {
    "fallback_labels": lambda v: f"Channel labels missing (using {v} placeholders)",
    "channel_metadata_missing": lambda v: f"{v} channels missing metadata (using fallbacks)",
    "acquisition_info_missing": lambda _: "Device info not available",
    "duplicate_markers": lambda v: f"Multiple Markers channels ({v}), extras set to MISC",
    "types_inferred": lambda v: f"{v} channel types inferred from labels (review recommended)",
}


class StreamListItem(QtWidgets.QFrame):
    """Clickable stream item for the left panel with include checkbox."""

    clicked = QtCore.pyqtSignal(str)  # Emits stream uid when row clicked
    inclusion_changed = QtCore.pyqtSignal(str, bool)  # Emits (uid, included)

    def __init__(self, uid: str, stream_data: StreamMetadata, parent=None):
        super().__init__(parent)
        self.uid = uid
        self.stream_data = stream_data
        self.is_viewing = False  # Whether this row is being viewed (highlighted)
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedHeight(56)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._update_style()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 12, 8)
        layout.setSpacing(10)

        self.include_toggle = TogglePillWidget(
            initial_state=False, on_color=STATUS_OK, show_label=False, width=28, height=14
        )
        self.include_toggle.toggled.connect(self._on_toggle_changed)
        layout.addWidget(self.include_toggle)

        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(3)
        info_layout.setContentsMargins(0, 0, 0, 0)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        type_label = QtWidgets.QLabel(self.stream_data.type)
        type_label.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 11px;")
        top_row.addWidget(type_label)

        name = self.stream_data.name
        display_name = name[:20] + "..." if len(name) > 20 else name
        name_label = QtWidgets.QLabel(display_name)
        name_label.setStyleSheet("color: white; font-weight: 500; font-size: 11px;")
        if len(name) > 20:
            name_label.setToolTip(name)
        top_row.addWidget(name_label)

        if self.stream_data.has_metadata_issues:
            warn_label = QtWidgets.QLabel("⚠")
            warn_label.setStyleSheet(f"color: {STATUS_WARN}; font-size: 10px;")
            warn_label.setToolTip("Has metadata issues - click to configure")
            top_row.addWidget(warn_label)

        top_row.addStretch()
        info_layout.addLayout(top_row)

        details = f"{self.stream_data.channel_count}ch · {int(self.stream_data.sample_rate)}Hz"
        details_label = QtWidgets.QLabel(details)
        details_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        info_layout.addWidget(details_label)

        layout.addLayout(info_layout, stretch=1)

    def _on_toggle_changed(self, checked: bool):
        """Handle toggle state change."""
        self.inclusion_changed.emit(self.uid, checked)

    def _update_style(self):
        """Update frame style based on viewing state."""
        if self.is_viewing:
            self.setStyleSheet("""
                QFrame {
                    background: #252525;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: transparent;
                    border-radius: 4px;
                }
                QFrame:hover {
                    background: #1e1e1e;
                }
            """)

    def set_viewing(self, viewing: bool):
        """Set whether this stream is currently being viewed/edited."""
        self.is_viewing = viewing
        self._update_style()

    def is_included(self) -> bool:
        """Check if stream is included."""
        return self.include_toggle.isChecked()

    def set_included(self, included: bool):
        """Set inclusion state."""
        self.include_toggle.setChecked(included)

    def mousePressEvent(self, event):
        # Only emit click if not clicking the toggle
        toggle_rect = self.include_toggle.geometry()
        if not toggle_rect.contains(event.pos()):
            self.clicked.emit(self.uid)
        super().mousePressEvent(event)


class StreamConfigPanel(QtWidgets.QWidget):
    """Right panel for editing stream configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_uid = None
        self.current_stream = None
        self.channel_table_model = None
        self.channel_table_view = None
        self.acquisition_widgets: dict[str, QtWidgets.QLineEdit] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT["spacing_lg"], 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing"])

        self.placeholder = QtWidgets.QLabel("Select a stream to configure")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(WidgetStyles.label("large", color=TEXT_MUTED))
        layout.addWidget(self.placeholder)

        self.config_container = QtWidgets.QWidget()
        self.config_container.hide()
        config_layout = QtWidgets.QVBoxLayout(self.config_container)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(LAYOUT["spacing"])

        self.stream_header = QtWidgets.QLabel()
        self.stream_header.setStyleSheet(WidgetStyles.label("subtitle", weight="bold"))
        config_layout.addWidget(self.stream_header)

        self.issues_container = QtWidgets.QWidget()
        self.issues_layout = QtWidgets.QVBoxLayout(self.issues_container)
        self.issues_layout.setContentsMargins(0, 0, 0, 0)
        self.issues_container.hide()
        config_layout.addWidget(self.issues_container)

        self.pill_nav = PillNavigation(
            tabs=[("basic", "Info"), ("channels", "Channels"), ("device", "Device")], size="medium"
        )
        config_layout.addWidget(self.pill_nav)

        self.stacked = QtWidgets.QStackedWidget()
        self.stacked.addWidget(self._create_basic_tab())
        self.stacked.addWidget(self._create_channels_tab())
        self.stacked.addWidget(self._create_device_tab())
        config_layout.addWidget(self.stacked, stretch=1)

        self.pill_nav.section_changed.connect(
            lambda tab_id: self.stacked.setCurrentIndex(
                {"basic": 0, "channels": 1, "device": 2}[tab_id]
            )
        )

        layout.addWidget(self.config_container, stretch=1)

    def _create_basic_tab(self) -> QtWidgets.QWidget:
        """Create basic info tab (read-only)."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setSpacing(LAYOUT["spacing"])

        self.info_labels = {}
        for field in ["name", "type", "sample_rate", "source_id", "channel_count"]:
            label = QtWidgets.QLabel("—")
            label.setStyleSheet(WidgetStyles.label())
            self.info_labels[field] = label

        layout.addRow("Stream Name:", self.info_labels["name"])
        layout.addRow("Type:", self.info_labels["type"])
        layout.addRow("Sample Rate:", self.info_labels["sample_rate"])
        layout.addRow("Source ID:", self.info_labels["source_id"])
        layout.addRow("Channels:", self.info_labels["channel_count"])

        layout.addItem(
            QtWidgets.QSpacerItem(
                0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
            )
        )
        return tab

    def _create_channels_tab(self) -> QtWidgets.QWidget:
        """Create channel configuration tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(LAYOUT["spacing"])

        bulk_layout = QtWidgets.QHBoxLayout()
        bulk_layout.addWidget(QtWidgets.QLabel("Assign type:"))

        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(CHANNEL_TYPES)
        self.type_combo.setStyleSheet(WidgetStyles.combobox())
        bulk_layout.addWidget(self.type_combo)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setStyleSheet(WidgetStyles.button(size="small"))
        apply_btn.clicked.connect(self._bulk_assign_type)
        bulk_layout.addWidget(apply_btn)
        bulk_layout.addStretch()
        layout.addLayout(bulk_layout)

        self.channel_table_view = QtWidgets.QTableView()
        self.channel_table_view.setStyleSheet(WidgetStyles.tablewidget)
        layout.addWidget(self.channel_table_view, stretch=1)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addWidget(self.summary_label)

        return tab

    def _create_device_tab(self) -> QtWidgets.QWidget:
        """Create device info tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setSpacing(LAYOUT["spacing"])

        note = QtWidgets.QLabel("Optional device information")
        note.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addRow("", note)

        for field_key, label_text, placeholder in ACQUISITION_INFO_FIELDS:
            edit = QtWidgets.QLineEdit()
            edit.setStyleSheet(WidgetStyles.input())
            edit.setPlaceholderText(placeholder)
            layout.addRow(label_text, edit)
            self.acquisition_widgets[field_key] = edit

        layout.addItem(
            QtWidgets.QSpacerItem(
                0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
            )
        )
        return tab

    def load_stream(self, uid: str, stream_data: StreamMetadata):
        """Load a stream for editing."""
        self.current_uid = uid
        self.current_stream = stream_data

        self.placeholder.hide()
        self.config_container.show()

        self.stream_header.setText(stream_data.name)
        self._update_issues_display()

        self.info_labels["name"].setText(stream_data.name or "—")
        self.info_labels["type"].setText(stream_data.type or "—")
        self.info_labels["sample_rate"].setText(
            f"{stream_data.sample_rate} Hz" if stream_data.sample_rate else "—"
        )
        self.info_labels["source_id"].setText(stream_data.source_id or "—")
        self.info_labels["channel_count"].setText(str(stream_data.channel_count))

        self.channel_table_model = ChannelTableModel(uid, stream_data, self)
        self.channel_table_view.setModel(self.channel_table_model)
        configure_channel_table(self.channel_table_view, min_visible_rows=8)
        setup_channel_delegates(self.channel_table_view, parent=self)
        self.channel_table_model.dataChanged.connect(self._update_summary)
        self._update_summary()

        for field_key, widget in self.acquisition_widgets.items():
            value = stream_data.acquisition_info.get(field_key, "")
            widget.setText(value)

    def _update_issues_display(self):
        """Update the issues display."""
        clear_layout(self.issues_layout)

        if not self.current_stream or not self.current_stream.has_metadata_issues:
            self.issues_container.hide()
            return

        self.issues_container.show()
        for issue_key, issue_value in self.current_stream.metadata_issues.items():
            formatter = ISSUE_MESSAGES.get(issue_key)
            text = formatter(issue_value) if formatter else issue_key
            label = QtWidgets.QLabel(text)
            label.setStyleSheet(f"color: {STATUS_WARN}; font-size: 11px;")
            self.issues_layout.addWidget(label)

    def _bulk_assign_type(self):
        """Bulk assign channel type."""
        if not self.channel_table_model:
            return
        channel_type = self.type_combo.currentText()
        selection = self.channel_table_view.selectionModel()
        selected_rows = None
        if selection and selection.hasSelection():
            selected_rows = {index.row() for index in selection.selectedIndexes()}
        self.channel_table_model.bulk_assign_type(channel_type, selected_rows)
        self._update_summary()

    def _update_summary(self):
        """Update channel summary."""
        if not self.channel_table_model:
            return
        types = self.channel_table_model.get_channel_types()
        counts = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        parts = [f"{c} {t}" for t, c in counts.items() if c > 0]
        self.summary_label.setText(", ".join(parts) if parts else "No channels")

    def get_updated_data(self) -> dict:
        """Get updated stream data."""
        if not self.current_stream or not self.channel_table_model:
            return {}

        labels = []
        for i in range(self.current_stream.channel_count):
            index = self.channel_table_model.index(i, 1)
            label = self.channel_table_model.data(index, QtCore.Qt.ItemDataRole.DisplayRole)
            labels.append(label if label else f"Ch{i + 1}")

        types = self.channel_table_model.get_channel_types()
        units = self.channel_table_model.get_channel_units()

        acq_info = {}
        for key, widget in self.acquisition_widgets.items():
            value = widget.text().strip()
            if value:
                acq_info[key] = value

        return {
            "labels": labels,
            "channel_types": types,
            "channel_units": units,
            "acquisition_info": acq_info,
        }


class StreamSetupDialog(QtWidgets.QDialog):
    """
    Dialog for stream discovery and configuration.

    Split-panel layout:
    - Left: Stream list with toggles for inclusion
    - Right: Configuration panel for selected stream

    Runs LSL discovery automatically on open, and supports rescanning.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Discovery")
        self.setMinimumSize(900, 550)
        self.resize(950, 600)
        self.setStyleSheet(WidgetStyles.dialog())

        self.logger = logging.getLogger(__name__)
        self.discovered_streams: dict[str, StreamMetadata] = {}
        self.stream_items: dict[str, StreamListItem] = {}
        self.viewing_uid: str | None = None  # Which stream is being viewed/edited
        self.included_streams: set = set()  # None included by default
        self.pending_changes: dict[str, dict] = {}  # uid -> updated data

        self._setup_ui()
        self._update_global_summary()

        # Run discovery before dialog appears (blocking)
        self._run_discovery()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing"])

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(LAYOUT["spacing_sm"])

        header = QtWidgets.QLabel("Discovered Streams")
        header.setStyleSheet(WidgetStyles.label("subtitle", weight="bold"))
        left_layout.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.stream_list = QtWidgets.QWidget()
        self.stream_list_layout = QtWidgets.QVBoxLayout(self.stream_list)
        self.stream_list_layout.setContentsMargins(0, 0, 0, 0)
        self.stream_list_layout.setSpacing(6)

        self.stream_list_layout.addStretch()
        scroll.setWidget(self.stream_list)
        left_layout.addWidget(scroll, stretch=1)

        self.rescan_btn = QtWidgets.QPushButton("Discovering...")
        self.rescan_btn.setStyleSheet(WidgetStyles.button())
        self.rescan_btn.setEnabled(False)
        self.rescan_btn.clicked.connect(self._run_discovery)
        left_layout.addWidget(self.rescan_btn)

        left_panel.setFixedWidth(320)
        splitter.addWidget(left_panel)

        self.config_panel = StreamConfigPanel()
        splitter.addWidget(self.config_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.global_summary = QtWidgets.QLabel()
        self.global_summary.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        layout.addWidget(self.global_summary)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(WidgetStyles.button())
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QtWidgets.QPushButton("Apply All")
        apply_btn.setStyleSheet(WidgetStyles.button(variant="primary"))
        apply_btn.clicked.connect(self.accept)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    def _on_stream_clicked(self, uid: str):
        """Handle stream item click - view that stream's config."""
        if self.viewing_uid and self.viewing_uid in self.discovered_streams:
            self._save_current_changes()

        self._view_stream(uid)

    def _on_inclusion_changed(self, uid: str, included: bool):
        """Handle checkbox toggle for stream inclusion with type constraints."""
        if included:
            stream_type = self.discovered_streams[uid].type
            if stream_type in ("EEG", "Events"):
                existing_count = sum(
                    1
                    for u in self.included_streams
                    if u in self.discovered_streams
                    and self.discovered_streams[u].type == stream_type
                )
                if existing_count >= 1:
                    self.stream_items[uid].set_included(False)
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Selection Limit",
                        f"Only one {stream_type} stream can be selected.\n"
                        f"Deselect the current {stream_type} stream first.",
                    )
                    return
            self.included_streams.add(uid)
        else:
            self.included_streams.discard(uid)
        self._update_global_summary()

    def _fix_duplicate_markers(self, stream_data: StreamMetadata) -> StreamMetadata:
        """Check for duplicate Markers channels, keep first, reclassify others as MISC."""
        marker_indices = [i for i, t in enumerate(stream_data.channel_types) if t == "Markers"]
        if len(marker_indices) <= 1:
            return stream_data

        self.logger.warning(
            f"Stream {stream_data.name}: {len(marker_indices)} Markers channels found, keeping first"
        )

        new_types = list(stream_data.channel_types)
        for idx in marker_indices[1:]:
            new_types[idx] = "MISC"

        new_issues = dict(stream_data.metadata_issues)
        new_issues["duplicate_markers"] = len(marker_indices)

        return stream_data.model_copy(
            update={
                "channel_types": new_types,
                "has_metadata_issues": True,
                "metadata_issues": new_issues,
            }
        )

    def _view_stream(self, uid: str):
        """View a stream's configuration."""
        for item_uid, item in self.stream_items.items():
            item.set_viewing(item_uid == uid)

        self.viewing_uid = uid

        if uid in self.discovered_streams:
            self.config_panel.load_stream(uid, self.discovered_streams[uid])

    def _save_current_changes(self):
        """Save changes for currently viewed stream."""
        if not self.viewing_uid:
            return
        updated = self.config_panel.get_updated_data()
        if updated:
            self.pending_changes[self.viewing_uid] = updated

    def _run_discovery(self):
        """Run LSL stream discovery (used for initial discovery and rescan)."""
        self._set_discovering_state(True)
        QtWidgets.QApplication.processEvents()

        def log_callback(msg: str):
            self.logger.info(f"Discovery: {msg}")

        try:
            streams = discover_all_lsl_streams(log_callback, timeout=LSL_DISCOVERY_TIMEOUT)
            self.discovered_streams = streams if streams else {}
            self._rebuild_stream_list()

            if not streams:
                # Use parent() since dialog isn't visible yet during __init__
                QtWidgets.QMessageBox.information(
                    self.parent(),
                    "No Streams Found",
                    "No LSL streams were found on the network.\n\n"
                    "Make sure your LSL-enabled devices are running.",
                )
        except Exception as e:
            self.logger.error(f"Discovery failed: {e}", exc_info=True)
            QtWidgets.QMessageBox.warning(self.parent() or self, "Discovery Failed", str(e))
        finally:
            self._set_discovering_state(False)

    def _set_discovering_state(self, discovering: bool):
        """Update UI to show discovery in progress."""
        self.rescan_btn.setEnabled(not discovering)
        self.rescan_btn.setText("Discovering..." if discovering else "Rescan Streams")

    def _rebuild_stream_list(self):
        """Rebuild the stream list after rescan."""
        clear_layout(self.stream_list_layout)
        self.stream_items.clear()
        self.included_streams = set()

        for uid, stream_data in self.discovered_streams.items():
            stream_data = self._fix_duplicate_markers(stream_data)
            self.discovered_streams[uid] = stream_data
            item = StreamListItem(uid, stream_data)
            item.clicked.connect(self._on_stream_clicked)
            item.inclusion_changed.connect(self._on_inclusion_changed)
            self.stream_items[uid] = item
            self.stream_list_layout.addWidget(item)

        self.stream_list_layout.addStretch()

        if self.discovered_streams:
            first_uid = next(iter(self.discovered_streams))
            self._view_stream(first_uid)

        self._update_global_summary()

    def _update_global_summary(self):
        """Update the global summary of all included streams."""
        if not self.included_streams:
            self.global_summary.setText("No streams included")
            return

        total_streams = len(self.discovered_streams)
        included_count = len(self.included_streams)

        type_counts = {}
        for uid in self.included_streams:
            if uid in self.discovered_streams:
                stream = self.discovered_streams[uid]
                for ch_type in stream.channel_types:
                    type_counts[ch_type] = type_counts.get(ch_type, 0) + 1

        stream_text = f"{included_count} of {total_streams} streams"
        channel_parts = [f"{count} {t}" for t, count in type_counts.items() if count > 0]
        channel_text = ", ".join(channel_parts) if channel_parts else "No channels"

        self.global_summary.setText(f"{stream_text} · {channel_text}")

    def get_configured_streams(self) -> dict[str, StreamMetadata]:
        """Get only INCLUDED streams with their updated configurations."""
        self._save_current_changes()

        result = {}
        for uid, stream_data in self.discovered_streams.items():
            if uid not in self.included_streams:
                continue

            if uid in self.pending_changes:
                updated = self.pending_changes[uid]
                stream_dict = stream_data.model_dump()
                stream_dict.update(updated)
                stream_dict["metadata_issues"] = evaluate_metadata_issues(
                    stream_dict, user_reviewed=True
                )
                result[uid] = StreamMetadata(**stream_dict)
            else:
                result[uid] = stream_data

        return result
