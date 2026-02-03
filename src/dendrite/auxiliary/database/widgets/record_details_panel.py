"""
Record Details Panel Widget

Right panel for viewing record details and performing actions.
Replaces the modal DatabaseRecordDialog.
"""

import os
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.constants import DATABASE_PATH
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_MAIN,
    BG_PANEL,
    SPACING,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

from ..utils import detect_h5_file_type, find_metrics_file, find_raw_eeg_file, show_file_error


class RecordDetailsPanel(QtWidgets.QWidget):
    """Panel for displaying record details with action buttons."""

    record_deleted = QtCore.pyqtSignal()  # Emitted when record is deleted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_record: dict | None = None
        self.current_record_type: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"background: {BG_MAIN};")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT["spacing_lg"], 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing"])

        # Placeholder when no record selected
        self.placeholder = QtWidgets.QLabel("Select a record to view details")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(WidgetStyles.label("large", color=TEXT_MUTED))
        layout.addWidget(self.placeholder)

        # Details container (hidden initially)
        self.details_container = QtWidgets.QWidget()
        self.details_container.hide()
        details_layout = QtWidgets.QVBoxLayout(self.details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(LAYOUT["spacing"])

        self.header_label = QtWidgets.QLabel()
        self.header_label.setStyleSheet(WidgetStyles.label("subtitle", weight="bold"))
        details_layout.addWidget(self.header_label)

        # Scroll area for record fields
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.fields_widget = QtWidgets.QWidget()
        self.fields_layout = QtWidgets.QVBoxLayout(self.fields_widget)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        self.fields_layout.setSpacing(SPACING["sm"])
        scroll.setWidget(self.fields_widget)
        details_layout.addWidget(scroll, stretch=1)

        # Action buttons container - single row at bottom
        self.actions_container = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(self.actions_container)
        actions_layout.setContentsMargins(0, 0, SPACING["md"], SPACING["sm"])
        actions_layout.setSpacing(SPACING["sm"])
        actions_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Session Report button (for recordings - hidden for other types)
        self.session_report_btn = QtWidgets.QPushButton("Session Report")
        self.session_report_btn.setStyleSheet(WidgetStyles.button())
        self.session_report_btn.clicked.connect(self._run_session_report)
        actions_layout.addWidget(self.session_report_btn)

        # BIDS export dropdown
        self.bids_export_btn = QtWidgets.QPushButton("Export BIDS")
        self.bids_export_btn.setStyleSheet(WidgetStyles.button())

        self.bids_menu = QtWidgets.QMenu(self)
        self.bids_menu.addAction("This Recording", self._export_recording_bids)
        self.bids_study_action = self.bids_menu.addAction("Entire Study", self._export_study_bids)
        self.bids_export_btn.setMenu(self.bids_menu)
        actions_layout.addWidget(self.bids_export_btn)

        # Container to group report buttons for show/hide
        self.report_buttons = [self.session_report_btn, self.bids_export_btn]

        actions_layout.addStretch()

        # Delete button (visible for Recording, Decoder, Study)
        self.delete_btn = QtWidgets.QPushButton("✕")
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setStyleSheet(WidgetStyles.button(severity="error", size="small"))
        self.delete_btn.setToolTip("Delete")
        self.delete_btn.clicked.connect(self._delete_record)
        actions_layout.addWidget(self.delete_btn)

        details_layout.addWidget(self.actions_container)
        layout.addWidget(self.details_container, stretch=1)

    def load_record(self, record: dict, record_type: str):
        """Load a record for display."""
        self.current_record = record
        self.current_record_type = record_type

        self.placeholder.hide()
        self.details_container.show()

        # Update header with just the name
        if record_type == "Recording":
            name = record.get("recording_name", "Unnamed")
        elif record_type == "Decoder":
            name = record.get("decoder_name", "Unnamed")
        elif record_type == "Study":
            name = record.get("study_name", "Unnamed")
        elif record_type == "Dataset":
            name = record.get("name", "Unnamed")
        else:
            name = "Unnamed"
        self.header_label.setText(name)

        # Clear existing fields
        while self.fields_layout.count():
            item = self.fields_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Build curated display based on type
        if record_type == "Recording":
            self._build_recording_display(record)
        elif record_type == "Decoder":
            self._build_decoder_display(record)
        elif record_type == "Study":
            self._build_study_display(record)
        elif record_type == "Dataset":
            self._build_dataset_display(record)

        self.fields_layout.addStretch()
        self._update_action_buttons()

    def _build_recording_display(self, record: dict):
        """Build curated display for recordings."""
        self._add_info_row("Study", record.get("study_name") or "—")

        # Subject | Session | Run (combined)
        sub = record.get("subject_id") or "—"
        ses = record.get("session_id") or "—"
        run = record.get("run_number", 1)
        self._add_info_row("", f"Subject: {sub}  |  Session: {ses}  |  Run: {run}")

        # Timestamp (formatted nicely)
        ts = record.get("session_timestamp", "")
        formatted = self._format_timestamp(ts)
        self._add_info_row("Recorded", formatted)

        self._add_spacer()

        # Quick stats from H5 (if file exists)
        h5_path = record.get("hdf5_file_path")
        if h5_path and os.path.exists(h5_path):
            self._add_quick_stats(h5_path)
            self._add_spacer()

        # Files section with status indicators
        self._add_section_header("Files")
        self._add_file_row("EEG data", record.get("hdf5_file_path"))
        self._add_file_row("Metrics", record.get("metrics_file_path"))
        self._add_file_row("Config", record.get("config_file_path"))

    def _build_decoder_display(self, record: dict):
        """Build curated display for decoders."""
        self._add_info_row("Type", record.get("model_type") or "—")

        study = record.get("study_name")
        if study:
            self._add_info_row("Study", study)

        # Classes | Channels
        classes = record.get("num_classes")
        channels = record.get("num_channels")
        if classes or channels:
            parts = []
            if classes:
                parts.append(f"Classes: {classes}")
            if channels:
                parts.append(f"Channels: {channels}")
            self._add_info_row("", "  |  ".join(parts))

        self._add_spacer()

        # Accuracy section with visual bars
        cv_mean = record.get("cv_mean_accuracy")
        cv_std = record.get("cv_std_accuracy")
        cv_folds = record.get("cv_folds")
        train_acc = record.get("training_accuracy")
        val_acc = record.get("validation_accuracy")

        has_accuracy = cv_mean is not None or train_acc is not None or val_acc is not None
        if has_accuracy:
            self._add_section_header("Accuracy")

            if cv_mean is not None:
                suffix = f" ({cv_folds}-fold)" if cv_folds else ""
                self._add_accuracy_bar("CV", cv_mean, cv_std, suffix)

            if train_acc is not None:
                self._add_accuracy_bar("Train", train_acc)

            if val_acc is not None:
                self._add_accuracy_bar("Val", val_acc)

            self._add_spacer()

        # Created timestamp
        created = record.get("created_at", "")
        if created:
            self._add_info_row("Created", self._format_timestamp(created))

        # Decoder path (with copy)
        decoder_path = record.get("decoder_path")
        if decoder_path:
            self._add_file_row("File", decoder_path)

    def _build_study_display(self, record: dict):
        """Build curated display for studies."""
        desc = record.get("description")
        if desc:
            self._add_info_row("Description", desc)

        # Created timestamp
        created = record.get("created_at", "")
        if created:
            self._add_info_row("Created", self._format_timestamp(created))

        # Updated timestamp
        updated = record.get("updated_at", "")
        if updated:
            self._add_info_row("Updated", self._format_timestamp(updated))

    def _build_dataset_display(self, record: dict):
        """Build curated display for datasets."""
        # Study (if linked)
        study = record.get("study_name")
        if study:
            self._add_info_row("Study", study)

        # Sampling rate
        srate = record.get("sampling_rate")
        if srate:
            self._add_info_row("Sampling", f"{srate:.0f} Hz")

        # Epoch window (only if values exist)
        tmin = record.get("epoch_tmin")
        tmax = record.get("epoch_tmax")
        if tmin is not None and tmax is not None:
            self._add_info_row("Epoch", f"{tmin} - {tmax} sec")

        self._add_spacer()

        # File path
        file_path = record.get("file_path")
        if file_path:
            self._add_file_row("File", file_path)

    def _add_info_row(self, label: str, value: str):
        """Add a label: value row."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])

        if label:
            lbl = QtWidgets.QLabel(f"{label}:")
            lbl.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED))
            lbl.setMinimumWidth(80)
            row.addWidget(lbl)

        val = QtWidgets.QLabel(str(value))
        val.setStyleSheet(WidgetStyles.label())
        row.addWidget(val, stretch=1)

        container = QtWidgets.QWidget()
        container.setLayout(row)
        self.fields_layout.addWidget(container)

    def _add_file_row(self, label: str, path: str | None):
        """Add file status row with ✓/— indicator and copy button."""
        exists = path and os.path.exists(str(path))
        status = "✓" if exists else "—"

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, SPACING["md"], 0)
        row.setSpacing(SPACING["sm"])

        lbl = QtWidgets.QLabel(f"  {label}")
        lbl.setStyleSheet(WidgetStyles.label())
        lbl.setMinimumWidth(80)
        row.addWidget(lbl)

        status_lbl = QtWidgets.QLabel(status)
        status_color = "#4ade80" if exists else TEXT_MUTED
        status_lbl.setStyleSheet(WidgetStyles.label(color=status_color))
        row.addWidget(status_lbl)
        row.addStretch()

        if exists:
            copy_btn = QtWidgets.QPushButton("📋")
            copy_btn.setFixedSize(24, 24)
            copy_btn.setToolTip(str(path))
            copy_btn.setStyleSheet(WidgetStyles.button(variant="secondary", size="small"))
            copy_btn.clicked.connect(lambda checked, p=str(path): self._copy_to_clipboard(p))
            row.addWidget(copy_btn)

        container = QtWidgets.QWidget()
        container.setLayout(row)
        self.fields_layout.addWidget(container)

    def _add_section_header(self, text: str):
        """Add a section header."""
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED, weight="bold"))
        self.fields_layout.addWidget(lbl)

    def _add_spacer(self):
        """Add vertical spacing."""
        spacer = QtWidgets.QWidget()
        spacer.setFixedHeight(SPACING["sm"])
        self.fields_layout.addWidget(spacer)

    def _format_timestamp(self, ts: str) -> str:
        """Format timestamp for display."""
        if not ts:
            return "—"
        try:
            from datetime import datetime

            # Try common formats
            for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(str(ts).split(".")[0], fmt)
                    return dt.strftime("%b %d, %Y %I:%M %p")
                except ValueError:
                    continue
            return str(ts)
        except (ValueError, TypeError):
            return str(ts)

    def _add_quick_stats(self, h5_path: str):
        """Add quick stats row from H5 metadata."""
        try:
            stats = self._extract_h5_stats(h5_path)
        except (OSError, ValueError, KeyError):
            return  # Silently skip if file unreadable

        if not stats:
            return

        # Stats row container with background
        stats_row = QtWidgets.QWidget()
        stats_row.setStyleSheet(f"background: {BG_PANEL}; border-radius: 6px;")
        row_layout = QtWidgets.QHBoxLayout(stats_row)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(SPACING["md"])

        # Duration | Channels | Sample Rate
        if "duration" in stats:
            self._add_stat_item(row_layout, stats["duration"], "duration")
        if "channels" in stats:
            self._add_stat_item(row_layout, f"{stats['channels']} ch", "channels")
        if "sample_rate" in stats:
            self._add_stat_item(row_layout, f"{stats['sample_rate']} Hz", "rate")

        row_layout.addStretch()
        self.fields_layout.addWidget(stats_row)

        # Events | Size (secondary row)
        events = stats.get("event_count")
        size = stats.get("file_size_mb")
        if events or size:
            parts = []
            if events:
                parts.append(f"Events: {events}")
            if size:
                parts.append(f"Size: {size:.1f} MB")
            self._add_info_row("", "  |  ".join(parts))

    def _extract_h5_stats(self, h5_path: str) -> dict[str, Any]:
        """Extract quick stats from H5 file attributes."""
        import h5py

        stats = {}

        stats["file_size_mb"] = os.path.getsize(h5_path) / (1024 * 1024)

        with h5py.File(h5_path, "r") as f:
            # Find main data dataset
            for ds_name in ["EEG", "EMG"]:
                if ds_name in f:
                    ds = f[ds_name]

                    # Channels from dtype field count (excluding timestamps)
                    if ds.dtype.names:
                        # Count fields that aren't timestamps
                        ch_count = len([n for n in ds.dtype.names if "timestamp" not in n.lower()])
                        if ch_count > 0:
                            stats["channels"] = ch_count

                    # Sample rate from attributes
                    for attr in ["sampling_frequency", "fs", "sample_rate", "srate"]:
                        if attr in ds.attrs:
                            stats["sample_rate"] = int(ds.attrs[attr])
                            break

                    # Duration from timestamp field (case-insensitive for backward compat)
                    if ds.dtype.names:
                        ts_field = next(
                            (f for f in ds.dtype.names if f.lower() == "timestamp"), None
                        )
                        if ts_field and len(ds) > 1:
                            ts = ds[ts_field]
                            duration_s = float(ts[-1]) - float(ts[0])
                            stats["duration"] = self._format_duration(duration_s)
                    break

            # Event count
            if "Event" in f:
                stats["event_count"] = f["Event"].shape[0]

        return stats

    def _add_stat_item(self, layout: QtWidgets.QHBoxLayout, value: str, label: str):
        """Add a stat item (value + small label) to a layout."""
        item = QtWidgets.QVBoxLayout()
        item.setSpacing(0)
        item.setContentsMargins(0, 0, 0, 0)

        val_lbl = QtWidgets.QLabel(str(value))
        val_lbl.setStyleSheet(WidgetStyles.label(weight="bold"))
        val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item.addWidget(val_lbl)

        desc_lbl = QtWidgets.QLabel(label)
        desc_lbl.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        desc_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item.addWidget(desc_lbl)

        container = QtWidgets.QWidget()
        container.setLayout(item)
        layout.addWidget(container)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m"

    def _add_accuracy_bar(self, label: str, value: float, std: float | None = None, suffix: str = ""):
        """Add accuracy row with visual progress bar."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])

        lbl = QtWidgets.QLabel(f"{label}:")
        lbl.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED))
        lbl.setMinimumWidth(50)
        row.addWidget(lbl)

        # Progress bar
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(value * 100))
        bar.setTextVisible(False)
        bar.setFixedHeight(12)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: {BG_PANEL};
                border-radius: 6px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 6px;
            }}
        """)
        row.addWidget(bar, stretch=1)

        # Value text
        text = f"{value * 100:.1f}%"
        if std:
            text += f" ± {std * 100:.1f}%"
        text += suffix
        val_lbl = QtWidgets.QLabel(text)
        val_lbl.setStyleSheet(WidgetStyles.label())
        val_lbl.setMinimumWidth(100)
        row.addWidget(val_lbl)

        container = QtWidgets.QWidget()
        container.setLayout(row)
        self.fields_layout.addWidget(container)

    def clear(self):
        """Clear the panel to show placeholder."""
        self.current_record = None
        self.current_record_type = None
        self.details_container.hide()
        self.placeholder.show()

    def _update_action_buttons(self):
        """Update action buttons based on record type and file availability."""
        if self.current_record_type == "Recording":
            for btn in self.report_buttons:
                btn.show()

            h5_path = self.current_record.get("hdf5_file_path", "")
            has_valid_file = h5_path and os.path.exists(h5_path)
            study_name = self.current_record.get("study_name", "")

            # Session report button - enable if any H5 file exists
            raw_file = find_raw_eeg_file(self.current_record)
            metrics_file = find_metrics_file(self.current_record)
            has_any_file = (raw_file and os.path.exists(raw_file)) or (
                metrics_file and os.path.exists(metrics_file)
            )

            if has_any_file:
                self.session_report_btn.setEnabled(True)
                self.session_report_btn.setToolTip("")
            else:
                self.session_report_btn.setEnabled(False)
                self.session_report_btn.setToolTip("No data files found")

            # BIDS export button
            self.bids_export_btn.setEnabled(has_valid_file)
            self.bids_export_btn.setToolTip("" if has_valid_file else "H5 file not found")

            # Update study action in menu
            if study_name:
                self.bids_study_action.setText(f"Entire Study: {study_name}")
                self.bids_study_action.setEnabled(True)
            else:
                self.bids_study_action.setText("Entire Study (no study set)")
                self.bids_study_action.setEnabled(False)

        else:  # Non-recording types (Decoder, Study, Dataset)
            for btn in self.report_buttons:
                btn.hide()

        # Delete button: show for Recording, Decoder, Study; hide for Dataset
        if self.current_record_type in ("Recording", "Decoder", "Study"):
            self.delete_btn.show()
        else:
            self.delete_btn.hide()

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)
        # Brief tooltip feedback would be nice but keeping it simple

    def _run_session_report(self):
        """Run unified session report generation."""
        if not self.current_record:
            return

        from ..dialogs.report_progress import ReportProgressDialog

        # Find any available H5 file (prefer raw, fallback to metrics)
        raw_file = find_raw_eeg_file(self.current_record)
        metrics_file = find_metrics_file(self.current_record)

        h5_file_path = None
        if raw_file and os.path.exists(raw_file):
            h5_file_path = raw_file
        elif metrics_file and os.path.exists(metrics_file):
            h5_file_path = metrics_file

        if h5_file_path:
            file_type = detect_h5_file_type(h5_file_path)
            if file_type == "corrupted":
                show_file_error(self, "corrupted", h5_file_path)
                return
            dialog = ReportProgressDialog(h5_file_path, self, record=self.current_record)
            dialog.start_report()
            dialog.exec()
        else:
            show_file_error(self, "no_eeg_file")

    def _export_recording_bids(self):
        """Export single recording to BIDS format."""
        if not self.current_record:
            return

        h5_path = self.current_record.get("hdf5_file_path", "")
        if not h5_path or not os.path.exists(h5_path):
            show_file_error(self, "no_eeg_file")
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select BIDS Output Directory", "", QtWidgets.QFileDialog.Option.ShowDirsOnly
        )

        if not output_dir:
            return

        try:
            from dendrite.data.io import export_recording_to_bids

            study_name = self.current_record.get("study_name")
            fif_path = export_recording_to_bids(h5_path, output_dir, study_name=study_name)

            QtWidgets.QMessageBox.information(
                self,
                "BIDS Export Complete",
                f"Recording exported successfully.\n\nOutput: {fif_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export recording to BIDS:\n\n{e!s}"
            )

    def _export_study_bids(self):
        """Export entire study to BIDS format."""
        if not self.current_record:
            return

        study_name = self.current_record.get("study_name", "")
        if not study_name:
            QtWidgets.QMessageBox.warning(self, "No Study", "This recording has no study name set.")
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            f"Select BIDS Output Directory for Study: {study_name}",
            "",
            QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )

        if not output_dir:
            return

        try:
            from dendrite.data.io import export_study_to_bids

            # Find DB path from parent explorer
            db_explorer = self._find_db_explorer()
            db_path = db_explorer.db.db_path if db_explorer else str(DATABASE_PATH)

            export_study_to_bids(study_name, output_dir, db_path=db_path)

            QtWidgets.QMessageBox.information(
                self,
                "BIDS Export Complete",
                f"Study '{study_name}' exported successfully.\n\nOutput: {output_dir}/{study_name}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export study to BIDS:\n\n{e!s}"
            )

    def _delete_record(self):
        """Delete the current record after confirmation."""
        if not self.current_record:
            return

        # Dataset deletion not supported
        if self.current_record_type == "Dataset":
            return

        # Find DBExplorer parent (needed for Study cascade counts)
        db_explorer = self._find_db_explorer()

        # Get record identifier and build warning message
        if self.current_record_type == "Recording":
            record_id = self.current_record.get("recording_id", "N/A")
            record_name = self.current_record.get("recording_name", "Unknown")
            warning_msg = (
                f"Are you sure you want to delete this recording?\n\n"
                f"Name: {record_name}\n\n"
                f"This action cannot be undone."
            )
        elif self.current_record_type == "Decoder":
            record_id = self.current_record.get("decoder_id", "N/A")
            record_name = self.current_record.get("decoder_name", "Unknown")
            warning_msg = (
                f"Are you sure you want to delete this decoder?\n\n"
                f"Name: {record_name}\n\n"
                f"This action cannot be undone."
            )
        elif self.current_record_type == "Study":
            record_id = self.current_record.get("study_id", "N/A")
            record_name = self.current_record.get("study_name", "Unknown")

            # Get cascade counts for warning
            if db_explorer:
                recordings = db_explorer.repo.get_recordings_by_study(record_id)
                decoders = db_explorer.decoder_repo.get_decoders_by_study(record_id)
                rec_count = len(recordings)
                dec_count = len(decoders)
            else:
                rec_count = dec_count = "?"

            warning_msg = (
                f"Are you sure you want to delete this study?\n\n"
                f"Study: {record_name}\n\n"
                f"WARNING: This will also delete:\n"
                f"  - {rec_count} recording(s)\n"
                f"  - {dec_count} decoder(s)\n\n"
                f"This action cannot be undone."
            )
        else:
            return

        # Confirmation dialog
        reply = QtWidgets.QMessageBox.question(
            self,
            f"Delete {self.current_record_type}",
            warning_msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                if db_explorer is None:
                    QtWidgets.QMessageBox.critical(
                        self, "Error", "Cannot access database. Parent window not found."
                    )
                    return

                # Delete based on record type
                success = False
                if self.current_record_type == "Recording":
                    success = db_explorer.repo.delete_recording(record_id)
                elif self.current_record_type == "Decoder":
                    success = db_explorer.decoder_repo.delete_decoder(record_id)
                elif self.current_record_type == "Study":
                    success = db_explorer.study_repo.delete_study(record_id)

                if success:
                    QtWidgets.QMessageBox.information(
                        self, "Success", f"{self.current_record_type} deleted successfully."
                    )
                    self.clear()
                    self.record_deleted.emit()
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Deletion Failed",
                        f"Failed to delete {self.current_record_type.lower()}.",
                    )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Error deleting {self.current_record_type.lower()}:\n{e!s}"
                )

    def _find_db_explorer(self):
        """Find the DBExplorer parent window."""
        parent = self.parent()
        while parent is not None:
            if (
                hasattr(parent, "repo")
                and hasattr(parent, "decoder_repo")
                and hasattr(parent, "study_repo")
            ):
                return parent
            parent = parent.parent()
        return None
