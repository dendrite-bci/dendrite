"""
Add Recording Panel Widget

Right panel for adding a new recording to the database.
Replaces the modal AddRecordingDialog.
"""

import logging
import os
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from PyQt6 import QtCore, QtWidgets

from dendrite.data.io import get_h5_info
from dendrite.gui.config.study_schemas import StudyConfig
from dendrite.gui.styles.design_tokens import (
    SPACING,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

from ..utils import detect_h5_file_type, format_file_size, get_existing_studies, parse_h5_filename

logger = logging.getLogger(__name__)


class AddRecordingPanel(QtWidgets.QWidget):
    """Panel for adding a recording from an H5 file."""

    recording_added = QtCore.pyqtSignal(dict)  # Emitted with recording data on success
    cancelled = QtCore.pyqtSignal()  # Emitted when user cancels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.h5_file_path: str | None = None
        self.metadata: dict[str, Any] = {}
        self.metrics_file_path: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT["spacing_lg"], 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing"])

        header = QtWidgets.QLabel("Add Recording")
        header.setStyleSheet(WidgetStyles.label("subtitle", weight="bold"))
        layout.addWidget(header)

        # File picker section
        file_section = QtWidgets.QWidget()
        file_layout = QtWidgets.QVBoxLayout(file_section)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(SPACING["sm"])

        file_label = QtWidgets.QLabel("H5 File")
        file_label.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED, weight="bold"))
        file_layout.addWidget(file_label)

        file_row = QtWidgets.QHBoxLayout()
        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet(WidgetStyles.label())
        self.file_path_label.setWordWrap(True)
        file_row.addWidget(self.file_path_label, stretch=1)

        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.setStyleSheet(WidgetStyles.button())
        self.browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.browse_btn)
        file_layout.addLayout(file_row)

        layout.addWidget(file_section)

        # File info section (shown after file selected)
        self.file_info_section = QtWidgets.QWidget()
        self.file_info_section.hide()
        info_layout = QtWidgets.QFormLayout(self.file_info_section)
        info_layout.setSpacing(SPACING["xs"])
        info_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.file_size_label = QtWidgets.QLabel("—")
        self.file_size_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        info_layout.addRow("Size:", self.file_size_label)

        self.channels_label = QtWidgets.QLabel("—")
        self.channels_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        info_layout.addRow("Channels:", self.channels_label)

        self.sample_rate_label = QtWidgets.QLabel("—")
        self.sample_rate_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        info_layout.addRow("Sample Rate:", self.sample_rate_label)

        self.duration_label = QtWidgets.QLabel("—")
        self.duration_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_MUTED))
        info_layout.addRow("Duration:", self.duration_label)

        layout.addWidget(self.file_info_section)

        # Metadata section
        metadata_section = QtWidgets.QWidget()
        metadata_layout = QtWidgets.QFormLayout(metadata_section)
        metadata_layout.setSpacing(SPACING["sm"])
        metadata_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        meta_label = QtWidgets.QLabel("Recording Metadata")
        meta_label.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED, weight="bold"))
        metadata_layout.addRow(meta_label)

        self.recording_name_input = QtWidgets.QLineEdit()
        self.recording_name_input.setStyleSheet(WidgetStyles.input())
        self.recording_name_input.setPlaceholderText("Enter recording session name")
        metadata_layout.addRow("Name:*", self.recording_name_input)

        self.study_combo = QtWidgets.QComboBox()
        self.study_combo.setStyleSheet(WidgetStyles.combobox())
        self.study_combo.setEditable(True)
        self.study_combo.setPlaceholderText("Select or type study name...")
        metadata_layout.addRow("Study:*", self.study_combo)

        self.subject_id_input = QtWidgets.QLineEdit()
        self.subject_id_input.setStyleSheet(WidgetStyles.input())
        self.subject_id_input.setPlaceholderText("e.g., 01, pilot_1")
        metadata_layout.addRow("Subject ID:", self.subject_id_input)

        self.session_id_input = QtWidgets.QLineEdit()
        self.session_id_input.setStyleSheet(WidgetStyles.input())
        self.session_id_input.setPlaceholderText("e.g., 01, 02")
        metadata_layout.addRow("Session ID:", self.session_id_input)

        self.timestamp_input = QtWidgets.QLineEdit()
        self.timestamp_input.setStyleSheet(WidgetStyles.input())
        self.timestamp_input.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        metadata_layout.addRow("Timestamp:", self.timestamp_input)

        layout.addWidget(metadata_section)

        # Metrics file section (optional)
        metrics_section = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QVBoxLayout(metrics_section)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(SPACING["sm"])

        metrics_label = QtWidgets.QLabel("Metrics File (Optional)")
        metrics_label.setStyleSheet(WidgetStyles.label(color=TEXT_MUTED, weight="bold"))
        metrics_layout.addWidget(metrics_label)

        metrics_row = QtWidgets.QHBoxLayout()
        self.metrics_file_label = QtWidgets.QLabel("No file selected")
        self.metrics_file_label.setStyleSheet(WidgetStyles.label("small"))
        metrics_row.addWidget(self.metrics_file_label, stretch=1)

        self.browse_metrics_btn = QtWidgets.QPushButton("Browse...")
        self.browse_metrics_btn.setStyleSheet(
            WidgetStyles.button(variant="secondary", size="small")
        )
        self.browse_metrics_btn.clicked.connect(self._browse_metrics_file)
        metrics_row.addWidget(self.browse_metrics_btn)

        self.clear_metrics_btn = QtWidgets.QPushButton("Clear")
        self.clear_metrics_btn.setStyleSheet(WidgetStyles.button(variant="secondary", size="small"))
        self.clear_metrics_btn.clicked.connect(self._clear_metrics_file)
        self.clear_metrics_btn.setEnabled(False)
        metrics_row.addWidget(self.clear_metrics_btn)

        metrics_layout.addLayout(metrics_row)
        layout.addWidget(metrics_section)

        layout.addStretch()

        # Action buttons
        buttons_layout = QtWidgets.QHBoxLayout()

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))
        self.cancel_btn.clicked.connect(self._on_cancel)
        buttons_layout.addWidget(self.cancel_btn)

        buttons_layout.addStretch()

        self.save_btn = QtWidgets.QPushButton("Add to Database")
        self.save_btn.setStyleSheet(WidgetStyles.button(variant="primary"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        buttons_layout.addWidget(self.save_btn)

        layout.addLayout(buttons_layout)

        # Populate study combo
        self._populate_studies()

    def _populate_studies(self):
        """Populate study combo with existing studies."""
        self.study_combo.clear()
        existing_studies = get_existing_studies()
        if existing_studies:
            self.study_combo.addItems(existing_studies)
        self.study_combo.setCurrentIndex(-1)

    def reset(self):
        """Reset panel to initial state."""
        self.h5_file_path = None
        self.metadata = {}
        self.metrics_file_path = None

        self.file_path_label.setText("No file selected")
        self.file_info_section.hide()
        self.recording_name_input.clear()
        self.study_combo.setCurrentIndex(-1)
        self.subject_id_input.clear()
        self.session_id_input.clear()
        self.timestamp_input.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.metrics_file_label.setText("No file selected")
        self.clear_metrics_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self._populate_studies()

    def _browse_file(self):
        """Browse for H5 file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select H5 Recording File", "", "HDF5 Files (*.h5);;All Files (*)"
        )

        if file_path:
            self._load_file(file_path)

    def _load_file(self, file_path: str):
        """Load H5 file and extract metadata."""
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", f"Selected file does not exist:\n{file_path}"
            )
            return

        self.h5_file_path = os.path.abspath(file_path)
        self.metadata = self._extract_h5_metadata()

        self.file_path_label.setText(os.path.basename(self.h5_file_path))
        self.file_path_label.setToolTip(self.h5_file_path)

        self.file_size_label.setText(format_file_size(self.metadata["file_size"]))
        self.channels_label.setText(str(self.metadata["channels"]))
        self.sample_rate_label.setText(f"{self.metadata['sample_rate']} Hz")
        self.duration_label.setText(f"{self.metadata['duration']:.2f} seconds")
        self.file_info_section.show()

        # Pre-fill metadata fields
        self.recording_name_input.setText(self.metadata.get("recording_name", ""))
        if self.metadata.get("study_name"):
            idx = self.study_combo.findText(self.metadata["study_name"])
            if idx >= 0:
                self.study_combo.setCurrentIndex(idx)
            else:
                self.study_combo.setCurrentText(self.metadata["study_name"])
        self.subject_id_input.setText(self.metadata.get("subject_id", ""))
        self.session_id_input.setText(self.metadata.get("session_id", ""))
        self.timestamp_input.setText(self.metadata.get("session_timestamp", ""))

        self.save_btn.setEnabled(True)

    def _extract_h5_metadata(self) -> dict[str, Any]:
        """Extract metadata from H5 file."""
        metadata = {
            "file_path": self.h5_file_path,
            "file_name": os.path.basename(self.h5_file_path),
            "file_size": 0,
            "channels": 0,
            "sample_rate": 0,
            "duration": 0,
            "recording_name": "",
            "study_name": "",
            "subject_id": "",
            "session_id": "",
            "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            # Parse filename for study/experiment/timestamp
            parsed = parse_h5_filename(self.h5_file_path)
            metadata["study_name"] = parsed["study_name"]
            metadata["recording_name"] = parsed["experiment_name"]
            metadata["session_timestamp"] = parsed["session_timestamp"]

            # Use centralized H5 info function
            h5_info = get_h5_info(self.h5_file_path)
            metadata["file_size"] = h5_info["file_size"]

            # Extract BIDS fields from root attributes
            root_attrs = h5_info.get("root_attributes", {})
            metadata["subject_id"] = root_attrs.get("subject_id", "")
            metadata["session_id"] = root_attrs.get("session_id", "")

            # Extract EEG dataset info
            if "EEG" in h5_info["datasets"]:
                eeg_info = h5_info["datasets"]["EEG"]
                dtype_details = eeg_info["dtype_details"]

                if dtype_details["is_structured"]:
                    timestamp_fields = {"timestamp", "local_timestamp"}
                    all_fields = set(dtype_details["fields"].keys())
                    channel_fields = {f for f in all_fields if f.lower() not in timestamp_fields}
                    metadata["channels"] = len(channel_fields)
                else:
                    shape = eeg_info["shape"]
                    metadata["channels"] = shape[1] if len(shape) >= 2 else 0

                n_samples = eeg_info["shape"][0]
                attrs = eeg_info["attributes"]
                if "sampling_frequency" in attrs:
                    metadata["sample_rate"] = attrs["sampling_frequency"]
                elif "sample_rate" in attrs:
                    metadata["sample_rate"] = attrs["sample_rate"]

                if metadata["sample_rate"] > 0:
                    metadata["duration"] = n_samples / metadata["sample_rate"]

        except Exception as e:
            logger.error(f"Error extracting H5 metadata: {e}")

        return metadata

    def _browse_metrics_file(self):
        """Browse for metrics file."""
        start_dir = os.path.dirname(self.h5_file_path) if self.h5_file_path else ""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Metrics File", start_dir, "HDF5 Files (*.h5);;All Files (*)"
        )

        if file_path:
            file_type = detect_h5_file_type(file_path)
            if file_type == "metrics":
                self.metrics_file_path = os.path.abspath(file_path)
                self.metrics_file_label.setText(os.path.basename(file_path))
                self.clear_metrics_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid File",
                    "Selected file is not a metrics file.\nPlease select a file starting with 'metrics_'.",
                )

    def _clear_metrics_file(self):
        """Clear selected metrics file."""
        self.metrics_file_path = None
        self.metrics_file_label.setText("No file selected")
        self.clear_metrics_btn.setEnabled(False)

    def _on_cancel(self):
        """Handle cancel button."""
        self.reset()
        self.cancelled.emit()

    def _on_save(self):
        """Handle save button."""
        # Gather fields
        study_name = self.study_combo.currentText().strip() or "Unknown"
        subject_id = self.subject_id_input.text().strip()
        session_id = self.session_id_input.text().strip()
        recording_name = self.recording_name_input.text().strip()

        # Validate with Pydantic schema
        try:
            StudyConfig(
                study_name=study_name,
                subject_id=subject_id or "001",  # Allow empty for optional
                session_id=session_id or "01",
                recording_name=recording_name or "recording",
            )
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            QtWidgets.QMessageBox.warning(self, "Validation Error", f"Invalid field: {error_msg}")
            return

        # Recording name is required
        if not recording_name:
            QtWidgets.QMessageBox.warning(
                self, "Missing Information", "Recording name is required."
            )
            return

        recording_data = {
            "recording_name": recording_name,
            "study_name": study_name,
            "subject_id": subject_id,
            "session_id": session_id,
            "session_timestamp": self.timestamp_input.text().strip(),
            "hdf5_file_path": self.metadata["file_path"],
        }

        self.recording_added.emit(recording_data)
