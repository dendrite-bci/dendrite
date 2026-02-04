"""Dialog classes for offline streaming GUI."""

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_INPUT,
    BG_MAIN,
    BG_PANEL,
    FONT_SIZE,
    FONT_WEIGHT,
    PADDING,
    RADIUS,
    SPACING,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import WidgetStyles
from dendrite.gui.widgets.common.toggle_pill import TogglePillWidget
from dendrite.utils.format_loaders import get_file_filter

from ..utils import STREAM_TYPE_PRESETS, TYPE_NORMALIZATION, get_file_info, get_source_display


class StreamDetailsDialog(QtWidgets.QDialog):
    """Dialog showing stream details and events."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle(f"Stream Details: {config.get('name', 'Unnamed')}")
        self.setFixedWidth(400)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(SPACING["md"])

        source_display = get_source_display(self.config)
        source_type = self.config.get("source_type", "generated")

        # Basic info
        info = [
            ("Name", self.config.get("name", "Unnamed")),
            ("Type", self.config["type"]),
            ("Source", source_display),
            ("Channels", str(self.config.get("channels", "-"))),
            ("Sample Rate", f"{self.config.get('sample_rate', '-')} Hz"),
        ]

        # Add source-specific info
        if source_type == "moabb":
            info.append(("Subject", str(self.config.get("subject_id", "?"))))
            info.append(
                ("Events Stream", "Enabled" if self.config.get("enable_events") else "Disabled")
            )
        elif source_type == "file" and self.config.get("file_path"):
            info.append(("File", Path(self.config["file_path"]).name))
            info.append(
                ("Events Stream", "Enabled" if self.config.get("enable_events") else "Disabled")
            )

        for key, value in info:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"{key}:"))
            row.addWidget(QtWidgets.QLabel(str(value)))
            layout.addLayout(row)

        # Events section for file-based streams
        if source_type == "file" and self.config.get("file_path"):
            file_info = get_file_info(self.config["file_path"])
            if file_info:
                duration, events, event_ids = file_info
                layout.addWidget(QtWidgets.QLabel(f"Duration: {duration:.1f}s"))

                if event_ids:
                    self.events_label = QtWidgets.QLabel("Events:")
                    layout.addWidget(self.events_label)

                    events_text = QtWidgets.QTextEdit()
                    events_text.setReadOnly(True)
                    events_text.setMaximumHeight(150)
                    events_text.setText(
                        f"{len(events)} events, {len(event_ids)} types:\n"
                        + "\n".join([f"  {k} -> {v}" for k, v in event_ids.items()])
                    )
                    layout.addWidget(events_text)

        # Events section for MOABB streams
        elif source_type == "moabb":
            preset_name = self.config.get("preset_name")
            if preset_name:
                from dendrite.data import get_moabb_dataset_info

                info = get_moabb_dataset_info(preset_name)
                if info:
                    preset_config = info["config"]
                    if preset_config.events:
                        layout.addWidget(QtWidgets.QLabel("Event mapping (marker codes):"))
                        events_text = QtWidgets.QTextEdit()
                        events_text.setReadOnly(True)
                        events_text.setMaximumHeight(100)
                        # Show +1 offset codes (0 = no event in marker channel)
                        events_text.setText(
                            "\n".join(
                                [
                                    f"  {name} -> {code + 1}"
                                    for name, code in preset_config.events.items()
                                ]
                            )
                            + "\n  (0 = no event)"
                        )
                        layout.addWidget(events_text)

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(WidgetStyles.button())
        layout.addWidget(close_btn)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_MAIN}; color: {TEXT_MAIN}; }}
            QLabel {{ color: {TEXT_LABEL}; font-size: {FONT_SIZE["md"]}px; }}
            QTextEdit {{ background: {BG_INPUT}; color: {TEXT_MAIN}; border: none;
                        border-radius: {RADIUS["md"]}px; padding: {PADDING["md"]}px; }}
        """)


class MOAABSubjectDialog(QtWidgets.QDialog):
    """Dialog for selecting MOABB subject and streaming options."""

    def __init__(self, preset_name: str, config, parent=None):
        super().__init__(parent)
        self.preset_name = preset_name
        self.config = config
        self.setWindowTitle(f"MOABB: {preset_name}")
        self.setFixedWidth(400)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(SPACING["lg"])

        title = QtWidgets.QLabel(self.preset_name.replace("_", " "))
        title.setStyleSheet(WidgetStyles.label(variant="header"))
        layout.addWidget(title)

        if self.config.description:
            desc = QtWidgets.QLabel(self.config.description)
            desc.setWordWrap(True)
            desc.setStyleSheet(WidgetStyles.label(size="small"))
            layout.addWidget(desc)

        # Info row
        info_text = (
            f"Paradigm: {self.config.moabb_paradigm} | Sample rate: {self.config.sample_rate} Hz"
        )
        info = QtWidgets.QLabel(info_text)
        info.setStyleSheet(WidgetStyles.label(size="small"))
        layout.addWidget(info)

        # Subject selection
        subject_group = QtWidgets.QGroupBox("Subject Selection")
        subject_layout = QtWidgets.QFormLayout(subject_group)
        subject_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.subject_combo = QtWidgets.QComboBox()
        for subject_id in self.config.subjects:
            self.subject_combo.addItem(f"Subject {subject_id}", subject_id)
        subject_layout.addRow("Subject:", self.subject_combo)
        layout.addWidget(subject_group)

        # Event stream option
        events_row = QtWidgets.QHBoxLayout()
        events_label = QtWidgets.QLabel("Create separate Events stream:")
        events_label.setStyleSheet(WidgetStyles.label(size="small"))
        self.events_toggle = TogglePillWidget(initial_state=True, show_label=False)
        self.events_toggle.setToolTip("When enabled, events are streamed on a separate LSL outlet")
        events_row.addWidget(events_label)
        events_row.addWidget(self.events_toggle)
        events_row.addStretch()
        layout.addLayout(events_row)

        events_help = QtWidgets.QLabel("Events will stream on a separate LSL outlet")
        events_help.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        layout.addWidget(events_help)

        # Note about first-time download
        note = QtWidgets.QLabel(
            "Note: First-time use will download the dataset from MOABB servers."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        layout.addWidget(note)

        layout.addStretch()

        button_layout = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))

        add_btn = QtWidgets.QPushButton("Add Stream")
        add_btn.clicked.connect(self.accept)
        add_btn.setStyleSheet(WidgetStyles.button())
        add_btn.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(add_btn)
        layout.addLayout(button_layout)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_MAIN}; color: {TEXT_MAIN}; }}
            QGroupBox {{
                font-size: {FONT_SIZE["md"]}px;
                font-weight: {FONT_WEIGHT["semibold"]};
                color: {TEXT_MAIN};
                border: none;
                margin-top: {SPACING["lg"]}px;
                padding-top: {SPACING["sm"]}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {SPACING["lg"]}px;
                padding: 0 {SPACING["sm"]}px;
                color: {ACCENT};
            }}
            QComboBox {{
                background: {BG_INPUT};
                color: {TEXT_MAIN};
                border: none;
                padding: {PADDING["md"]}px;
                border-radius: {RADIUS["md"]}px;
            }}
        """)

    def get_config(self) -> dict:
        subject_id = self.subject_combo.currentData()
        return {
            "type": "EEG",
            "name": f"MOABB_{self.preset_name}_S{subject_id}",
            "source_type": "moabb",
            "preset_name": self.preset_name,
            "subject_id": subject_id,
            "enable_events": self.events_toggle.isChecked(),
            "channels": 0,  # Will be determined at stream time
            "sample_rate": self.config.sample_rate,
        }


class StreamConfigDialog(QtWidgets.QDialog):
    """Stream configuration dialog with custom stream type and conditional file path enabling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Stream")
        self.setFixedSize(450, 600)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(SPACING["lg"])

        self.dialog_title = QtWidgets.QLabel("Create New Stream")
        self.dialog_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.dialog_title)

        # Stream type selection
        type_group = QtWidgets.QGroupBox("Stream Type")
        type_layout = QtWidgets.QVBoxLayout(type_group)

        self.type_combo = QtWidgets.QComboBox()
        # Common physiological signal types
        common_types = ["EEG", "EMG", "ECG/EKG", "Events", "ContinuousEvents", "GSR/EDA"]
        # Additional signal types
        additional_types = [
            "Position",
            "Force/Torque",
            "Acceleration",
            "Gyroscope",
            "Temperature",
            "Breathing",
            "Custom",
        ]
        self.type_combo.addItems(common_types + additional_types)
        self.type_combo.currentIndexChanged.connect(self._update_type_settings)
        type_layout.addWidget(self.type_combo)

        # Custom type input (initially hidden)
        self.custom_type = QtWidgets.QLineEdit()
        self.custom_type.setPlaceholderText(
            "Enter custom stream type (e.g., 'Pressure', 'pH', 'Humidity')..."
        )
        self.custom_type.setVisible(False)
        type_layout.addWidget(self.custom_type)

        layout.addWidget(type_group)

        # Source selection
        source_group = QtWidgets.QGroupBox("Data Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)

        self.source = QtWidgets.QComboBox()
        self.source.addItems(["Synthetic", "File"])
        self.source.currentIndexChanged.connect(self._toggle_file)
        source_layout.addWidget(self.source)

        # File input (initially hidden)
        self.file_layout = QtWidgets.QHBoxLayout()
        self.file_input = QtWidgets.QLineEdit()
        self.file_input.setPlaceholderText("Select file (.set or .fif for EEG)...")
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse)
        self.file_layout.addWidget(self.file_input)
        self.file_layout.addWidget(self.browse_btn)

        # Create a widget to contain the file layout for easy hiding
        self.file_widget = QtWidgets.QWidget()
        self.file_widget.setLayout(self.file_layout)
        self.file_widget.setVisible(False)
        source_layout.addWidget(self.file_widget)

        # Events stream option (for file sources)
        self.events_widget = QtWidgets.QWidget()
        events_container = QtWidgets.QVBoxLayout(self.events_widget)
        events_container.setContentsMargins(0, 0, 0, 0)
        events_container.setSpacing(SPACING["xs"])

        events_row = QtWidgets.QHBoxLayout()
        events_label = QtWidgets.QLabel("Create separate Events stream:")
        events_label.setStyleSheet(WidgetStyles.label(size="small"))
        self.events_toggle = TogglePillWidget(initial_state=True, show_label=False)
        self.events_toggle.setToolTip("When enabled, events are streamed on a separate LSL outlet")
        events_row.addWidget(events_label)
        events_row.addWidget(self.events_toggle)
        events_row.addStretch()
        events_container.addLayout(events_row)

        events_help = QtWidgets.QLabel("Events will stream on a separate LSL outlet")
        events_help.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        events_container.addWidget(events_help)

        self.events_widget.setVisible(False)
        source_layout.addWidget(self.events_widget)

        layout.addWidget(source_group)

        # Stream parameters
        params_group = QtWidgets.QGroupBox("Stream Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)

        self.name = QtWidgets.QLineEdit("MyStream")
        params_layout.addRow("Name:", self.name)

        self.channels = QtWidgets.QLineEdit("64")
        self.channels.setToolTip("Number of channels for this stream type")
        params_layout.addRow("Channels:", self.channels)

        self.sample_rate = QtWidgets.QLineEdit("500")
        self.sample_rate.setToolTip("Sample rate in Hz (e.g., 500 for EEG, 1000 for EMG)")
        params_layout.addRow("Sample Rate (Hz):", self.sample_rate)

        layout.addWidget(params_group)

        # Help text
        self.help_text = QtWidgets.QLabel()
        self.help_text.setWordWrap(True)
        layout.addWidget(self.help_text)

        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.create_btn = QtWidgets.QPushButton("Create")
        self.create_btn.clicked.connect(self.accept)
        self.create_btn.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.create_btn)
        layout.addLayout(button_layout)

        # Initialize settings for default type
        self._update_type_settings()

    def _is_default_name(self, name: str) -> bool:
        """Check if name is a default/placeholder value."""
        return name == "MyStream" or name.endswith("_Stream") or not name.strip()

    def _normalize_stream_type(self, base_type: str) -> str:
        """Normalize display type names to canonical names (e.g., 'ECG/EKG' -> 'ECG')."""
        if base_type == "Custom" and self.custom_type.text():
            return self.custom_type.text().strip()
        return TYPE_NORMALIZATION.get(base_type, base_type)

    def _update_type_settings(self):
        """Update UI settings based on selected stream type."""
        stream_type = self.type_combo.currentText()

        # Show/hide custom type input
        self.custom_type.setVisible(stream_type == "Custom")
        if stream_type != "Custom":
            self.custom_type.clear()

        # Apply preset from data dict
        preset = STREAM_TYPE_PRESETS.get(stream_type, (1, 100, f"{stream_type}_Stream", ""))
        channels, sample_rate, default_name, help_text = preset

        self.channels.setText(str(channels))
        self.sample_rate.setText(str(sample_rate))
        self.help_text.setText(help_text)

        # Update name field
        if self.source.currentIndex() == 1 and self.file_input.text():
            self._update_name_from_file(self.file_input.text())
        elif self._is_default_name(self.name.text()):
            self.name.setText(default_name)

    def apply_styles(self):
        """Apply V2 styles."""
        self.setStyleSheet(f"""
            QDialog {{
                background: {BG_MAIN};
                color: {TEXT_MAIN};
                border: none;
            }}
            QGroupBox {{
                font-size: {FONT_SIZE["md"]}px;
                font-weight: {FONT_WEIGHT["semibold"]};
                color: {TEXT_MAIN};
                border: none;
                border-radius: {RADIUS["lg"]}px;
                margin-top: {SPACING["lg"]}px;
                padding-top: {SPACING["sm"]}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {SPACING["lg"]}px;
                padding: 0 {SPACING["sm"]}px;
                color: {ACCENT};
            }}
            QComboBox, QLineEdit {{
                background: {BG_INPUT};
                color: {TEXT_MAIN};
                border: none;
                padding: {PADDING["md"]}px;
                border-radius: {RADIUS["md"]}px;
                font-size: {FONT_SIZE["md"]}px;
            }}
            QComboBox:focus, QLineEdit:focus {{
                background: {BG_PANEL};
            }}
            QLabel {{
                color: {TEXT_LABEL};
                font-size: {FONT_SIZE["md"]}px;
                border: none;
            }}
        """)

        # Apply V2 button styling
        self.cancel_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))
        self.create_btn.setStyleSheet(WidgetStyles.button())
        self.browse_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))

        # Apply title styling
        self.dialog_title.setStyleSheet(WidgetStyles.label(variant="title"))

    def _toggle_file(self, index):
        """Show/hide file input based on source selection"""
        show_file = index == 1  # 1 = 'File'
        self.file_widget.setVisible(show_file)
        self.events_widget.setVisible(show_file)
        if not show_file:
            self.file_input.clear()

    def _browse(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Data File", "", get_file_filter()
        )
        if file:
            self.file_input.setText(file)
            # Auto-update the name field to include filename
            self._update_name_from_file(file)

    def _update_name_from_file(self, file_path: str):
        """Update the name field to include the filename when a file is selected."""
        if not file_path:
            return

        filename_without_ext = Path(file_path).stem
        stream_type = self._normalize_stream_type(self.type_combo.currentText())

        if self._is_default_name(self.name.text()):
            self.name.setText(f"{stream_type}_{filename_without_ext}")

    def get_config(self) -> dict:
        stream_type = self._normalize_stream_type(self.type_combo.currentText())

        # Parse sample rate
        try:
            sample_rate = float(self.sample_rate.text() or 100)
        except ValueError:
            sample_rate = 100.0

        # Parse channels
        try:
            channels = int(self.channels.text() or 1)
        except ValueError:
            channels = 1

        # Generate stream name
        stream_name = self.name.text() or f"{stream_type}_Stream"

        is_file_source = self.source.currentIndex() == 1
        return {
            "type": stream_type,
            "name": stream_name,
            "source_type": "file" if is_file_source else "generated",
            "file_path": self.file_input.text() if is_file_source else "",
            "channels": channels,
            "sample_rate": sample_rate,
            "enable_events": self.events_toggle.isChecked() if is_file_source else False,
        }


class FileStreamDialog(QtWidgets.QDialog):
    """Dialog for file-based stream configuration with events option."""

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.file_info = get_file_info(file_path)
        self.setWindowTitle("File Stream")
        self.setFixedWidth(400)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(SPACING["lg"])

        # Header - filename
        filename = Path(self.file_path).stem
        title = QtWidgets.QLabel(filename)
        title.setStyleSheet(WidgetStyles.label(variant="header"))
        title.setWordWrap(True)
        layout.addWidget(title)

        # File info
        if self.file_info:
            duration, events, event_ids = self.file_info
            info_text = f"Duration: {duration:.1f}s"
            info = QtWidgets.QLabel(info_text)
            info.setStyleSheet(WidgetStyles.label(size="small"))
            layout.addWidget(info)

            # Events info
            if event_ids:
                events_group = QtWidgets.QGroupBox("Events")
                events_layout = QtWidgets.QVBoxLayout(events_group)

                events_text = QtWidgets.QTextEdit()
                events_text.setReadOnly(True)
                events_text.setMaximumHeight(100)
                events_text.setText(
                    f"{len(events)} events, {len(event_ids)} types:\n"
                    + "\n".join([f"  {k} -> {v}" for k, v in event_ids.items()])
                )
                events_layout.addWidget(events_text)
                layout.addWidget(events_group)

        # Event stream option
        events_row = QtWidgets.QHBoxLayout()
        events_label = QtWidgets.QLabel("Create separate Events stream:")
        events_label.setStyleSheet(WidgetStyles.label(size="small"))
        self.events_toggle = TogglePillWidget(initial_state=True, show_label=False)
        self.events_toggle.setToolTip("When enabled, events are streamed on a separate LSL outlet")
        events_row.addWidget(events_label)
        events_row.addWidget(self.events_toggle)
        events_row.addStretch()
        layout.addLayout(events_row)

        events_help = QtWidgets.QLabel("Events will stream on a separate LSL outlet")
        events_help.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {FONT_SIZE['sm']}px;")
        layout.addWidget(events_help)

        layout.addStretch()

        button_layout = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))

        add_btn = QtWidgets.QPushButton("Add Stream")
        add_btn.clicked.connect(self.accept)
        add_btn.setStyleSheet(WidgetStyles.button())
        add_btn.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(add_btn)
        layout.addLayout(button_layout)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_MAIN}; color: {TEXT_MAIN}; }}
            QGroupBox {{
                font-size: {FONT_SIZE["md"]}px;
                font-weight: {FONT_WEIGHT["semibold"]};
                color: {TEXT_MAIN};
                border: none;
                margin-top: {SPACING["lg"]}px;
                padding-top: {SPACING["sm"]}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {SPACING["lg"]}px;
                padding: 0 {SPACING["sm"]}px;
                color: {ACCENT};
            }}
            QTextEdit {{ background: {BG_INPUT}; color: {TEXT_MAIN}; border: none;
                        border-radius: {RADIUS["md"]}px; padding: {PADDING["md"]}px; }}
        """)

    def get_config(self) -> dict:
        """Return stream configuration."""
        from dendrite.utils.format_loaders import load_file

        # Load file to get metadata
        loaded = load_file(self.file_path)
        filename = Path(self.file_path).stem

        return {
            "type": "EEG",
            "name": filename,
            "source_type": "file",
            "file_path": self.file_path,
            "channels": len(loaded.channel_names),
            "sample_rate": loaded.sample_rate,
            "enable_events": self.events_toggle.isChecked(),
        }
