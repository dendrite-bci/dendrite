"""
Preprocessing Configuration Widget

A widget for configuring preprocessing parameters including filters and chunk size.
Updated to dynamically support different modalities while maintaining backward compatibility.
"""

import re

from pydantic import ValidationError
from PyQt6 import QtCore, QtWidgets

from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_ELEVATED,
    BG_PANEL,
    BORDER_INPUT,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import TogglePillWidget
from dendrite.processing.preprocessing.preprocessing_schemas import (
    DEFAULT_EEG_CONFIG,
    DEFAULT_EMG_CONFIG,
    DEFAULT_EOG_CONFIG,
    PreprocessingConfig,
    QualityControlConfig,
)
from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from dendrite.processing.preprocessing.utils import get_valid_sample_rates
from dendrite.utils.logger_central import get_logger


class PreprocessingWidget(QtWidgets.QWidget):
    """Widget for configuring preprocessing parameters dynamically based on available modalities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.logger = get_logger()

        # Get stream config manager
        self.stream_config_manager = get_stream_config_manager()

        # Track dynamic widgets
        self.modality_widgets = {}
        self.current_modalities = {}
        self.streams_data = {}  # Hierarchical stream -> modality data
        self.stream_target_combos = {}  # Per-stream target sample rate combos
        self._last_signature = None  # For change detection

        self.setup_ui()

        # Connect to modality updates
        self.stream_config_manager.modality_data_changed.connect(self.update_modalities)

        # Initial update
        self.update_modalities()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(WidgetStyles.scrollarea)
        main_layout.addWidget(scroll_area)

        content_widget = QtWidgets.QWidget()
        scroll_area.setWidget(content_widget)

        preprocessing_layout = QtWidgets.QVBoxLayout(content_widget)
        preprocessing_layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"]
        )
        preprocessing_layout.setSpacing(LAYOUT["spacing"])

        prep_header = QtWidgets.QLabel("PREPROCESSING")
        prep_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        preprocessing_layout.addWidget(prep_header)
        preprocessing_layout.addSpacing(LAYOUT["spacing_sm"])

        prep_group = QtWidgets.QFrame()
        prep_group.setStyleSheet(
            WidgetStyles.frame(
                bg=BG_PANEL, border=False, radius=LAYOUT["radius"], padding=LAYOUT["padding_md"]
            )
        )
        prep_layout_inner = QtWidgets.QVBoxLayout(prep_group)
        prep_layout_inner.setContentsMargins(
            LAYOUT["padding_md"], LAYOUT["padding_md"], LAYOUT["padding_md"], LAYOUT["padding_md"]
        )
        prep_layout_inner.setSpacing(LAYOUT["spacing"])
        preprocessing_layout.addWidget(prep_group)

        toggle_layout = QtWidgets.QHBoxLayout()
        toggle_layout.setSpacing(LAYOUT["spacing_md"])

        toggle_label = QtWidgets.QLabel("Enable:")
        toggle_label.setStyleSheet(WidgetStyles.label())
        toggle_layout.addWidget(toggle_label)

        self.preprocess_toggle = TogglePillWidget(initial_state=True)
        toggle_layout.addWidget(self.preprocess_toggle)

        toggle_layout.addStretch()
        prep_layout_inner.addLayout(toggle_layout)

        preprocessing_layout.addSpacing(LAYOUT["spacing_xl"])

        modality_header = QtWidgets.QLabel("STREAMS")
        modality_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        preprocessing_layout.addWidget(modality_header)
        preprocessing_layout.addSpacing(LAYOUT["spacing_sm"])

        self.modality_container = QtWidgets.QWidget()
        self.modality_layout = QtWidgets.QVBoxLayout(self.modality_container)
        self.modality_layout.setContentsMargins(0, 0, 0, 0)
        self.modality_layout.setSpacing(LAYOUT["spacing_lg"])
        preprocessing_layout.addWidget(self.modality_container)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet(WidgetStyles.label("small", color=STATUS_ERROR))
        self.status_label.hide()
        preprocessing_layout.addWidget(self.status_label)

        preprocessing_layout.addStretch()

    def update_modalities(self):
        """Update widgets based on available streams and their modalities."""
        # Skip if manager is updating to prevent recursion
        if (
            hasattr(self.stream_config_manager, "_updating")
            and self.stream_config_manager._updating
        ):
            return

        # Get hierarchical stream -> modality data
        streams_data = self.stream_config_manager.get_modalities_by_stream()

        if not streams_data:
            self._show_status("No modalities detected. Configure streams first.", True)
            self._clear_modality_widgets()
            return

        self._show_status("", False)  # Clear status

        # Check if streams changed (by comparing stream UIDs and their modalities)
        current_signature = {
            uid: set(info["modalities"].keys()) for uid, info in streams_data.items()
        }
        if hasattr(self, "_last_signature") and current_signature == self._last_signature:
            return  # No change needed

        self._last_signature = current_signature
        self.streams_data = streams_data

        # Build flat modality lookup for backward compatibility
        self.current_modalities = {}
        for stream_uid, stream_info in streams_data.items():
            stream = stream_info["stream"]
            for modality_name, channels in stream_info["modalities"].items():
                # Track stream info alongside modality for sample_rate lookup
                self.current_modalities[modality_name] = {
                    "total_count": len(channels),
                    "channels": channels,
                    "stream_uid": stream_uid,
                    "sample_rate": stream.sample_rate,
                }

        self.logger.info(f"Updating preprocessing for streams: {list(streams_data.keys())}")

        # Rebuild widgets
        self._clear_modality_widgets()
        self._create_stream_widgets()

        # Check for missing expected modalities
        if "eeg" not in self.current_modalities:
            self._show_status("Missing EEG modality. Configure EEG stream first.", True)

    def _create_stream_widgets(self):
        """Create widgets for each stream with nested modality sections."""
        self.stream_target_combos.clear()

        for stream_uid, stream_info in self.streams_data.items():
            stream = stream_info["stream"]
            modalities = stream_info["modalities"]

            stream_container = QtWidgets.QFrame()
            stream_container.setStyleSheet("QFrame { background: transparent; }")
            container_layout = QtWidgets.QVBoxLayout(stream_container)
            container_layout.setContentsMargins(
                LAYOUT["padding_lg"],
                LAYOUT["padding_sm"],
                LAYOUT["padding_lg"],
                LAYOUT["padding_sm"],
            )
            container_layout.setSpacing(LAYOUT["spacing_md"])

            header_row = QtWidgets.QHBoxLayout()
            header_row.setSpacing(LAYOUT["spacing_md"])

            stream_type = list(modalities.keys())[0].upper() if modalities else "STREAM"
            stream_header = QtWidgets.QLabel(f"{stream_type} ({int(stream.sample_rate)} Hz)")
            stream_header.setStyleSheet(WidgetStyles.label("md", color=ACCENT, weight="semibold"))
            header_row.addWidget(stream_header)

            header_row.addStretch()

            target_combo = QtWidgets.QComboBox()
            target_combo.setStyleSheet(WidgetStyles.combobox())
            target_combo.setToolTip("Target sample rate for downsampling")

            source_rate = int(stream.sample_rate)
            target_combo.addItem(f"Native ({source_rate} Hz)", None)
            valid_rates = get_valid_sample_rates(source_rate)
            for rate in valid_rates:
                if rate != source_rate:
                    target_combo.addItem(f"{rate} Hz", rate)

            target_combo.currentIndexChanged.connect(self._validate_nyquist)
            header_row.addWidget(target_combo)

            self.stream_target_combos[stream_uid] = target_combo
            container_layout.addLayout(header_row)

            for modality_name, channels in modalities.items():
                modality_info = {
                    "total_count": len(channels),
                    "sample_rate": stream.sample_rate,
                    "stream_uid": stream_uid,
                }
                self._create_modality_section(modality_name, modality_info, container_layout)

            self.modality_layout.addWidget(stream_container)

    def _create_modality_section(
        self, modality_name: str, modality_info: dict, parent_layout: QtWidgets.QLayout = None
    ):
        """Create a configuration section for a specific modality.

        Args:
            modality_name: Name of the modality (e.g., 'eeg', 'emg')
            modality_info: Dict with 'total_count' and 'sample_rate'
            parent_layout: Optional parent layout. If None, adds to self.modality_layout
        """
        processor_class = OnlinePreprocessor.PROCESSOR_REGISTRY.get(modality_name.lower())
        processor_name = processor_class.__name__ if processor_class else "PassthroughProcessor"

        section = QtWidgets.QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_ELEVATED};
                border: 1px solid {BORDER_INPUT};
                border-radius: {LAYOUT["radius"]}px;
            }}
        """)

        section_layout = QtWidgets.QVBoxLayout(section)
        section_layout.setContentsMargins(
            LAYOUT["padding_sm"], LAYOUT["padding_sm"], LAYOUT["padding_sm"], LAYOUT["padding_sm"]
        )
        section_layout.setSpacing(LAYOUT["spacing_sm"])

        header = QtWidgets.QLabel(
            f"{modality_name.upper()} ({modality_info.get('total_count', 0)} ch)"
        )
        header.setStyleSheet(WidgetStyles.label("small", color=TEXT_MAIN))
        section_layout.addWidget(header)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form_layout.setSpacing(LAYOUT["spacing_sm"])
        section_layout.addLayout(form_layout)

        widgets = {}

        if processor_name == "EEGProcessor":
            widgets.update(self._create_eeg_inputs(form_layout))
        elif processor_name == "EMGProcessor":
            widgets.update(self._create_emg_inputs(form_layout))
        elif processor_name == "EOGProcessor":
            widgets.update(self._create_eog_inputs(form_layout))
        else:
            info = QtWidgets.QLabel("No preprocessing (passthrough)")
            info.setStyleSheet(WidgetStyles.label(color=TEXT_LABEL, style="italic"))
            form_layout.addRow(info)

        target_layout = parent_layout if parent_layout else self.modality_layout
        target_layout.addWidget(section)

        self.modality_widgets[modality_name] = {
            "section": section,
            "widgets": widgets,
            "processor_type": processor_name,
        }

    def _create_eeg_inputs(self, form_layout):
        """Create EEG-specific input widgets with schema defaults."""
        widgets = {}

        self._create_filter_input(
            widgets, "lowcut", DEFAULT_EEG_CONFIG["lowcut"], "Lowcut (Hz):", "EEG", form_layout
        )
        self._create_filter_input(
            widgets, "highcut", DEFAULT_EEG_CONFIG["highcut"], "Highcut (Hz):", "EEG", form_layout
        )

        widgets["apply_rereferencing"] = TogglePillWidget(
            initial_state=DEFAULT_EEG_CONFIG["apply_rereferencing"],
            show_label=False,
            width=28,
            height=14,
        )
        form_layout.addRow(self._create_label("Re-reference:"), widgets["apply_rereferencing"])

        separator_qc = QtWidgets.QFrame()
        separator_qc.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator_qc.setStyleSheet(WidgetStyles.separator())
        form_layout.addRow(separator_qc)

        qc_layout = QtWidgets.QHBoxLayout()
        qc_layout.setSpacing(LAYOUT["spacing_sm"])
        self.quality_control_toggle = TogglePillWidget(
            initial_state=False, show_label=False, width=28, height=14
        )
        self.quality_control_toggle.setToolTip("Detect channels with abnormally high amplitude")
        qc_label = QtWidgets.QLabel("Channel Quality Monitoring")
        qc_label.setStyleSheet(WidgetStyles.label())
        qc_layout.addWidget(self.quality_control_toggle)
        qc_layout.addWidget(qc_label)
        qc_layout.addStretch()
        form_layout.addRow(qc_layout)

        return widgets

    def _create_emg_inputs(self, form_layout):
        """Create EMG-specific input widgets with schema defaults."""
        widgets = {}

        self._create_filter_input(
            widgets, "lowcut", DEFAULT_EMG_CONFIG["lowcut"], "Lowcut (Hz):", "EMG", form_layout
        )
        self._create_filter_input(
            widgets, "highcut", DEFAULT_EMG_CONFIG["highcut"], "Highcut (Hz):", "EMG", form_layout
        )
        self._create_filter_input(
            widgets,
            "line_freq",
            DEFAULT_EMG_CONFIG["line_freq"],
            "Line Freq (Hz):",
            "EMG",
            form_layout,
        )

        return widgets

    def _create_eog_inputs(self, form_layout):
        """Create EOG-specific input widgets with schema defaults."""
        widgets = {}

        self._create_filter_input(
            widgets, "lowcut", DEFAULT_EOG_CONFIG["lowcut"], "Lowcut (Hz):", "EOG", form_layout
        )
        self._create_filter_input(
            widgets, "highcut", DEFAULT_EOG_CONFIG["highcut"], "Highcut (Hz):", "EOG", form_layout
        )

        return widgets

    def _create_label(self, text: str) -> QtWidgets.QLabel:
        """Create a styled label."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(WidgetStyles.label())
        return label

    def _create_filter_input(
        self,
        widgets: dict,
        key: str,
        default_value: float,
        label: str,
        modality: str,
        form_layout: QtWidgets.QFormLayout,
    ) -> None:
        """
        Helper to create filter input widgets (lowcut/highcut/line_freq).

        Args:
            widgets: Dictionary to store widget reference
            key: Widget key (e.g., 'lowcut', 'highcut')
            default_value: Default numeric value from schema
            label: Display label text
            modality: Modality name for validation context
            form_layout: Form layout to add widget to
        """
        widgets[key] = QtWidgets.QLineEdit(str(default_value))
        widgets[key].setStyleSheet(WidgetStyles.input())
        widgets[key].editingFinished.connect(
            lambda w=widgets[key], k=key: self._validate_field(w, k, modality)
        )
        if key == "highcut":
            widgets[key].editingFinished.connect(self._validate_nyquist)
        form_layout.addRow(self._create_label(label), widgets[key])

    def _set_toggle_from_config(
        self, config: dict, config_key: str, toggle_attr: str, nested_key: str = "enabled"
    ) -> None:
        """
        Helper to set toggle state from nested config dictionary.

        Args:
            config: Full configuration dictionary
            config_key: Top-level key (e.g., 'quality_control')
            toggle_attr: Widget attribute name (e.g., 'quality_control_toggle')
            nested_key: Key within nested dict (default: 'enabled')
        """
        if config_key in config and hasattr(self, toggle_attr):
            nested_config = config[config_key]
            if nested_key in nested_config:
                getattr(self, toggle_attr).setChecked(nested_config[nested_key])

    def _handle_validation_error(
        self, error_msg: str, log_level: str = "debug"
    ) -> tuple[bool, str]:
        """
        Helper to handle validation errors consistently.

        Args:
            error_msg: Error message to display and return
            log_level: Logging level ('debug' or 'error')

        Returns:
            Tuple of (False, error_msg)
        """
        self._show_status(error_msg, True)
        if log_level == "error":
            self.logger.error(error_msg)
        else:
            self.logger.debug(f"Preprocessing validation: {error_msg}")
        return False, error_msg

    def _clear_modality_widgets(self):
        """Clear all modality widgets and stream headers."""
        while self.modality_layout.count():
            item = self.modality_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.modality_widgets.clear()

    def _show_status(self, message: str, is_warning: bool):
        """Show or hide status message."""
        if message:
            color = STATUS_WARN if is_warning else STATUS_OK
            self.status_label.setStyleSheet(WidgetStyles.label("small", color=color))
            self.status_label.setText(message)
            self.status_label.show()
        else:
            self.status_label.hide()

    def _mark_field_invalid(self, field_widget: QtWidgets.QWidget, error_msg: str):
        """Mark a field as invalid with visual feedback."""
        if isinstance(field_widget, QtWidgets.QLineEdit):
            current_style = field_widget.styleSheet()
            if "border:" in current_style:
                current_style = re.sub(
                    r"border:[^;]+;", f"border: 2px solid {STATUS_ERROR};", current_style
                )
            else:
                current_style += f" border: 2px solid {STATUS_ERROR};"
            field_widget.setStyleSheet(current_style)

        self._show_status(error_msg, True)

    def _mark_field_valid(self, field_widget: QtWidgets.QWidget):
        """Mark a field as valid by removing error styling."""
        if isinstance(field_widget, QtWidgets.QLineEdit):
            field_widget.setStyleSheet(WidgetStyles.input())

        self._show_status("", False)

    def _validate_field(
        self, field_widget: QtWidgets.QLineEdit, field_name: str, modality: str | None = None
    ):
        """
        Validate field using Pydantic schema when user finishes editing.

        Validates entire config and shows errors specific to this field.
        Updates visual feedback (border styling and status message) based on validation result.

        Args:
            field_widget: The QLineEdit widget that was edited
            field_name: Name of the field (e.g., 'chunk_size', 'lowcut')
            modality: Optional modality name (reserved for future use, maintains API compatibility)
        """
        try:
            config = self.get_preprocessing_config()
            PreprocessingConfig(**config)
            self._mark_field_valid(field_widget)

        except ValidationError as e:
            for error in e.errors():
                error_msg = error["msg"]

                if field_name in error["loc"]:
                    self._mark_field_invalid(field_widget, error_msg)
                    return

                # Model validator errors have empty loc but field names in message
                if not error["loc"]:
                    if field_name.lower() in error_msg.lower():
                        self._mark_field_invalid(field_widget, error_msg)
                        return

            self._mark_field_valid(field_widget)

        except (ValueError, KeyError) as e:
            self._mark_field_invalid(field_widget, str(e) or "Invalid value")
        except Exception as e:
            self.logger.error(f"Validation error for {field_name}: {e}", exc_info=True)
            self._mark_field_valid(field_widget)  # Don't show confusing errors

    def _validate_nyquist(self):
        """Warn if highcut >= effective Nyquist after downsampling."""
        warnings = []

        for modality_name, widget_info in self.modality_widgets.items():
            modality_info = self.current_modalities.get(modality_name, {})
            stream_uid = modality_info.get("stream_uid")
            if not stream_uid:
                continue

            combo = self.stream_target_combos.get(stream_uid)
            if not combo:
                continue

            target_rate = combo.currentData()
            if not target_rate:
                continue  # No resampling for this stream

            effective_nyquist = target_rate / 2
            widgets = widget_info["widgets"]
            if "highcut" in widgets:
                try:
                    highcut = float(widgets["highcut"].text())
                    if highcut >= effective_nyquist:
                        warnings.append(
                            f"{modality_name}: highcut {highcut}Hz >= Nyquist {effective_nyquist}Hz"
                        )
                except ValueError:
                    pass

        if warnings:
            self._show_status(warnings[0], True)
        else:
            self._show_status("", False)

    def get_preprocessing_config(self) -> dict:
        """
        Get the current preprocessing configuration.

        Raises:
            ValueError: If any numeric input field contains invalid data (non-numeric text)
        """
        config = {
            "preprocess_data": self.preprocess_toggle.isChecked(),
            "target_sample_rate": None,
            "modality_preprocessing": {},
            "quality_control": QualityControlConfig().model_dump(),
        }

        if hasattr(self, "quality_control_toggle"):
            try:
                config["quality_control"]["enabled"] = self.quality_control_toggle.isChecked()
            except RuntimeError:
                pass  # Widget deleted during modality update

        for modality_name, widget_info in self.modality_widgets.items():
            modality_info = self.current_modalities.get(modality_name, {})

            stream_uid = modality_info.get("stream_uid")
            target_combo = self.stream_target_combos.get(stream_uid) if stream_uid else None
            target_rate = target_combo.currentData() if target_combo else None

            modality_config = {
                "num_channels": modality_info.get("total_count", 0),
                "sample_rate": modality_info.get("sample_rate", 500),
                "target_sample_rate": target_rate,
            }

            widgets = widget_info["widgets"]
            for key, widget in widgets.items():
                if isinstance(widget, QtWidgets.QLineEdit):
                    value = widget.text().strip()
                    if key in ["lowcut", "highcut", "line_freq", "notch_width"]:
                        if value:
                            try:
                                modality_config[key] = float(value)
                            except ValueError:
                                raise ValueError(
                                    f"{modality_name} {key} must be a valid number, got: '{value}'"
                                ) from None
                        else:
                            modality_config[key] = None
                    else:
                        if value:
                            try:
                                modality_config[key] = int(value)
                            except ValueError:
                                raise ValueError(
                                    f"{modality_name} {key} must be a valid integer, got: '{value}'"
                                ) from None
                        else:
                            modality_config[key] = None
                elif isinstance(widget, (QtWidgets.QCheckBox, TogglePillWidget)):
                    modality_config[key] = widget.isChecked()

            config["modality_preprocessing"][modality_name] = modality_config

        return config

    def set_preprocessing_config(self, config: dict):
        """Set the preprocessing configuration from a dictionary."""
        self.preprocess_toggle.setChecked(config.get("preprocess_data", True))

        self._set_toggle_from_config(config, "quality_control", "quality_control_toggle")

        modality_preprocessing = config.get("modality_preprocessing", {})
        for modality_name, modality_config in modality_preprocessing.items():
            target_rate = modality_config.get("target_sample_rate")
            modality_info = self.current_modalities.get(modality_name, {})
            stream_uid = modality_info.get("stream_uid")
            if stream_uid and stream_uid in self.stream_target_combos:
                combo = self.stream_target_combos[stream_uid]
                if target_rate is not None:
                    idx = combo.findData(target_rate)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentIndex(0)

            if modality_name in self.modality_widgets:
                widgets = self.modality_widgets[modality_name]["widgets"]
                for key, value in modality_config.items():
                    if key in widgets:
                        widget = widgets[key]
                        if isinstance(widget, QtWidgets.QLineEdit):
                            widget.setText(str(value))
                        elif isinstance(widget, (QtWidgets.QCheckBox, TogglePillWidget)):
                            widget.setChecked(bool(value))

    def validate_inputs(self) -> tuple[bool, str]:
        """
        Validate all input values using Pydantic schema.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            config = self.get_preprocessing_config()
        except ValueError as e:
            return self._handle_validation_error(str(e))

        try:
            PreprocessingConfig(**config)
            self._show_status("", False)
            return True, ""
        except ValidationError as e:
            errors = [
                f"{' → '.join(str(loc) for loc in error['loc'])}: {error['msg']}"
                for error in e.errors()
            ]

            error_msg = errors[0] if errors else "Validation failed"
            full_error_msg = "\n".join(f"• {err}" for err in errors)
            self._show_status(error_msg, True)
            self.logger.debug(f"Preprocessing validation failed:\n{full_error_msg}")
            return False, full_error_msg
        except Exception as e:
            return self._handle_validation_error(f"Validation error: {e!s}", log_level="error")
