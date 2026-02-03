"""
Synchronous mode configuration.

Handles configuration UI for synchronous (trial-based) mode including:
- Training settings (decoder search, augmentation)
- Event mappings and time window configuration
"""

from typing import Any

from PyQt6 import QtWidgets

from dendrite.gui.styles.design_tokens import TEXT_LABEL
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.ml.decoders import get_available_decoders
from dendrite.ml.decoders.decoder_schemas import NeuralNetConfig
from dendrite.ml.decoders.registry import get_decoder_entry

from .base import BaseModeConfig, DecoderSource
from .factory import ModeWidgetFactory, get_decoder_display_name


class SynchronousModeConfig(BaseModeConfig):
    """Configuration for Synchronous mode."""

    def create_config_widget(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create main configuration widget with pill navigation."""
        tabs = [
            ("general", "General"),
            ("decoder", "Decoder"),
            ("training", "Training"),
            ("events", "Events"),
        ]

        page, stacked_widget, pill_nav = self._create_standard_tab_widget_container(parent, tabs)
        config = self.get_config_from_manager()

        # Add tab widgets
        stacked_widget.addWidget(self._create_general_tab(config))
        stacked_widget.addWidget(self._create_decoder_tab(config))
        stacked_widget.addWidget(self._create_training_tab(config))
        stacked_widget.addWidget(self._create_events_tab(config))

        return page

    def _create_general_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create general settings tab with two-column layout."""
        tab, layout = self._create_general_tab_with_channels(config)
        layout.addStretch()
        return tab

    def _create_events_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create events configuration tab with time offsets."""
        return ModeWidgetFactory.create_events_tab_complete(
            config=config,
            parent_config=self,
            title="Event Mappings",
            info_text=(
                "Define event IDs and their labels for synchronous trials.\n"
                "Events are used to trigger data collection and classification.\n"
                "Time offsets define the data window relative to each event."
            ),
            include_time_offsets=True,
        )

    def _create_decoder_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create decoder configuration tab."""
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        main_layout.setSpacing(LAYOUT["spacing"])

        # Create pipeline section (new model configuration)
        pipeline_widget = self._create_pipeline_section(config)
        main_layout.addWidget(pipeline_widget)

        return tab

    def _create_pipeline_section(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create pipeline configuration section."""
        pipeline_widget = QtWidgets.QWidget()
        pipeline_main_layout = QtWidgets.QVBoxLayout(pipeline_widget)
        pipeline_main_layout.setContentsMargins(0, 10, 0, 0)

        # Decoder selection
        decoder_group = QtWidgets.QGroupBox("Decoder Selection")
        decoder_group.setStyleSheet(WidgetStyles.groupbox())
        decoder_layout = QtWidgets.QFormLayout(decoder_group)
        decoder_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        decoder_layout.setVerticalSpacing(LAYOUT["spacing_lg"])
        decoder_layout.setHorizontalSpacing(LAYOUT["spacing_xl"])

        model_combo = QtWidgets.QComboBox()
        model_combo.setStyleSheet(WidgetStyles.combobox())
        model_combo.setToolTip(
            "Select decoder type. Includes neural networks and classical ML pipelines."
        )

        # Populate from decoder registry
        available_decoders = get_available_decoders()
        for decoder in available_decoders:
            display_name = get_decoder_display_name(decoder)
            model_combo.addItem(display_name, decoder)

        default_model = NeuralNetConfig.model_fields["model_type"].default
        index = model_combo.findData(default_model)
        if index >= 0:
            model_combo.setCurrentIndex(index)

        decoder_label = QtWidgets.QLabel("Decoder:")
        decoder_label.setStyleSheet(WidgetStyles.label("small"))
        decoder_layout.addRow(decoder_label, model_combo)

        # Pipeline display (read-only)
        pipeline_label = QtWidgets.QLabel()
        pipeline_label.setStyleSheet(WidgetStyles.label("small", style="italic", color=TEXT_LABEL))
        preview_label = QtWidgets.QLabel("Pipeline:")
        preview_label.setStyleSheet(WidgetStyles.label("small"))
        decoder_layout.addRow(preview_label, pipeline_label)

        pipeline_main_layout.addWidget(decoder_group)

        # Store widgets
        self.widgets["model_combo"] = model_combo
        self.widgets["pipeline_label"] = pipeline_label

        # Connect signals
        model_combo.currentIndexChanged.connect(self._update_pipeline_display)

        # Initial display
        self._update_pipeline_display()

        return pipeline_widget

    def _create_training_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create simplified training settings tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        # Main training settings
        self._create_training_section(layout, config)

        # Info text
        info_label = QtWidgets.QLabel(
            "Decoder search uses Optuna to automatically find optimal hyperparameters. "
            "When disabled, proven defaults are used. For fine-grained control, use the Offline ML GUI."
        )
        info_label.setStyleSheet(WidgetStyles.label("small", style="italic", color=TEXT_LABEL))
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        return tab

    def _create_training_section(
        self, layout: QtWidgets.QVBoxLayout, config: dict[str, Any]
    ) -> None:
        """Create main training settings section."""
        group = QtWidgets.QGroupBox("Training Settings")
        group.setStyleSheet(WidgetStyles.groupbox())
        form = QtWidgets.QFormLayout(group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(LAYOUT["spacing_lg"])
        form.setHorizontalSpacing(LAYOUT["spacing_xl"])
        form.setContentsMargins(15, 15, 15, 15)

        # Training interval
        training_interval_edit = QtWidgets.QLineEdit(str(config.get("training_interval", 10)))
        training_interval_edit.setStyleSheet(WidgetStyles.input())
        training_interval_edit.setToolTip("Train model every N epochs collected")
        label = QtWidgets.QLabel("Training Interval (epochs):")
        label.setStyleSheet(WidgetStyles.label("small"))
        form.addRow(label, training_interval_edit)

        # Decoder search (prominent)
        enable_decoder_search = QtWidgets.QCheckBox("Enable Decoder Search (recommended)")
        enable_decoder_search.setStyleSheet(WidgetStyles.checkbox)
        enable_decoder_search.setChecked(config.get("enable_decoder_search", False))
        enable_decoder_search.setToolTip(
            "Trains multiple decoders with Optuna-sampled hyperparameters and selects the best. "
            "Recommended for optimal performance."
        )
        label = QtWidgets.QLabel("Hyperparameter Search:")
        label.setStyleSheet(WidgetStyles.label("small"))
        form.addRow(label, enable_decoder_search)

        # Data augmentation
        use_augmentation = QtWidgets.QCheckBox("Enable Data Augmentation")
        use_augmentation.setStyleSheet(WidgetStyles.checkbox)
        decoder_config = config.get("decoder_config", {})
        model_config = decoder_config.get("model_config", {})
        use_augmentation.setChecked(model_config.get("use_augmentation", False))
        use_augmentation.setToolTip(
            "Apply data augmentation during training to improve generalization"
        )
        label = QtWidgets.QLabel("Data Augmentation:")
        label.setStyleSheet(WidgetStyles.label("small"))
        form.addRow(label, use_augmentation)

        layout.addWidget(group)

        # Store widgets
        self.widgets["training_interval_edit"] = training_interval_edit
        self.widgets["enable_decoder_search"] = enable_decoder_search
        self.widgets["use_augmentation"] = use_augmentation

    def _update_pipeline_display(self, *args):
        """Update pipeline display based on selected decoder."""
        if "pipeline_label" not in self.widgets:
            return

        decoder = self.widgets["model_combo"].currentData()
        entry = get_decoder_entry(decoder)

        if entry and "default_steps" in entry:
            steps = entry["default_steps"]
        else:
            steps = ["classifier"]

        self.widgets["pipeline_label"].setText(" → ".join(steps))

    def _get_time_window_duration(self) -> float:
        """Get time window duration for synchronous mode (epoch length)."""
        config = self.get_config_from_manager()
        start_offset = config.get("start_offset", 0.0)
        end_offset = config.get("end_offset", 2.0)
        return end_offset - start_offset

    def _get_default_source(self) -> str:
        """Sync mode always uses new model."""
        return DecoderSource.RAW

    # Config extraction and validation

    def get_config_data(self) -> dict[str, Any]:
        """Extract configuration data from widgets."""
        model_config = {
            "model_type": self.widgets["model_combo"].currentData(),
            "use_augmentation": self.widgets.get(
                "use_augmentation", QtWidgets.QCheckBox()
            ).isChecked(),
            "use_scaler": True,  # Always use scaler for neural decoders
        }

        decoder_config = {"decoder_type": "Decoder", "model_config": model_config}

        return {
            "decoder_config": decoder_config,
            "start_offset": float(self.widgets["start_edit"].text() or 0.0),
            "end_offset": float(self.widgets["end_edit"].text() or 2.0),
            "training_interval": int(self.widgets["training_interval_edit"].text() or 10),
            "enable_decoder_search": self.widgets["enable_decoder_search"].isChecked(),
            "event_mapping": ModeWidgetFactory.get_event_mapping_from_table(
                self.widgets["events_table"]
            ),
        }

    def _update_widgets_from_config(self, data: dict[str, Any]):
        """Update widgets from configuration data."""
        with self._block_signals(
            [
                "model_combo",
                "start_edit",
                "end_edit",
                "training_interval_edit",
                "enable_decoder_search",
                "use_augmentation",
            ]
        ):
            # Extract model_type from nested structure
            model_type = None
            if "decoder_config" in data and "model_config" in data["decoder_config"]:
                model_type = data["decoder_config"]["model_config"].get("model_type")
            elif "model_type" in data:
                model_type = data["model_type"]

            # Set model type
            if "model_combo" in self.widgets:
                model_combo = self.widgets["model_combo"]
                if model_type is None:
                    model_type = NeuralNetConfig.model_fields["model_type"].default
                index = model_combo.findData(model_type)
                if index >= 0:
                    model_combo.setCurrentIndex(index)

            # Update simple fields
            if "start_offset" in data and "start_edit" in self.widgets:
                self.widgets["start_edit"].setText(str(data["start_offset"]))
            if "end_offset" in data and "end_edit" in self.widgets:
                self.widgets["end_edit"].setText(str(data["end_offset"]))
            if "training_interval" in data and "training_interval_edit" in self.widgets:
                self.widgets["training_interval_edit"].setText(str(data["training_interval"]))
            if "enable_decoder_search" in data and "enable_decoder_search" in self.widgets:
                self.widgets["enable_decoder_search"].setChecked(data["enable_decoder_search"])

            # Update use_augmentation from model_config
            decoder_config = data.get("decoder_config", {})
            model_config = decoder_config.get("model_config", {})
            if "use_augmentation" in self.widgets:
                use_aug = model_config.get("use_augmentation", False)
                self.widgets["use_augmentation"].setChecked(use_aug)
