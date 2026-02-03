"""
Asynchronous mode configuration widget.

Contains configuration for asynchronous mode including decoder source selection
and window parameters.
"""

import os
from typing import Any

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog

from dendrite.gui.styles.design_tokens import TEXT_MUTED
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.source_card import DescriptiveSourceCardGroup

from ..decoder_browser import InlineDecoderBrowser
from .base import BaseModeConfig, DecoderSource
from .factory import ModeWidgetFactory


class AsynchronousModeConfig(BaseModeConfig):
    """Configuration for Asynchronous mode"""

    # Source indices for QStackedWidget
    _SOURCE_INDEX = {
        DecoderSource.PRETRAINED: 0,
        DecoderSource.DATABASE: 1,
        DecoderSource.SYNC_MODE: 2,
    }

    def create_config_widget(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        # Define tabs
        tabs = [("general", "General"), ("decoder", "Decoder"), ("events", "Events")]

        # Create standard pill navigation container
        page, stacked_widget, pill_nav = self._create_standard_tab_widget_container(parent, tabs)

        # Cache config once for all tab creation methods
        config = self.get_config_from_manager()

        # Add mode-specific tab widgets
        stacked_widget.addWidget(self.create_general_settings_tab(config))
        stacked_widget.addWidget(self.create_decoder_config_tab(config))
        stacked_widget.addWidget(self.create_events_tab(config))

        return page

    def create_general_settings_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create general settings tab with two-column layout."""
        tab, right_layout = self._create_general_tab_with_channels(config)

        # Processing parameters
        processing_group = QtWidgets.QGroupBox("Processing Parameters")
        processing_group.setStyleSheet(WidgetStyles.groupbox())
        processing_layout = QtWidgets.QFormLayout(processing_group)
        processing_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        processing_layout.setVerticalSpacing(LAYOUT["spacing_lg"])

        step_size_edit = QtWidgets.QLineEdit()
        step_size_edit.setStyleSheet(WidgetStyles.input())
        step_size_edit.setPlaceholderText("50")
        step_size_label = QtWidgets.QLabel("Step Size (ms):")
        step_size_label.setStyleSheet(WidgetStyles.label("small"))
        step_size_label.setToolTip("Prediction interval - how often predictions are made")
        processing_layout.addRow(step_size_label, step_size_edit)

        right_layout.addWidget(processing_group)
        self.widgets["step_size_edit"] = step_size_edit

        right_layout.addStretch()
        return tab

    def create_decoder_config_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create decoder configuration tab"""
        tab = QtWidgets.QWidget()
        layout = ModeWidgetFactory.create_styled_form_layout(tab)

        # Decoder source selection using descriptive option cards
        decoder_source_group = QtWidgets.QGroupBox("Decoder Source")
        decoder_source_group.setStyleSheet(WidgetStyles.groupbox())
        source_layout = QtWidgets.QVBoxLayout(decoder_source_group)

        # Create descriptive source card selector with clear labels and descriptions
        source_card_group = DescriptiveSourceCardGroup()
        source_card_group.add_card(
            DecoderSource.PRETRAINED, "Load from File", "Import saved decoder (.json)"
        )
        source_card_group.add_card(
            DecoderSource.DATABASE, "From Study", "Use previously trained decoder"
        )
        source_card_group.add_card(
            DecoderSource.SYNC_MODE, "Live Training", "Link to a Synchronous mode"
        )

        source_layout.addWidget(source_card_group)

        # Stacked widget for source-specific content (only one visible at a time)
        source_stack = QtWidgets.QStackedWidget()
        source_stack.setContentsMargins(0, LAYOUT["spacing"], 0, 0)

        pretrained_container = QtWidgets.QWidget()
        pretrained_vlayout = QtWidgets.QVBoxLayout(pretrained_container)
        pretrained_vlayout.setContentsMargins(0, LAYOUT["spacing"], 0, 0)
        pretrained_vlayout.setSpacing(LAYOUT["spacing"])

        # Horizontal row: path field + browse button
        file_row = QtWidgets.QHBoxLayout()
        file_row.setSpacing(LAYOUT["spacing"])

        pretrained_path_edit = QtWidgets.QLineEdit()
        pretrained_path_edit.setStyleSheet(WidgetStyles.input())
        pretrained_path_edit.setPlaceholderText("No file selected - paste path or browse")
        pretrained_path_edit.textChanged.connect(self._update_decoder_status_display)
        file_row.addWidget(pretrained_path_edit, stretch=1)

        pretrained_browse_btn = QtWidgets.QPushButton("Browse...")
        pretrained_browse_btn.setStyleSheet(WidgetStyles.button())
        pretrained_browse_btn.clicked.connect(self._browse_pretrained_model)
        file_row.addWidget(pretrained_browse_btn)

        pretrained_vlayout.addLayout(file_row)
        pretrained_vlayout.addStretch()
        source_stack.addWidget(pretrained_container)  # Index 0

        self.widgets["pretrained_path_edit"] = pretrained_path_edit
        self.widgets["pretrained_browse_btn"] = pretrained_browse_btn

        current_study = self._get_current_study_name()
        database_browser = InlineDecoderBrowser(default_study_name=current_study)
        database_browser.decoder_selected.connect(self._on_database_decoder_selected)
        source_stack.addWidget(database_browser)  # Index 1

        sync_container = QtWidgets.QWidget()
        sync_vlayout = QtWidgets.QVBoxLayout(sync_container)
        sync_vlayout.setContentsMargins(0, 0, 0, 0)
        sync_vlayout.setSpacing(LAYOUT["spacing"])

        # Sync mode selection
        sync_mode_row = QtWidgets.QHBoxLayout()
        sync_mode_label = QtWidgets.QLabel("Source Mode:")
        sync_mode_label.setStyleSheet(WidgetStyles.label("small"))

        sync_mode_combo = QtWidgets.QComboBox()
        sync_mode_combo.setStyleSheet(WidgetStyles.combobox())
        sync_mode_combo.setMinimumWidth(200)

        sync_mode_row.addWidget(sync_mode_label)
        sync_mode_row.addWidget(sync_mode_combo, stretch=1)
        sync_vlayout.addLayout(sync_mode_row)

        # Contextual help explaining the sync mode relationship
        sync_mode_help = QtWidgets.QLabel(
            "This mode will use the decoder trained by the selected Synchronous mode.\n"
            "The decoder becomes available after the Sync mode completes training."
        )
        sync_mode_help.setStyleSheet(WidgetStyles.label("small", style="italic", color=TEXT_MUTED))
        sync_mode_help.setWordWrap(True)
        sync_vlayout.addWidget(sync_mode_help)

        # Warning when no sync modes exist
        sync_mode_warning = QtWidgets.QLabel("No Synchronous modes configured - create one first")
        sync_mode_warning.setStyleSheet(
            WidgetStyles.label("small", style="italic", color=WidgetStyles.colors["status_warn"])
        )
        sync_mode_warning.setVisible(False)
        sync_vlayout.addWidget(sync_mode_warning)

        sync_vlayout.addStretch()
        source_stack.addWidget(sync_container)  # Index 2

        self.widgets["sync_mode_combo"] = sync_mode_combo
        self.widgets["sync_mode_help"] = sync_mode_help
        self.widgets["sync_mode_warning"] = sync_mode_warning
        self.widgets["source_stack"] = source_stack

        source_layout.addWidget(source_stack)

        source_card_group.selection_changed.connect(self._on_source_changed)

        # Populate sync modes and connect to manager signals for reactive updates
        self._populate_sync_modes()
        self.mode_config_manager.instance_added.connect(self._on_sync_modes_changed)
        self.mode_config_manager.instance_removed.connect(self._on_sync_modes_changed)

        layout.addRow(decoder_source_group)

        # Store remaining source-related widgets
        self.widgets["source_card_group"] = source_card_group
        self.widgets["database_browser"] = database_browser

        # Initial visibility setup
        self._on_source_changed(source_card_group.get_selected() or DecoderSource.PRETRAINED)

        # Decoder Status Information
        model_status_group, decoder_status_label = ModeWidgetFactory.create_decoder_status_display(
            "Current Decoder Status"
        )
        model_status_group.setStyleSheet(WidgetStyles.groupbox())
        layout.addRow(model_status_group)
        self.widgets["decoder_status_label"] = decoder_status_label

        # Connect sync mode combo changes to status updates
        sync_mode_combo.currentTextChanged.connect(self._update_decoder_status_display)

        return tab

    def _get_default_source(self) -> str:
        """Async mode defaults to pretrained decoder."""
        return DecoderSource.PRETRAINED

    def _browse_pretrained_model(self) -> None:
        """Browse for pretrained model file."""
        parent = self.widgets.get("stacked_widget")
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Pre-trained Model", "", "Dendrite Model files (*.json);;All files (*.*)"
        )
        if file_path:
            self.widgets["pretrained_path_edit"].setText(file_path)
            self._update_decoder_status_display()

    def _on_database_decoder_selected(self, decoder: dict) -> None:
        """Handle decoder selection from inline browser."""
        decoder_path = decoder.get("decoder_path")

        # Validate decoder file exists
        if decoder_path and not os.path.exists(decoder_path):
            parent_widget = self.widgets.get("stacked_widget")
            QtWidgets.QMessageBox.warning(
                parent_widget,
                "Decoder File Missing",
                f"The decoder file was not found:\n{decoder_path}\n\n"
                "The file may have been moved or deleted.",
            )
            return

        # Store decoder info
        self._selected_database_decoder = decoder

        # Set the path for runtime loading
        if decoder_path and "pretrained_path_edit" in self.widgets:
            self.widgets["pretrained_path_edit"].setText(decoder_path)

        # Update status display
        self._update_decoder_status_display()

    def _on_source_changed(self, source_id: str) -> None:
        """Handle source card selection change."""
        # Switch to the appropriate panel in the stack
        stack = self.widgets.get("source_stack")
        if stack and source_id in self._SOURCE_INDEX:
            stack.setCurrentIndex(self._SOURCE_INDEX[source_id])

        # Show sync mode warning only when sync mode is selected and no sync modes exist
        if source_id == DecoderSource.SYNC_MODE:
            warning_label = self.widgets.get("sync_mode_warning")
            help_label = self.widgets.get("sync_mode_help")
            if warning_label and help_label:
                combo = self.widgets.get("sync_mode_combo")
                no_sync_modes = combo is not None and not combo.isEnabled()
                warning_label.setVisible(no_sync_modes)
                help_label.setVisible(not no_sync_modes)

        # Unlock config fields when switching away from pretrained/database
        if source_id not in (DecoderSource.PRETRAINED, DecoderSource.DATABASE):
            self.unlock_decoder_config()

        self._update_decoder_status_display()
        self.refresh_status_displays()

    def _update_decoder_status_display(self) -> None:
        """Handle sync_mode source, delegate others to base."""
        status_widget = self.widgets.get("decoder_status_label")
        if not status_widget:
            return

        source = self.widgets.get("source_card_group")
        source_id = source.get_selected() if source else DecoderSource.PRETRAINED

        if source_id == DecoderSource.SYNC_MODE:
            self._update_sync_mode_status(status_widget)
        else:
            super()._update_decoder_status_display()

    def _update_sync_mode_status(self, status_widget):
        """Update status for sync mode source."""
        combo = self.widgets.get("sync_mode_combo")

        if not combo:
            status_widget.set_error("Configuration Error", "Sync mode selection unavailable")
            return

        sync_name = combo.currentData()
        if sync_name is None:
            status_widget.set_empty("No Sync Modes - create one first or use File/Database")
            return

        sync_config = self.mode_config_manager.get_instance(sync_name)
        if not sync_config:
            status_widget.set_error(sync_name, "Sync mode config not found")
            return

        decoder_path = sync_config.get("decoder_config", {}).get("decoder_path", "")
        if not decoder_path:
            status_widget.set_info(
                f"Linked: {sync_name}", "No decoder yet - sync mode will train one"
            )
            return

        # Has decoder path - delegate to base class validation
        self._update_decoder_status(status_widget, decoder_path, sync_name)

    def _populate_sync_modes(self) -> None:
        """Populate sync mode combo with available Synchronous mode instances."""
        combo = self.widgets.get("sync_mode_combo")
        if not combo:
            return

        combo.clear()

        # Get all mode instances and filter for Synchronous types
        all_instances = self.mode_config_manager.get_all_instances()
        sync_modes = [
            (name, config)
            for name, config in all_instances.items()
            if config.get("mode_type") == "Synchronous"
        ]

        # Show/hide the warning label based on sync mode availability
        warning_label = self.widgets.get("sync_mode_warning")
        help_label = self.widgets.get("sync_mode_help")

        if not sync_modes:
            combo.addItem("No sync modes configured", None)
            combo.setEnabled(False)
            if warning_label:
                warning_label.setVisible(True)
            if help_label:
                help_label.setVisible(False)
        else:
            combo.setEnabled(True)
            if warning_label:
                warning_label.setVisible(False)
            if help_label:
                help_label.setVisible(True)
            for name, _ in sync_modes:
                combo.addItem(name, name)

    def _on_sync_modes_changed(self, *args):
        """Refresh sync mode combo when modes are added/removed."""
        self._populate_sync_modes()
        self._update_decoder_status_display()

    def _get_time_window_duration(self) -> float:
        """Get time window duration for asynchronous mode (sliding window)."""
        config = self.get_config_from_manager()
        return config.get("window_length_sec", 1.0)

    def create_events_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create events configuration tab"""
        return ModeWidgetFactory.create_events_tab_complete(
            config=config,
            parent_config=self,
            title="Event Mappings (Optional)",
            info_text=(
                "Event mappings are optional for asynchronous mode.\n"
                "They may be used for performance evaluation if available."
            ),
            include_time_offsets=False,
        )

    def get_config_data(self) -> dict[str, Any]:
        """Extract configuration data from widgets"""
        # Determine decoder source from card group
        decoder_source = (
            self.widgets["source_card_group"].get_selected() or DecoderSource.PRETRAINED
        )

        # Build decoder_config (nested structure matching sync mode)
        decoder_config = {
            "decoder_type": "Decoder",
        }

        # Add decoder path if pretrained or database mode
        if decoder_source == DecoderSource.PRETRAINED:
            decoder_config["decoder_path"] = self.widgets["pretrained_path_edit"].text()
        elif decoder_source == DecoderSource.DATABASE:
            if self._selected_database_decoder:
                decoder_config["decoder_path"] = self._selected_database_decoder.get(
                    "decoder_path", ""
                )
                decoder_config["decoder_id"] = self._selected_database_decoder.get("decoder_id")
            else:
                # No decoder selected - status display will show warning
                decoder_config["decoder_path"] = ""

        config = {
            "decoder_config": decoder_config,  # Nested like sync mode
            "step_size_ms": int(self.widgets["step_size_edit"].text() or 50),
            "decoder_source": decoder_source,
            "event_mapping": ModeWidgetFactory.get_event_mapping_from_table(
                self.widgets["events_table"]
            ),
        }

        # Get window_length_sec from cached decoder metadata or use default
        config["window_length_sec"] = getattr(self, "_decoder_window_sec", 1.0)

        # Add sync mode selection if sync mode is chosen
        if decoder_source == DecoderSource.SYNC_MODE:
            sync_mode_combo = self.widgets["sync_mode_combo"]
            sync_name = sync_mode_combo.currentData()
            if sync_name:  # Only set if a valid sync mode is selected
                config["source_sync_mode"] = sync_name

        return config

    def _update_widgets_from_config(self, data: dict[str, Any]):
        """Update widgets from configuration data"""
        # Block signals to prevent cascading updates during config loading
        with self._block_signals(
            [
                "source_card_group",
                "sync_mode_combo",
                "pretrained_path_edit",
                "step_size_edit",
            ]
        ):
            # Update step size (use default if not in config)
            if "step_size_edit" in self.widgets:
                step_size = data.get("step_size_ms", 50)  # Default to 50ms
                self.widgets["step_size_edit"].setText(str(step_size))

            # Restore window_length_sec if present
            if "window_length_sec" in data:
                self._decoder_window_sec = data["window_length_sec"]

            # Set decoder source from card group and update stack
            if "decoder_source" in data and "source_card_group" in self.widgets:
                source_id = data["decoder_source"]
                self.widgets["source_card_group"].set_selected(source_id)
                # Update stack directly since signals are blocked
                stack = self.widgets.get("source_stack")
                if stack and source_id in self._SOURCE_INDEX:
                    stack.setCurrentIndex(self._SOURCE_INDEX[source_id])

            # Set sync mode selection if present
            if "source_sync_mode" in data and "sync_mode_combo" in self.widgets:
                sync_mode_combo = self.widgets["sync_mode_combo"]
                sync_mode_index = sync_mode_combo.findData(data["source_sync_mode"])
                if sync_mode_index >= 0:
                    sync_mode_combo.setCurrentIndex(sync_mode_index)

            # Get decoder path from nested decoder_config
            decoder_config = data.get("decoder_config", {})
            decoder_path = decoder_config.get("decoder_path", "")

            # Set decoder path if present
            if decoder_path and "pretrained_path_edit" in self.widgets:
                self.widgets["pretrained_path_edit"].setText(decoder_path)

            # Restore database decoder selection if present
            decoder_id = decoder_config.get("decoder_id")
            if decoder_id is not None:
                self._restore_database_decoder(decoder_id, decoder_path)
                browser = self.widgets.get("database_browser")
                if browser:
                    browser.select_decoder_by_id(decoder_id)

        # Update model status display once after all changes
        self._update_decoder_status_display()
