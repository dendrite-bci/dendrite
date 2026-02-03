"""
Dialog classes for mode configuration.

Contains the main configuration dialog for mode instances.
"""

import copy
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.config.mode_config_manager import get_mode_config_manager
from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.design_tokens import SEPARATOR_SUBTLE
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.pill_navigation import PillNavigation
from dendrite.processing.modes.mode_schemas import validate_mode_config
from dendrite.utils.logger_central import get_logger

from .configs import ModeConfigRegistry


class ModeInstanceConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring mode instances."""

    def __init__(self, instance_name: str, available_decoders: list[str], parent=None):
        super().__init__(parent)
        self.mode_config_manager = get_mode_config_manager()
        manager_data = self.mode_config_manager.get_instance(instance_name)
        self.is_new_instance = manager_data is None

        if manager_data:
            self.instance_data = copy.deepcopy(manager_data)
        else:
            self.instance_data = {"name": instance_name, "mode": "Synchronous"}

        self.original_name = instance_name
        self.available_decoders = available_decoders
        self.parent_gui = parent
        self.mode_configs = {}
        self.name_changed = False

        self.setWindowTitle("Configure Mode Instance")
        self.setModal(True)
        self.resize(950, 620)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing"])

        # Name input (centered, full width)
        self.name_edit = QtWidgets.QLineEdit(self.instance_data.get("name", ""))
        self.name_edit.setStyleSheet(WidgetStyles.input(clean=True, size=18, weight="bold"))
        self.name_edit.setPlaceholderText("Instance Name")
        self.name_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.name_edit.textChanged.connect(self.on_name_changed)
        layout.addWidget(self.name_edit)

        # Mode selector (centered)
        mode_tabs = [(mode, mode) for mode in ModeConfigRegistry.get_modes()]
        self.mode_selector = PillNavigation(tabs=mode_tabs, size="large")
        self.mode_selector.set_current_tab(self.instance_data.get("mode", "Synchronous"))
        self.mode_selector.section_changed.connect(self.on_mode_changed)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addStretch()
        mode_row.addWidget(self.mode_selector)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        layout.addSpacing(LAYOUT["spacing_lg"])
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(
            f"background: transparent; border-top: 1px solid {SEPARATOR_SUBTLE}; max-height: 1px;"
        )
        layout.addWidget(sep)
        layout.addSpacing(LAYOUT["spacing_lg"])

        self.config_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.config_stack, stretch=1)

        instance_name = self.instance_data.get("name")
        for mode_name in ModeConfigRegistry.get_modes():
            mode_config = ModeConfigRegistry.create_config(
                mode_name, self.available_decoders, instance_name=instance_name
            )
            config_widget = mode_config.create_config_widget(self)
            self.mode_configs[mode_name] = mode_config
            self.config_stack.addWidget(config_widget)

        self.refresh_sync_mode_options()
        for mc in self.mode_configs.values():
            mc.set_config_data(self.instance_data)

        self.switch_to_mode(self.instance_data.get("mode", "Synchronous"))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(WidgetStyles.dialog_buttonbox)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def current_mode(self) -> str:
        """Get the currently selected mode name."""
        return self.mode_selector.tabs[self.mode_selector.current_index][0]

    def switch_to_mode(self, mode_name: str):
        if mode_name in self.mode_configs:
            idx = list(self.mode_configs.keys()).index(mode_name)
            self.config_stack.setCurrentIndex(idx)

    def on_name_changed(self):
        self.name_changed = True

    def on_mode_changed(self, mode_name: str):
        self.switch_to_mode(mode_name)
        if mode_name in self.mode_configs:
            new_name = self.mode_config_manager.generate_unique_name(
                mode_name, exclude_name=self.original_name
            )
            self.name_edit.setText(new_name)
            self.name_changed = True
        self.refresh_sync_mode_options()

    def get_instance_data(self) -> dict[str, Any]:
        """Collect all instance data including channel selections."""
        mode = self.current_mode
        data = {"name": self.name_edit.text(), "mode": mode}

        if mode in self.mode_configs:
            mc = self.mode_configs[mode]
            data.update(mc.get_config_data())

            if "channel_widget" in mc.widgets:
                channel_widget = mc.widgets["channel_widget"]
                data["channel_selection"] = channel_widget.get_config()
                data["required_modalities"] = channel_widget.get_required_modalities()
                data["stream_sources"] = channel_widget.get_stream_sources()
                data["modality_labels"] = channel_widget.get_modality_labels()
            else:
                data["channel_selection"] = self.instance_data.get("channel_selection", {})
                data["required_modalities"] = self.instance_data.get("required_modalities", ["eeg"])
                data["stream_sources"] = self.instance_data.get("stream_sources", {})
                data["modality_labels"] = self.instance_data.get("modality_labels", {})

        return data

    def validate_configuration(self) -> tuple[bool, list[str]]:
        config = self.get_instance_data()
        ctx = {"sample_rate": get_stream_config_manager().get_system_sample_rate()}
        valid, errors, _ = validate_mode_config(config, ctx)
        return valid, errors

    def accept(self):
        try:
            valid, errors = self.validate_configuration()
            if not valid:
                msg = "Configuration errors:\n\n" + "\n".join(f"• {e}" for e in errors)
                QtWidgets.QMessageBox.warning(self, "Configuration Error", msg)
                return

            final = self.get_instance_data()
            new_name = final.get("name")

            if self.is_new_instance:
                self.mode_config_manager.add_instance(new_name, final)
            elif self.original_name != new_name:
                self.mode_config_manager.rename_instance(self.original_name, new_name)
                self.mode_config_manager.update_instance(new_name, final)
            else:
                self.mode_config_manager.update_instance(new_name, final)

            super().accept()
        except Exception as e:
            get_logger().error(f"Failed to accept dialog: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n\n{e}")

    def refresh_sync_mode_options(self):
        if self.current_mode != "Asynchronous":
            return
        if "Asynchronous" not in self.mode_configs:
            return

        async_cfg = self.mode_configs["Asynchronous"]
        if "sync_mode_combo" not in async_cfg.widgets:
            return

        combo = async_cfg.widgets["sync_mode_combo"]
        current = combo.currentData()
        combo.clear()
        combo.addItem("Any Sync Mode", "any_sync_mode")

        current_name = self.instance_data.get("name", "")
        for name in self.mode_config_manager.get_all_instance_names():
            inst = self.mode_config_manager.get_instance(name)
            if inst and inst.get("mode", "").lower() == "synchronous" and name != current_name:
                combo.addItem(name, name)

        if current:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
