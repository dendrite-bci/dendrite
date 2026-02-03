"""
Mode Instance Badge Widget

A clean, modern card-like widget for displaying Dendrite mode instances
with status indicators and interactive configuration.
"""

import copy
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.constants import MODE_ASYNCHRONOUS, MODE_NEUROFEEDBACK
from dendrite.gui.config.mode_config_manager import get_mode_config_manager
from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_ELEVATED,
    BG_INPUT,
    BUTTON_HOVER_SUBTLE,
    STATUS_OK,
    STATUS_WARN,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.utils import load_icon
from dendrite.gui.widgets.common import StatusPillWidget
from dendrite.processing.modes.mode_schemas import validate_mode_config

CONFIG_KEY_SYNC_SOURCE = "sync_source"
CONFIG_VALUE_ANY_SYNC = "any_sync_mode"
CONFIG_VALUE_SYNC_MODE = "sync_mode"

STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"
STATUS_WARNING = "warning"
STATUS_LINKING = "linking"
STATUS_STOPPED = "stopped"

# Status color mapping: status -> (color, active_state)
# For STATUS_READY, depends on _enabled state so handled separately
STATUS_COLORS = {
    STATUS_RUNNING: (STATUS_OK, True),
    STATUS_ERROR: ("#dc3545", True),
    STATUS_WARNING: (TEXT_DISABLED, False),
    STATUS_LINKING: (STATUS_WARN, True),
    STATUS_STOPPED: (TEXT_DISABLED, False),
}

BADGE_HEIGHT = 96  # Room for 2 rows + margins + button padding
BADGE_HEIGHT_COMPACT = 48  # 24px buttons + 24px margins
BADGE_MIN_WIDTH = 220
REMOVE_BUTTON_SIZE = 24


class ModeInstanceBadge(QtWidgets.QFrame):
    """Modern card-like widget for displaying mode instance information."""

    def __init__(self, instance_name: str, parent=None, compact: bool = False):
        super().__init__(parent)
        self.instance_name = instance_name
        self.parent_window = parent
        self.status = STATUS_READY
        self._enabled = False
        self._config_valid = True
        self._compact = compact  # Initialize with provided value

        self._mode_config_manager = get_mode_config_manager()
        self._mode_config_manager.instance_updated.connect(self._on_instance_updated)
        self._mode_config_manager.instance_renamed.connect(self._on_instance_renamed)

        self._stream_config_manager = get_stream_config_manager()
        self._stream_config_manager.streams_updated.connect(self._on_streams_changed)

        self.setup_ui()
        self.apply_styles()
        self.update_summary()
        self._update_validation_indicator()

        self.status_indicator.clicked.connect(self._toggle_enabled)

    @property
    def instance_data(self) -> dict[str, Any]:
        """Get instance data from manager."""
        return self._mode_config_manager.get_instance(self.instance_name) or {}

    def _on_instance_updated(self, updated_name: str, config: dict):
        """React to manager updates for this instance."""
        if updated_name == self.instance_name:
            self.update_summary()

    def _on_instance_renamed(self, old_name: str, new_name: str):
        """React to instance rename."""
        if old_name == self.instance_name:
            self.instance_name = new_name
            self.update_summary()

    def _on_streams_changed(self, streams: dict):
        """Re-validate when stream config changes (decoder compatibility may break)."""
        self._update_validation_indicator()

    def setup_ui(self):
        """Set up the user interface components."""
        self.setFixedHeight(BADGE_HEIGHT)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        # Expand horizontally to fill available space
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(
            LAYOUT["spacing_xl"],  # left: 16
            LAYOUT["padding_lg"],  # top: 12
            LAYOUT["spacing_xl"],  # right: 16
            LAYOUT["padding_lg"],  # bottom: 12
        )
        self.main_layout.setSpacing(LAYOUT["spacing_lg"])

        header_layout = self._create_header()
        self.main_layout.addLayout(header_layout)

        details_widget = self._create_details()
        self.main_layout.addWidget(details_widget)

        self.event_mapping_label = QtWidgets.QLabel()
        self.event_mapping_label.setStyleSheet(
            WidgetStyles.label(
                "tiny",
                style="italic",
                bg=BG_INPUT,
                padding="8px 10px",
                margin="2px 0px",
                line_height=1.5,
            )
        )
        self.event_mapping_label.setWordWrap(True)
        self.event_mapping_label.hide()
        self.main_layout.addWidget(self.event_mapping_label)

        if self._compact:
            self._apply_compact_layout()

    def _create_header(self) -> QtWidgets.QHBoxLayout:
        """Create the header with status pill and name."""
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setSpacing(LAYOUT["spacing"])

        self.status_indicator = StatusPillWidget()
        self.header_layout.addWidget(self.status_indicator)

        instance_name = self.instance_data.get("name", "Unnamed")
        self.name_label = QtWidgets.QLabel(instance_name)
        self.name_label.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 13px; font-weight: 600;")
        self.name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.header_layout.addWidget(self.name_label)

        self.link_label = QtWidgets.QLabel()
        self.link_label.setStyleSheet(f"color: {STATUS_WARN}; font-size: 11px;")
        self.link_label.setMaximumWidth(120)
        self.link_label.hide()
        self.header_layout.addWidget(self.link_label)

        self.header_layout.addStretch()

        return self.header_layout

    def _create_details(self) -> QtWidgets.QWidget:
        """Create the details row: MODE TYPE | Detail | Buttons."""
        self.details_widget = QtWidgets.QWidget()
        self.details_layout = QtWidgets.QHBoxLayout(self.details_widget)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(LAYOUT["spacing"])

        mode_name = self.instance_data.get("mode", "Unknown")
        self.mode_type_label = QtWidgets.QLabel(mode_name.upper())
        self.mode_type_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        self.details_layout.addWidget(self.mode_type_label)

        self.separator = QtWidgets.QLabel("·")
        self.separator.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: 12px;")
        self.details_layout.addWidget(self.separator)

        self.detail_label = QtWidgets.QLabel()
        self.detail_label.setStyleSheet(f"color: {TEXT_LABEL}; font-size: 11px;")
        self.details_layout.addWidget(self.detail_label)

        self.details_layout.addStretch()

        self.configure_button = QtWidgets.QPushButton("Configure")
        self.configure_button.setMinimumHeight(REMOVE_BUTTON_SIZE)
        self.configure_button.setStyleSheet(WidgetStyles.button(size="small", transparent=True))
        self.details_layout.addWidget(self.configure_button)

        self.clone_button = QtWidgets.QPushButton()
        self.clone_button.setToolTip("Clone this mode")
        self.clone_button.setFixedSize(REMOVE_BUTTON_SIZE, REMOVE_BUTTON_SIZE)
        self.clone_button.setStyleSheet(
            WidgetStyles.button(
                variant="icon", fixed_size=REMOVE_BUTTON_SIZE, min_width=REMOVE_BUTTON_SIZE
            )
        )
        clone_icon = load_icon("icons/copy.svg")
        if not clone_icon.isNull():
            self.clone_button.setIcon(clone_icon)
            self.clone_button.setIconSize(QtCore.QSize(14, 14))
        self.clone_button.clicked.connect(self._clone_mode)
        self.details_layout.addWidget(self.clone_button)

        self.remove_button = QtWidgets.QPushButton()
        self.remove_button.setToolTip("Remove this mode")
        self.remove_button.setFixedSize(REMOVE_BUTTON_SIZE, REMOVE_BUTTON_SIZE)
        self.remove_button.setStyleSheet(
            WidgetStyles.button(
                variant="icon", fixed_size=REMOVE_BUTTON_SIZE, min_width=REMOVE_BUTTON_SIZE
            )
        )
        remove_icon = load_icon("icons/x.svg")
        if not remove_icon.isNull():
            self.remove_button.setIcon(remove_icon)
            self.remove_button.setIconSize(QtCore.QSize(14, 14))
        self.details_layout.addWidget(self.remove_button)

        return self.details_widget

    def _get_status_color_and_state(self, status: str | None = None) -> tuple[str, bool]:
        """Get status indicator color and active state based on status."""
        current_status = status or self.status

        # STATUS_READY depends on _enabled state
        if current_status == STATUS_READY:
            if self._enabled:
                return STATUS_WARN, True
            return TEXT_DISABLED, False

        return STATUS_COLORS.get(current_status, (TEXT_DISABLED, False))

    def _validate_instance_config(self) -> tuple[bool, list[str]]:
        """Validate instance configuration using Pydantic schemas and decoder compatibility."""
        stream_context = {"sample_rate": self._stream_config_manager.get_system_sample_rate()}
        return validate_mode_config(self.instance_data, stream_context)[:2]

    def _update_validation_indicator(self) -> None:
        """Update visual indicators based on configuration validity."""
        is_valid, errors = self._validate_instance_config()
        self._config_valid = is_valid

        if not is_valid:
            self.set_status(STATUS_WARNING)
            error_text = "Configuration incomplete:\n" + "\n".join(f"• {err}" for err in errors[:5])
            if len(errors) > 5:
                error_text += f"\n...and {len(errors) - 5} more issues"
            self.setToolTip(error_text)

            if self._enabled:
                self._enabled = False
                self._update_pill_visual()
        else:
            self.setToolTip("")

            if self.status == STATUS_WARNING:
                self.set_status(STATUS_READY)

    def apply_styles(self):
        """Apply modern card-like styling."""
        self.setObjectName("mode_badge")
        self.setStyleSheet(
            WidgetStyles.frame(
                id_selector="mode_badge",
                bg=BG_ELEVATED,
                border=True,
                border_color=BUTTON_HOVER_SUBTLE,
                radius=LAYOUT["radius_lg"],
                padding=LAYOUT["padding_md"],
                hover_bg="#363636",
            )
        )

    def update_summary(self):
        """Update the display with current instance data."""
        instance_name = self.instance_data.get("name", "Unnamed")
        self.name_label.setText(instance_name)

        mode_type = self.instance_data.get("mode", "Unknown")
        self.mode_type_label.setText(mode_type.upper())

        if mode_type.lower() == MODE_NEUROFEEDBACK:
            feature_config = self.instance_data.get("feature_config", {})
            if "target_bands" in feature_config:
                band_count = len(feature_config["target_bands"])
                self.detail_label.setText(f"{band_count} target bands")
            elif "target_band" in feature_config:
                target_band = feature_config["target_band"]
                self.detail_label.setText(f"Band: {target_band[0]}-{target_band[1]} Hz")
            else:
                self.detail_label.setText("Default: 8-12 Hz")
        else:
            decoder_config = self.instance_data.get("decoder_config", {})
            model_config = decoder_config.get("model_config", {})
            model_type = model_config.get("model_type")

            if model_type:
                self.detail_label.setText(model_type)
            else:
                self.detail_label.setText("Not configured")

        if self._is_linked_to_sync():
            link_target = self._get_link_target()
            self.link_label.setText(f"→ {link_target}")
            self.link_label.show()
        else:
            self.link_label.hide()

        self._update_event_mapping_display()
        self._update_validation_indicator()

    def _update_event_mapping_display(self):
        """Update the event mapping preview."""
        event_mappings = self.instance_data.get("event_mappings", {})

        if event_mappings:
            mapping_text = f"Events: {', '.join(event_mappings.values())}"
            self.event_mapping_label.setText(mapping_text)
            self.event_mapping_label.show()
        else:
            self.event_mapping_label.hide()

    def _is_linked_to_sync(self) -> bool:
        """Check if this instance is linked to a synchronous mode."""
        # Check for async mode decoder sharing
        if self.instance_data.get("mode", "").lower() == MODE_ASYNCHRONOUS:
            decoder_source = self.instance_data.get("decoder_source", "")
            source_sync_mode = self.instance_data.get("source_sync_mode", "")
            return decoder_source == CONFIG_VALUE_SYNC_MODE and source_sync_mode

        # Legacy sync_source check
        sync_source = self.instance_data.get(CONFIG_KEY_SYNC_SOURCE)
        return sync_source is not None and sync_source != self.instance_data.get("name")

    def _get_link_target(self) -> str:
        """Get the name of the linked synchronous instance."""
        # Check for async mode model sharing
        if self.instance_data.get("mode", "").lower() == MODE_ASYNCHRONOUS:
            source_sync_mode = self.instance_data.get("source_sync_mode", "")
            if source_sync_mode == CONFIG_VALUE_ANY_SYNC:
                return "Any Sync"
            elif source_sync_mode:
                return source_sync_mode

        # Legacy sync_source check
        return self.instance_data.get(CONFIG_KEY_SYNC_SOURCE, "Unknown")

    def set_status(self, status: str):
        """Set the status indicator."""
        self.status = status
        color, active = self._get_status_color_and_state(status)
        self.status_indicator.set_status(color, active)

    def update_name(self, new_name: str):
        """Update the display name (called reactively via signals)."""
        self.instance_name = new_name
        self.name_label.setText(new_name)

    def is_enabled(self) -> bool:
        """Check if the instance is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool):
        """Set the enabled state programmatically."""
        if enabled and not self._config_valid:
            return  # Don't allow enabling invalid configurations
        self._enabled = enabled
        self._update_pill_visual()

    def _toggle_enabled(self):
        """Toggle the enabled state when pill is clicked."""
        if not self._config_valid:
            # Animate to "on" then back to "off" as rejection feedback
            self.status_indicator.animate_reject()
            return

        # Don't allow toggling while mode is running
        if self.status == STATUS_RUNNING:
            self.status_indicator.animate_reject(from_active=True)
            return

        self._enabled = not self._enabled
        self._update_pill_visual()

    def _update_pill_visual(self):
        """Update the pill visual based on current enabled state."""
        color, active = self._get_status_color_and_state()
        self.status_indicator.set_status(color, active)

    def get_linked_sync_instance(self) -> str:
        """Get the linked synchronous instance name."""
        # Check for async mode model sharing
        if self.instance_data.get("mode", "").lower() == MODE_ASYNCHRONOUS:
            source_sync_mode = self.instance_data.get("source_sync_mode", "")
            return (
                source_sync_mode
                if source_sync_mode and source_sync_mode != CONFIG_VALUE_ANY_SYNC
                else ""
            )

        # Legacy sync_source check
        return self.instance_data.get(CONFIG_KEY_SYNC_SOURCE, "")

    def _clone_mode(self):
        """Create a copy of this mode instance with a new unique name."""
        base_name = self.instance_name
        if "_" in base_name:
            parts = base_name.rsplit("_", 1)
            if parts[1].isdigit():
                base_name = parts[0]

        new_name = self._mode_config_manager.generate_unique_name(base_name)

        config_copy = copy.deepcopy(self.instance_data)
        config_copy["name"] = new_name

        self._mode_config_manager.add_instance(new_name, config_copy)

    def _apply_compact_layout(self):
        """Apply compact layout: move buttons to header, reduce margins."""
        self.main_layout.setContentsMargins(
            LAYOUT["spacing"], LAYOUT["spacing_xs"], LAYOUT["spacing"], LAYOUT["spacing_xs"]
        )
        self.details_layout.removeWidget(self.configure_button)
        self.details_layout.removeWidget(self.clone_button)
        self.details_layout.removeWidget(self.remove_button)
        self.header_layout.addWidget(self.configure_button)
        self.header_layout.addWidget(self.clone_button)
        self.header_layout.addWidget(self.remove_button)
        self.configure_button.setText("")
        settings_icon = load_icon("icons/settings.svg")
        if not settings_icon.isNull():
            self.configure_button.setIcon(settings_icon)
            self.configure_button.setIconSize(QtCore.QSize(14, 14))
        self.configure_button.setFixedSize(REMOVE_BUTTON_SIZE, REMOVE_BUTTON_SIZE)
        self.details_widget.hide()
        self.event_mapping_label.hide()
        self.setFixedHeight(BADGE_HEIGHT_COMPACT)

    def _apply_full_layout(self):
        """Apply full card layout: buttons in details row, normal margins."""
        self.main_layout.setContentsMargins(
            LAYOUT["spacing_xl"], LAYOUT["padding_lg"], LAYOUT["spacing_xl"], LAYOUT["padding_lg"]
        )
        self.header_layout.removeWidget(self.configure_button)
        self.header_layout.removeWidget(self.clone_button)
        self.header_layout.removeWidget(self.remove_button)
        self.details_layout.addWidget(self.configure_button)
        self.details_layout.addWidget(self.clone_button)
        self.details_layout.addWidget(self.remove_button)
        self.configure_button.setText("Configure")
        self.configure_button.setIcon(QtGui.QIcon())  # Clear icon
        self.configure_button.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        self.configure_button.setMinimumHeight(REMOVE_BUTTON_SIZE)
        self.details_widget.show()
        self.event_mapping_label.setVisible(bool(self.instance_data.get("event_mappings")))
        self.setFixedHeight(BADGE_HEIGHT)

    def set_compact(self, compact: bool):
        """Toggle between card view and compact list view."""
        if self._compact == compact:
            return
        self._compact = compact

        if compact:
            self._apply_compact_layout()
        else:
            self._apply_full_layout()
