"""
Protocol Configuration Card

A reusable collapsible card for configuring output protocols.
Used by OutputWidget to create consistent LSL, Socket, ZMQ, and ROS2 sections.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    BG_INPUT,
    BG_PANEL,
    BORDER,
    PADDING,
    RADIUS,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_DISABLED,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.pill_selector import PillSelector
from dendrite.gui.widgets.common.toggle_pill import TogglePillWidget


@dataclass
class FieldConfig:
    """Configuration for a form field."""

    name: str
    label: str
    field_type: str  # 'text', 'spinbox', 'combo', 'pill'
    default: Any
    tooltip: str = ""
    options: list[str] = field(default_factory=list)  # For combo/pill
    min_val: int = 1
    max_val: int = 65535


class ProtocolConfigCard(QtWidgets.QFrame):
    """
    Collapsible protocol configuration card with enable/disable toggle.

    Features:
    - Header with checkbox, protocol name, and status label
    - Collapsible config frame (visible only when enabled)
    - Support for availability checking (ROS2, ZMQ)
    - Field validation via Pydantic schema
    """

    enabled_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        protocol_name: str,
        fields: list[FieldConfig],
        schema_class: type,
        default_config: dict[str, Any],
        available: bool = True,
        availability_msg: str = "",
        default_enabled: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.protocol_name = protocol_name
        self.fields = fields
        self.schema_class = schema_class
        self.available = available
        self.availability_msg = availability_msg

        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._connected = False
        self._setup_ui(default_enabled)

    def _setup_ui(self, default_enabled: bool):
        """Set up the card UI."""
        # Apply card styling - compact with subtle border
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 6px;
                margin: 2px 0px;
            }}
            QFrame QFrame {{
                border: none;
            }}
            QFrame QLineEdit {{
                border: none;
                background-color: {BG_PANEL};
                padding: 4px;
                border-radius: 4px;
            }}
            QFrame QSpinBox {{
                border: none;
                background-color: {BG_PANEL};
                padding: 4px;
                border-radius: 4px;
            }}
            QFrame QCheckBox {{ border: none; }}
            QFrame QLabel {{ border: none; }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with toggle and status
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(LAYOUT["spacing_sm"])

        self._toggle = TogglePillWidget(
            initial_state=default_enabled if self.available else False,
            show_label=False,
            width=28,
            height=14,
        )

        if self.available:
            self._toggle.toggled.connect(self._on_enabled_changed)
        else:
            self._toggle.setEnabled(False)
            self._toggle.setToolTip(f"Not available: {self.availability_msg}")

        header.addWidget(self._toggle)

        # Protocol name label
        self._name_label = QtWidgets.QLabel(self.protocol_name)
        self._name_label.setStyleSheet(WidgetStyles.label())
        header.addWidget(self._name_label)

        # Status label
        if self.available:
            status_text = "Ready" if default_enabled else "Disabled"
            status_color = STATUS_WARN if default_enabled else TEXT_DISABLED
            self._status_label = QtWidgets.QLabel(status_text)
            self._status_label.setStyleSheet(WidgetStyles.label("small", color=status_color))
        else:
            self._status_label = QtWidgets.QLabel(f"Not Available ({self.availability_msg})")
            self._status_label.setStyleSheet(
                WidgetStyles.label("small", color=STATUS_ERROR, weight="bold")
            )
            self._status_label.setToolTip(self.availability_msg)

        header.addWidget(self._status_label)
        header.addStretch()
        layout.addLayout(header)

        # Config frame (collapsible)
        self._config_frame = QtWidgets.QFrame()
        config_layout = QtWidgets.QFormLayout(self._config_frame)
        config_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        for field_cfg in self.fields:
            widget = self._create_field_widget(field_cfg)
            self._widgets[field_cfg.name] = widget

            label = QtWidgets.QLabel(field_cfg.label)
            label.setStyleSheet(WidgetStyles.label())
            config_layout.addRow(label, widget)

        self._config_frame.setVisible(default_enabled and self.available)
        layout.addWidget(self._config_frame)

    def _create_field_widget(self, field_cfg: FieldConfig) -> QtWidgets.QWidget:
        """Create the appropriate widget for a field."""
        if field_cfg.field_type == "text":
            widget = QtWidgets.QLineEdit(str(field_cfg.default))
            widget.setStyleSheet(WidgetStyles.input())
            if field_cfg.tooltip:
                widget.setToolTip(field_cfg.tooltip)
            widget.editingFinished.connect(lambda fc=field_cfg: self._validate_field(fc.name))
            return widget

        elif field_cfg.field_type == "spinbox":
            widget = QtWidgets.QSpinBox()
            widget.setStyleSheet(WidgetStyles.spinbox)
            widget.setRange(field_cfg.min_val, field_cfg.max_val)
            widget.setValue(int(field_cfg.default))
            if field_cfg.tooltip:
                widget.setToolTip(field_cfg.tooltip)
            widget.editingFinished.connect(lambda fc=field_cfg: self._validate_field(fc.name))
            return widget

        elif field_cfg.field_type == "combo":
            widget = QtWidgets.QComboBox()
            widget.setStyleSheet(WidgetStyles.combobox())
            widget.addItems(field_cfg.options)
            if field_cfg.default in field_cfg.options:
                widget.setCurrentText(str(field_cfg.default))
            if field_cfg.tooltip:
                widget.setToolTip(field_cfg.tooltip)
            return widget

        elif field_cfg.field_type == "pill":
            # Convert options list to (value, label, tooltip) tuples
            pill_options = [(opt, opt, "") for opt in field_cfg.options]
            widget = PillSelector(pill_options)
            if field_cfg.default in field_cfg.options:
                widget.set_current(str(field_cfg.default))
            if field_cfg.tooltip:
                widget.setToolTip(field_cfg.tooltip)
            return widget

        raise ValueError(f"Unknown field type: {field_cfg.field_type}")

    def _on_enabled_changed(self, enabled: bool):
        """Handle toggle state change."""
        self._config_frame.setVisible(enabled)
        self._update_status(connected=False)
        self.enabled_changed.emit(enabled)

    def _update_status(self, connected: bool = False):
        """Update the status label."""
        self._connected = connected
        if connected:
            self._status_label.setText("Connected")
            self._status_label.setStyleSheet(WidgetStyles.label("small", color=STATUS_OK))
        elif self.is_enabled():
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet(WidgetStyles.label("small", color=STATUS_WARN))
        else:
            self._status_label.setText("Disabled")
            self._status_label.setStyleSheet(WidgetStyles.label("small", color=TEXT_DISABLED))

    def _validate_field(self, field_name: str):
        """Validate field using Pydantic schema."""
        widget = self._widgets.get(field_name)
        if not widget:
            return

        try:
            config = self.get_config()
            self.schema_class(**config)
            self._mark_field_valid(widget)
        except ValidationError as e:
            for error in e.errors():
                if field_name in error["loc"]:
                    self._mark_field_invalid(widget, error["msg"])
                    return
            self._mark_field_valid(widget)
        except (ValueError, KeyError) as e:
            self._mark_field_invalid(widget, str(e) or "Invalid value")

    def _mark_field_valid(self, widget: QtWidgets.QWidget):
        """Remove error styling from field."""
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setStyleSheet(WidgetStyles.input())

    def _mark_field_invalid(self, widget: QtWidgets.QWidget, error_msg: str):
        """Add error styling to field."""
        if isinstance(widget, QtWidgets.QLineEdit):
            error_style = f"""
                QLineEdit {{
                    background-color: {BG_INPUT};
                    color: {TEXT_MAIN};
                    border: 2px solid {STATUS_ERROR};
                    padding: {PADDING["sm"]}px {PADDING["md"]}px;
                    border-radius: {RADIUS["sm"]}px;
                    min-height: 1.4em;
                }}
            """
            widget.setStyleSheet(error_style)
            widget.setToolTip(error_msg)

    # Public API

    def is_enabled(self) -> bool:
        """Check if protocol is enabled."""
        return self._toggle.isChecked()

    def set_enabled(self, enabled: bool):
        """Set whether protocol is enabled."""
        self._toggle.setChecked(enabled)

    def set_connected(self, connected: bool):
        """Set connection status."""
        self._update_status(connected)

    def get_config(self) -> dict[str, Any]:
        """Get current configuration values."""
        config = {}
        for field_cfg in self.fields:
            widget = self._widgets[field_cfg.name]
            if field_cfg.field_type == "text":
                config[field_cfg.name] = widget.text().strip()
            elif field_cfg.field_type == "spinbox":
                config[field_cfg.name] = widget.value()
            elif field_cfg.field_type == "combo":
                config[field_cfg.name] = widget.currentText()
            elif field_cfg.field_type == "pill":
                config[field_cfg.name] = widget.current
        return config

    def set_config(self, config: dict[str, Any]):
        """Load configuration values."""
        for field_cfg in self.fields:
            if field_cfg.name in config:
                widget = self._widgets[field_cfg.name]
                value = config[field_cfg.name]
                if field_cfg.field_type == "text":
                    widget.setText(str(value))
                elif field_cfg.field_type == "spinbox":
                    widget.setValue(int(value))
                elif field_cfg.field_type == "combo":
                    widget.setCurrentText(str(value))
                elif field_cfg.field_type == "pill":
                    widget.set_current(str(value))

    def validate(self) -> tuple[bool, list[str]]:
        """Validate current config against Pydantic schema."""
        try:
            config = self.get_config()
            self.schema_class(**config)
            return True, []
        except ValidationError as e:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return False, errors

    def get_widget(self, name: str) -> QtWidgets.QWidget | None:
        """Get a specific field widget by name."""
        return self._widgets.get(name)
