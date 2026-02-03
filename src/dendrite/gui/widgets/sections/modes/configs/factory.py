"""
Widget factory for mode configuration components.

Provides standardized widgets for common configuration elements,
plus the mode configuration registry pattern.
"""

from typing import Any

from PyQt6 import QtWidgets

from dendrite.gui.config.stream_config_manager import get_stream_config_manager
from dendrite.gui.styles.design_tokens import TEXT_LABEL
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.sections.modes.channel_selection import ChannelSelectionWidget
from dendrite.gui.widgets.sections.modes.decoder_status import DecoderStatusWidget
from dendrite.ml.decoders.registry import get_decoder_entry

# Mode metadata for UI display
MODE_INFO: dict[str, dict[str, str]] = {
    "Synchronous": {
        "description": "Trial-based ERP analysis with cued events",
        "short": "Trial-based",
    },
    "Asynchronous": {"description": "Continuous real-time classification", "short": "Continuous"},
    "Neurofeedback": {"description": "Real-time frequency band feedback", "short": "Feedback"},
}


class ModeConfigRegistry:
    """Registry for mode configuration handlers."""

    _modes: dict[str, type] = {}

    @classmethod
    def register_mode(cls, mode_name: str, config_class: type):
        """Register a mode configuration class."""
        cls._modes[mode_name] = config_class

    @classmethod
    def get_modes(cls) -> list[str]:
        """Get list of registered mode names."""
        return list(cls._modes.keys())

    @classmethod
    def create_config(cls, mode_name: str, *args, **kwargs):
        """Create a mode configuration instance."""
        if mode_name not in cls._modes:
            raise ValueError(f"Unknown mode: {mode_name}")
        return cls._modes[mode_name](*args, **kwargs)

    @classmethod
    def is_registered(cls, mode_name: str) -> bool:
        """Check if a mode is registered."""
        return mode_name in cls._modes

    @classmethod
    def get_mode_info(cls, mode_name: str) -> dict[str, str]:
        """Get mode metadata (description, short name)."""
        return MODE_INFO.get(mode_name, {"description": "", "short": mode_name})


def get_decoder_display_name(decoder_name: str) -> str:
    """
    Generate display name with modality tags for a decoder.

    Args:
        decoder_name: Name of the decoder

    Returns:
        Display name with modality tag (e.g., "EEGNet [EEG Only]")
    """
    entry = get_decoder_entry(decoder_name)
    capabilities = entry.get("modalities", ["eeg"]) if entry else ["eeg"]

    # Create clean tag based on capabilities
    if "any" in [cap.lower() for cap in capabilities]:
        # Handle "primary + any" case (e.g., EEG + Any)
        primary = capabilities[0] if capabilities else "EEG"
        tag = f"[{primary.upper()} + Any]"
    elif len(capabilities) > 1:
        # For multi-modality, show all modalities connected with +
        if len(capabilities) > 4:
            # Too many to display, use Multi abbreviation
            tag = "[Multi]"
        else:
            # Show all modalities in uppercase connected with +
            tag = f"[{' + '.join([cap.upper() for cap in capabilities])}]"
    else:
        # Single modality
        tag = f"[{capabilities[0].upper()} Only]"

    return f"{decoder_name} {tag}"


class ModeWidgetFactory:
    """Factory for creating common mode configuration widgets."""

    @staticmethod
    def create_styled_form_layout(parent: QtWidgets.QWidget) -> QtWidgets.QFormLayout:
        """
        Create form layout with standard margins and spacing.

        Eliminates repeated boilerplate across all tab creation functions.

        Args:
            parent: Parent widget for the layout

        Returns:
            Configured QFormLayout with standard styling
        """
        layout = QtWidgets.QFormLayout(parent)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing"])
        layout.setVerticalSpacing(LAYOUT["spacing_lg"])
        layout.setHorizontalSpacing(LAYOUT["spacing_lg"])
        # Ensure fields expand on macOS (different default than Linux)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return layout

    @staticmethod
    def create_styled_groupbox(
        title: str, layout_type: str = "form"
    ) -> tuple[QtWidgets.QGroupBox, QtWidgets.QLayout]:
        """
        Create groupbox with standard styling and spacing.

        Eliminates repeated groupbox + layout setup pattern.

        Args:
            title: GroupBox title
            layout_type: Layout type - 'form', 'vbox', or 'hbox'

        Returns:
            Tuple of (group_box, layout) ready for widget addition
        """
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet(WidgetStyles.groupbox())

        if layout_type == "form":
            layout = QtWidgets.QFormLayout(group)
            layout.setVerticalSpacing(LAYOUT["spacing"])
            layout.setHorizontalSpacing(LAYOUT["spacing_lg"])
            layout.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
        elif layout_type == "hbox":
            layout = QtWidgets.QHBoxLayout(group)
            layout.setSpacing(LAYOUT["spacing"])
        else:  # vbox
            layout = QtWidgets.QVBoxLayout(group)
            layout.setSpacing(LAYOUT["spacing"])

        return group, layout

    @staticmethod
    def create_info_label(text: str) -> QtWidgets.QLabel:
        """Create standardized info/hint label with small italic styling."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(WidgetStyles.label("small", style="italic", color=TEXT_LABEL))
        label.setWordWrap(True)
        return label

    @staticmethod
    def create_events_table(
        event_mappings: list[dict[str, Any]] | None = None,
    ) -> QtWidgets.QTableWidget:
        """Create standardized events table."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Event ID", "Event Label"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(WidgetStyles.tablewidget)
        table.setMinimumHeight(100)
        table.setMaximumHeight(200)

        if event_mappings:
            for event in event_mappings:
                ModeWidgetFactory.add_event_row(
                    table,
                    event.get("event_id", ""),
                    event.get("event_label", ""),
                )
        else:
            ModeWidgetFactory.add_event_row(table)

        return table

    @staticmethod
    def add_event_row(
        table: QtWidgets.QTableWidget,
        event_id: str = "",
        event_label: str = "",
    ):
        """Add a row to an events table."""
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(event_id)))
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(event_label)))

    @staticmethod
    def remove_event_row(table: QtWidgets.QTableWidget):
        """Remove selected row from events table."""
        current_row = table.currentRow()
        if current_row >= 0:
            table.removeRow(current_row)

    @staticmethod
    def get_event_mapping_from_table(table: QtWidgets.QTableWidget) -> dict[int, str]:
        """Extract event mapping {event_id: event_label} from table."""
        event_mapping = {}
        for row in range(table.rowCount()):
            id_item = table.item(row, 0)
            label_item = table.item(row, 1)
            if id_item and label_item and id_item.text().strip() and label_item.text().strip():
                try:
                    event_id = int(id_item.text().strip())
                    event_label = label_item.text().strip()
                    event_mapping[event_id] = event_label
                except ValueError:
                    pass
        return event_mapping

    @staticmethod
    def create_channel_selection_widget(
        channel_selection: dict[str, Any], parent: QtWidgets.QWidget = None
    ) -> QtWidgets.QWidget:
        """
        Create a compact channel selection widget.

        Returns a widget that can be placed in a side panel or section.
        """
        channel_widget = ChannelSelectionWidget(parent=parent)
        manager = get_stream_config_manager()
        stream_data = manager.get_modalities_by_stream()
        channel_widget.update_content(stream_data=stream_data, current_config=channel_selection)

        return channel_widget

    @staticmethod
    def add_modality_section_to_form(
        layout: QtWidgets.QFormLayout,
        parent: QtWidgets.QWidget,
        mode_type: str,
        channel_selection: dict[str, Any],
        parent_widgets: dict[str, Any],
    ) -> None:
        """
        Add inline channel selection widget to a form layout.

        For better layout integration, consider using create_channel_selection_widget()
        directly and placing it in a custom layout.
        """
        channel_widget = ModeWidgetFactory.create_channel_selection_widget(
            channel_selection, parent
        )
        layout.addRow(channel_widget)
        parent_widgets["channel_widget"] = channel_widget

    @staticmethod
    def create_decoder_status_display(
        title: str = "Current Decoder Status",
    ) -> tuple[QtWidgets.QGroupBox, DecoderStatusWidget]:
        """
        Create standardized decoder status display with group box and status widget.

        Args:
            title: GroupBox title for the status section

        Returns:
            Tuple of (group_box, status_widget) ready for layout insertion
        """
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet(WidgetStyles.groupbox())
        layout = QtWidgets.QVBoxLayout(group)

        status_widget = DecoderStatusWidget()
        layout.addWidget(status_widget)

        return group, status_widget

    @staticmethod
    def create_events_tab_complete(
        config: dict[str, Any],
        parent_config,
        title: str = "Event Mappings",
        info_text: str | None = None,
        include_time_offsets: bool = False,
    ) -> QtWidgets.QWidget:
        """
        Create complete events configuration tab with optional time offsets.

        Unified implementation for all modes to eliminate duplication.

        Args:
            config: Mode configuration dictionary
            parent_config: Parent mode config instance (for callbacks and widget storage)
            title: GroupBox title for events section
            info_text: Custom info text, or None for default
            include_time_offsets: If True, adds time offset controls (sync mode)

        Returns:
            Configured tab widget with events table and optional time controls
        """
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )

        # Events table - convert event_mapping dict to list format for table
        event_mapping = config.get("event_mapping", {})
        events_list = [
            {"event_id": eid, "event_label": elabel} for eid, elabel in event_mapping.items()
        ]
        events_table = ModeWidgetFactory.create_events_table(events_list)

        events_group = QtWidgets.QGroupBox(title)
        events_group.setStyleSheet(WidgetStyles.groupbox())
        events_layout = QtWidgets.QVBoxLayout(events_group)
        events_layout.setSpacing(LAYOUT["spacing"])
        events_layout.addWidget(events_table)

        events_btn_layout = QtWidgets.QHBoxLayout()
        add_event_btn = QtWidgets.QPushButton("Add Event")
        add_event_btn.setStyleSheet(WidgetStyles.button(size="small"))
        add_event_btn.clicked.connect(lambda: ModeWidgetFactory.add_event_row(events_table))

        remove_event_btn = QtWidgets.QPushButton("Remove Event")
        remove_event_btn.setStyleSheet(WidgetStyles.button(size="small"))
        remove_event_btn.clicked.connect(lambda: ModeWidgetFactory.remove_event_row(events_table))

        events_btn_layout.addWidget(add_event_btn)
        events_btn_layout.addWidget(remove_event_btn)
        events_btn_layout.addStretch()
        events_layout.addLayout(events_btn_layout)

        layout.addWidget(events_group)

        if include_time_offsets:
            time_group = QtWidgets.QGroupBox("Event Time Windows")
            time_group.setStyleSheet(WidgetStyles.groupbox())
            time_layout = QtWidgets.QFormLayout(time_group)
            time_layout.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            time_layout.setVerticalSpacing(LAYOUT["spacing_lg"])
            time_layout.setHorizontalSpacing(LAYOUT["spacing_xl"])

            start_edit = QtWidgets.QLineEdit(str(config.get("start_offset", 0.0)))
            start_edit.setStyleSheet(WidgetStyles.input())
            end_edit = QtWidgets.QLineEdit(str(config.get("end_offset", 2.0)))
            end_edit.setStyleSheet(WidgetStyles.input())

            start_label = QtWidgets.QLabel("Start Offset (s):")
            start_label.setStyleSheet(WidgetStyles.label("small"))
            end_label = QtWidgets.QLabel("End Offset (s):")
            end_label.setStyleSheet(WidgetStyles.label("small"))

            time_layout.addRow(start_label, start_edit)
            time_layout.addRow(end_label, end_edit)

            layout.addWidget(time_group)
            parent_config.widgets["start_edit"] = start_edit
            parent_config.widgets["end_edit"] = end_edit

        if info_text is None:
            info_text = (
                "Define event IDs and their labels for classification.\n"
                "Events are used to trigger data collection and analysis."
            )

        events_info = QtWidgets.QLabel(info_text)
        events_info.setStyleSheet(WidgetStyles.label("small", style="italic", color=TEXT_LABEL))
        events_info.setWordWrap(True)
        layout.addWidget(events_info)

        parent_config.widgets["events_table"] = events_table

        return tab
