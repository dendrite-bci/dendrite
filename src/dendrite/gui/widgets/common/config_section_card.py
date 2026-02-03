"""
Config Section Card

A styled configuration section card for modality-specific settings.
Used by PreprocessingWidget to create consistent EEG, EMG, EOG sections.
"""

from PyQt6 import QtWidgets

from dendrite.gui.styles.design_tokens import BG_PANEL
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class ConfigSectionCard(QtWidgets.QFrame):
    """
    Styled configuration section with header and form layout.

    Used for modality-specific preprocessing configs (EEG, EMG, EOG)
    and other grouped configuration sections.
    """

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI with consistent styling."""
        # Apply flat telemetry-style section styling (elevated for contrast)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_PANEL};
                border: none;
                border-radius: 4px;
                padding: 8px;
                margin: 2px 0px;
            }}
            QFrame QLineEdit {{
                border: none;
                background-color: {BG_PANEL};
                padding: 6px;
                border-radius: 4px;
            }}
            QFrame QCheckBox {{ border: none; }}
            QFrame QLabel {{ border: none; }}
        """)

        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.setContentsMargins(
            LAYOUT["spacing_sm"], LAYOUT["spacing_sm"], LAYOUT["spacing_sm"], LAYOUT["spacing_sm"]
        )

        header_layout = QtWidgets.QHBoxLayout()

        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet(
            WidgetStyles.label("small", weight="semibold", padding="2px 0px 4px 0px")
        )
        header_layout.addWidget(title_label)

        if self.subtitle:
            info_label = QtWidgets.QLabel(f"({self.subtitle})")
            info_label.setStyleSheet(
                WidgetStyles.label("tiny", style="italic", padding="4px 2px", line_height=1.5)
            )
            header_layout.addWidget(info_label)

        header_layout.addStretch()
        self._main_layout.addLayout(header_layout)

        # Form layout for fields
        self._form_layout = QtWidgets.QFormLayout()
        self._form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._main_layout.addLayout(self._form_layout)

    def add_field(self, label: str, widget: QtWidgets.QWidget) -> None:
        """Add a field to the form layout."""
        label_widget = QtWidgets.QLabel(label)
        label_widget.setStyleSheet(WidgetStyles.label())
        self._form_layout.addRow(label_widget, widget)

    def add_separator(self) -> None:
        """Add a horizontal separator line."""
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet(WidgetStyles.separator())
        self._form_layout.addRow(separator)

    def add_checkbox(
        self, text: str, checked: bool = False, color: str | None = None
    ) -> QtWidgets.QCheckBox:
        """Add a checkbox row and return it."""
        checkbox = QtWidgets.QCheckBox(text)
        style = WidgetStyles.checkbox_colored(color) if color else WidgetStyles.checkbox
        checkbox.setStyleSheet(style)
        checkbox.setChecked(checked)
        self._form_layout.addRow(checkbox)
        return checkbox

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        """Add a custom widget spanning the full width."""
        self._form_layout.addRow(widget)

    def get_form_layout(self) -> QtWidgets.QFormLayout:
        """Get the form layout for custom additions."""
        return self._form_layout

    def set_subtitle(self, subtitle: str) -> None:
        """Update the subtitle text."""
        self.subtitle = subtitle
        # Find and update the info label if it exists
        header = self._main_layout.itemAt(0)
        if header and header.layout():
            for i in range(header.layout().count()):
                widget = header.layout().itemAt(i).widget()
                if widget and isinstance(widget, QtWidgets.QLabel):
                    if widget.text().startswith("("):
                        widget.setText(f"({subtitle})")
                        return
