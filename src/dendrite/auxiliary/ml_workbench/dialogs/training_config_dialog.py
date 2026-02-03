"""Training configuration dialog for offline ML trainer.

Profile-based search configuration with search space visibility.
"""

from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    BG_INPUT,
    BORDER,
    FONT_SIZE,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.ml.search import (
    PROFILES,
    OptunaConfig,
    get_profile,
    get_search_space_description,
)


class TrainingConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring Optuna search profile.

    Provides three presets (Quick/Balanced/Full) with visibility
    into what parameters are being searched.
    """

    def __init__(self, current_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self._config = current_config
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        self.setWindowTitle("Training Configuration")
        self.setMinimumWidth(380)
        self.setStyleSheet(WidgetStyles.dialog())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        # Profile selection group
        profile_group = self._create_profile_group()
        layout.addWidget(profile_group)

        # Collapsible search space details
        details_group = self._create_details_group()
        layout.addWidget(details_group)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.setStyleSheet(WidgetStyles.dialog_buttonbox)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_profile_group(self) -> QtWidgets.QGroupBox:
        """Profile selection with radio buttons."""
        group = QtWidgets.QGroupBox("Search Profile")
        group.setStyleSheet(WidgetStyles.groupbox())

        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(LAYOUT["spacing_md"])

        self._profile_buttons: dict[str, QtWidgets.QRadioButton] = {}

        for profile_key, profile in PROFILES.items():
            # Radio button with profile name and trials
            radio = QtWidgets.QRadioButton(f"{profile['name']} ({profile['trials']} trials)")
            radio.setStyleSheet(f"""
                QRadioButton {{
                    color: {TEXT_MAIN};
                    font-size: {FONT_SIZE["md"]}px;
                    font-weight: 500;
                    spacing: 8px;
                }}
                QRadioButton::indicator {{
                    width: 16px;
                    height: 16px;
                }}
            """)
            radio.setProperty("profile_key", profile_key)
            radio.toggled.connect(self._on_profile_changed)

            # Description label
            desc_label = QtWidgets.QLabel(f"    {profile['description']}")
            desc_label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: {FONT_SIZE['sm']}px; margin-left: 24px; background-color: transparent;"
            )

            layout.addWidget(radio)
            layout.addWidget(desc_label)

            self._profile_buttons[profile_key] = radio

            # Add spacing between profiles (except last)
            if profile_key != "full":
                layout.addSpacing(LAYOUT["spacing_sm"])

        return group

    def _create_details_group(self) -> QtWidgets.QWidget:
        """Collapsible search space details."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle button
        self._details_toggle = QtWidgets.QPushButton("▶ What's being searched")
        self._details_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {TEXT_LABEL};
                font-size: {FONT_SIZE["sm"]}px;
                background: transparent;
                border: none;
                text-align: left;
                padding: 4px 0;
            }}
            QPushButton:hover {{
                color: {TEXT_MAIN};
            }}
        """)
        self._details_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._details_toggle.clicked.connect(self._toggle_details)
        layout.addWidget(self._details_toggle)

        # Details content (hidden by default)
        self._details_content = QtWidgets.QTextEdit()
        self._details_content.setReadOnly(True)
        self._details_content.setVisible(False)
        self._details_content.setMaximumHeight(150)
        self._details_content.setStyleSheet(f"""
            QTextEdit {{
                background: {BG_INPUT};
                border: 1px solid {BORDER};
                border-radius: 4px;
                color: {TEXT_MUTED};
                font-size: {FONT_SIZE["xs"]}px;
                padding: 8px;
            }}
        """)
        layout.addWidget(self._details_content)

        return container

    def _toggle_details(self):
        """Toggle visibility of search space details."""
        visible = not self._details_content.isVisible()
        self._details_content.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self._details_toggle.setText(f"{arrow} What's being searched")

    def _on_profile_changed(self, checked: bool):
        """Update details when profile changes."""
        if not checked:
            return

        # Find selected profile
        for key, radio in self._profile_buttons.items():
            if radio.isChecked():
                profile = get_profile(key)
                descriptions = get_search_space_description(profile["space"])
                self._details_content.setText("\n".join(f"• {d}" for d in descriptions))
                break

    def _load_config(self):
        """Load values from config."""
        profile_key = self._config.get("profile", "balanced")
        if profile_key in self._profile_buttons:
            self._profile_buttons[profile_key].setChecked(True)
        else:
            self._profile_buttons["balanced"].setChecked(True)

    def _get_selected_profile(self) -> str:
        """Get currently selected profile key."""
        for key, radio in self._profile_buttons.items():
            if radio.isChecked():
                return key
        return "balanced"

    # Public API
    def get_config(self) -> dict[str, Any]:
        """Get config as dict."""
        profile_key = self._get_selected_profile()
        profile = get_profile(profile_key)
        return {
            "profile": profile_key,
            "n_trials": profile["trials"],
        }

    def get_optuna_config(self) -> OptunaConfig:
        """Get OptunaConfig based on selected profile."""
        profile_key = self._get_selected_profile()
        profile = get_profile(profile_key)
        return OptunaConfig(
            n_trials=profile["trials"],
            direction="maximize",
        )

    def get_search_space(self) -> dict[str, Any]:
        """Get search space for selected profile."""
        profile_key = self._get_selected_profile()
        profile = get_profile(profile_key)
        return profile["space"].copy()
