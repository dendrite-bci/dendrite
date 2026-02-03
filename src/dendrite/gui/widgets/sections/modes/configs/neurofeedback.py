"""
Neurofeedback mode configuration widget.

Contains configuration for neurofeedback mode including window parameters
and frequency band configuration.
"""

from typing import Any

from PyQt6 import QtWidgets

from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import PillSelector

from .base import BaseModeConfig, DecoderSource
from .factory import ModeWidgetFactory


class NeurofeedbackModeConfig(BaseModeConfig):
    """Configuration for Neurofeedback mode"""

    def create_config_widget(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        # Define tabs
        tabs = [("general", "General"), ("features", "Features")]

        # Create standard pill navigation container
        page, stacked_widget, pill_nav = self._create_standard_tab_widget_container(parent, tabs)

        # Cache config once for all tab creation methods
        config = self.get_config_from_manager()

        # Add mode-specific tab widgets
        stacked_widget.addWidget(self.create_general_settings_tab(config))
        stacked_widget.addWidget(self.create_feature_config_tab(config))

        return page

    def create_general_settings_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create general settings tab with two-column layout."""
        tab, right_layout = self._create_general_tab_with_channels(config)

        # Window parameters
        window_group = QtWidgets.QGroupBox("Window Parameters")
        window_group.setStyleSheet(WidgetStyles.groupbox())
        window_layout = QtWidgets.QFormLayout(window_group)
        window_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        window_layout.setVerticalSpacing(LAYOUT["spacing_lg"])

        window_length_edit = QtWidgets.QLineEdit(str(config.get("window_length_sec", 1.0)))
        window_length_edit.setStyleSheet(WidgetStyles.input())
        window_step_edit = QtWidgets.QLineEdit(str(config.get("step_size_ms", 250)))
        window_step_edit.setStyleSheet(WidgetStyles.input())

        window_length_label = QtWidgets.QLabel("Window Length (s):")
        window_length_label.setStyleSheet(WidgetStyles.label("small"))
        window_step_label = QtWidgets.QLabel("Step Size (ms):")
        window_step_label.setStyleSheet(WidgetStyles.label("small"))

        window_layout.addRow(window_length_label, window_length_edit)
        window_layout.addRow(window_step_label, window_step_edit)

        right_layout.addWidget(window_group)
        self.widgets["window_length_edit"] = window_length_edit
        self.widgets["window_step_edit"] = window_step_edit

        # Create horizontal container for both selectors
        feature_config = config.get("feature_config", {})
        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setSpacing(LAYOUT["spacing"])

        # Power calculation method
        power_group = QtWidgets.QGroupBox("Power Calculation")
        power_group.setStyleSheet(WidgetStyles.groupbox())
        power_layout = QtWidgets.QVBoxLayout(power_group)

        power_type_selector = PillSelector(
            [
                ("relative", "Relative", "Band power as % of total power"),
                ("absolute", "Absolute", "Raw band power in µV²/Hz"),
            ]
        )
        initial_power = "relative" if feature_config.get("use_relative_power", True) else "absolute"
        power_type_selector.set_current(initial_power)
        power_layout.addWidget(power_type_selector)

        selector_row.addWidget(power_group)

        # Cluster mode option
        cluster_group = QtWidgets.QGroupBox("Channel Output")
        cluster_group.setStyleSheet(WidgetStyles.groupbox())
        cluster_layout = QtWidgets.QVBoxLayout(cluster_group)

        cluster_mode_selector = PillSelector(
            [
                ("individual", "Individual", "Output each channel separately"),
                ("clustered", "Clustered", "Average channels together"),
            ]
        )
        initial_cluster = (
            "clustered" if feature_config.get("use_cluster_mode", False) else "individual"
        )
        cluster_mode_selector.set_current(initial_cluster)
        cluster_layout.addWidget(cluster_mode_selector)

        selector_row.addWidget(cluster_group)

        right_layout.addLayout(selector_row)
        right_layout.addStretch()

        self.widgets["power_type_selector"] = power_type_selector
        self.widgets["cluster_mode_selector"] = cluster_mode_selector

        return tab

    def create_feature_config_tab(self, config: dict[str, Any]) -> QtWidgets.QWidget:
        """Create feature configuration tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
            LAYOUT["dialog_margin"],
        )

        # Target bands table
        feature_config = config.get("feature_config", {})
        bands_table = self.create_bands_table(feature_config)

        bands_group = QtWidgets.QGroupBox("Target Frequency Bands")
        bands_group.setStyleSheet(WidgetStyles.groupbox())
        bands_layout = QtWidgets.QVBoxLayout(bands_group)
        bands_layout.addWidget(bands_table)

        # Band buttons
        bands_btn_layout = QtWidgets.QHBoxLayout()
        add_band_btn = QtWidgets.QPushButton("Add Band")
        add_band_btn.setStyleSheet(WidgetStyles.button(size="small"))
        add_band_btn.clicked.connect(lambda: self.add_band_row(bands_table))

        remove_band_btn = QtWidgets.QPushButton("Remove Band")
        remove_band_btn.setStyleSheet(WidgetStyles.button(size="small"))
        remove_band_btn.clicked.connect(lambda: self.remove_band_row(bands_table))

        bands_btn_layout.addWidget(add_band_btn)
        bands_btn_layout.addWidget(remove_band_btn)
        bands_btn_layout.addStretch()
        bands_layout.addLayout(bands_btn_layout)

        layout.addWidget(bands_group)

        # Bands description
        bands_info = ModeWidgetFactory.create_info_label(
            "Define frequency bands for neurofeedback training.\n"
            "Multiple bands can be targeted simultaneously for comprehensive feedback."
        )
        layout.addWidget(bands_info)

        layout.addStretch()

        self.widgets["bands_table"] = bands_table

        return tab

    def create_bands_table(self, feature_config: dict[str, Any]) -> QtWidgets.QTableWidget:
        """Create table for neurofeedback target bands"""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Band Name", "Low (Hz)", "High (Hz)"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(WidgetStyles.tablewidget)
        table.setMaximumHeight(150)

        # Populate from existing config
        if "target_bands" in feature_config:
            for band_name, band_range in feature_config["target_bands"].items():
                self.add_band_row(table, band_name, str(band_range[0]), str(band_range[1]))
        elif "target_band" in feature_config:
            target_band = feature_config["target_band"]
            self.add_band_row(table, "alpha", str(target_band[0]), str(target_band[1]))
        else:
            self.add_band_row(table, "alpha", "8.0", "12.0")

        return table

    def add_band_row(
        self,
        table: QtWidgets.QTableWidget,
        band_name: str = "",
        low_freq: str = "8.0",
        high_freq: str = "12.0",
    ):
        """Add a new band row to the bands table"""
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(band_name)))
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(low_freq)))
        table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(high_freq)))

    def remove_band_row(self, table: QtWidgets.QTableWidget):
        """Remove the selected band row from the bands table"""
        current_row = table.currentRow()
        if current_row >= 0:
            table.removeRow(current_row)

    def get_bands_from_table(self, table: QtWidgets.QTableWidget) -> dict[str, Any]:
        """Extract band configuration from the bands table"""
        bands = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            low_item = table.item(row, 1)
            high_item = table.item(row, 2)

            if name_item and low_item and high_item:
                name = name_item.text().strip()
                try:
                    low_freq = float(low_item.text().strip())
                    high_freq = float(high_item.text().strip())
                    if name and low_freq < high_freq:
                        bands[name] = [low_freq, high_freq]
                except ValueError:
                    pass

        # Always use target_bands (plural) to preserve band names
        if bands:
            return {"target_bands": bands}
        return {"target_bands": {"alpha": [8.0, 12.0]}}

    def get_config_data(self) -> dict[str, Any]:
        """Extract configuration data from widgets"""
        bands_config = self.get_bands_from_table(self.widgets["bands_table"])

        # Build feature_config
        feature_config = {
            **bands_config,
            "use_relative_power": self.widgets["power_type_selector"].current == "relative",
            "use_cluster_mode": self.widgets["cluster_mode_selector"].current == "clustered",
        }

        return {
            "window_length_sec": float(self.widgets["window_length_edit"].text() or 1.0),
            "step_size_ms": int(self.widgets["window_step_edit"].text() or 250),
            "feature_config": feature_config,
        }

    def _update_widgets_from_config(self, data: dict[str, Any]):
        """Update widgets from configuration data"""
        # Window parameters now come from instance_config directly
        if "window_length_sec" in data and "window_length_edit" in self.widgets:
            self.widgets["window_length_edit"].setText(str(data["window_length_sec"]))
        if "step_size_ms" in data and "window_step_edit" in self.widgets:
            self.widgets["window_step_edit"].setText(str(data["step_size_ms"]))

        # Update power type selector from feature_config
        if "feature_config" in data and "power_type_selector" in self.widgets:
            use_relative = data["feature_config"].get("use_relative_power", True)
            self.widgets["power_type_selector"].set_current(
                "relative" if use_relative else "absolute"
            )

        # Update cluster mode selector from feature_config
        if "feature_config" in data and "cluster_mode_selector" in self.widgets:
            use_cluster = data["feature_config"].get("use_cluster_mode", False)
            self.widgets["cluster_mode_selector"].set_current(
                "clustered" if use_cluster else "individual"
            )

    def _get_time_window_duration(self) -> float:
        """Get time window duration for neurofeedback mode."""
        config = self.get_config_from_manager()
        return config.get("window_length_sec", 1.0)

    def _get_default_source(self) -> str:
        """Neurofeedback doesn't use decoders."""
        return DecoderSource.NONE
