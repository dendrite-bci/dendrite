#!/usr/bin/env python
"""
Neurofeedback Plot Manager

Handles visualization of real-time neurofeedback features including band power
visualizations across multiple channels and frequency bands.

Bands are overlaid per channel as differently-colored lines, arranged in a
2-column grid for compact display.
"""

import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    NEUROFEEDBACK_BAND_COLORS,
    calculate_y_range,
    should_update_y_range,
    style_plot_axes,
)
from dendrite.gui.styles.design_tokens import TEXT_LABEL, TEXT_MUTED


class NeurofeedbackPlotManager:
    """
    Manages neurofeedback feature plots for real-time band power visualization.

    Each channel gets one plot with frequency bands overlaid as colored curves.
    Channels are arranged in a 2-column grid.
    """

    def __init__(self, parent_widget: QtWidgets.QWidget):
        self.parent_widget = parent_widget
        self.logger = logging.getLogger(__name__)

        self.neurofeedback_data: dict[str, dict[str, Any]] = {}
        self.mode_manager = None

        self.max_history_points = 100
        self.grid_cols = 2

        self._channel_limit_logged: set = set()

    def initialize_for_mode(self, mode_name: str, mode_manager) -> bool:
        """Initialize neurofeedback UI for a specific mode."""
        if mode_name in self.neurofeedback_data:
            return False

        self.mode_manager = mode_manager

        self.neurofeedback_data[mode_name] = {
            "ui": {
                "widget": None,
                "layout": None,
                "plots": {},
                "curves": {},
                "legend_widget": None,
            },
            "latest_features": {},
            "feature_history": defaultdict(lambda: deque(maxlen=self.max_history_points)),
            "last_plotted_data": {},
            "last_yrange": {},
            "target_bands": {},
            "known_bands": [],
            "last_channel_set": frozenset(),
        }

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)

        title_label = QtWidgets.QLabel(f"{mode_name} - Neurofeedback Features")
        title_label.setStyleSheet(f"font-weight: 500; color: {TEXT_LABEL}; font-size: 11px;")
        tab_layout.addWidget(title_label)

        graphics_widget = pg.GraphicsLayoutWidget()
        graphics_widget.setBackground(None)
        graphics_widget.setMinimumHeight(100)
        graphics_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        tab_layout.addWidget(graphics_widget)

        self.neurofeedback_data[mode_name]["ui"]["widget"] = graphics_widget
        self.neurofeedback_data[mode_name]["ui"]["layout"] = tab_layout

        mode_manager.add_external_tab(mode_name, tab)

        self.logger.info(f"Initialized neurofeedback UI for mode: {mode_name}")
        return True

    def update_features(self, mode_name: str, payload: Any) -> None:
        """Update neurofeedback features for a mode."""
        if mode_name not in self.neurofeedback_data:
            self.logger.warning(f"Mode {mode_name} not initialized in neurofeedback_data")
            return

        if isinstance(payload, dict):
            channel_powers = payload.get("channel_powers") or payload.get("data", {}).get(
                "channel_powers", {}
            )
            target_bands = payload.get("target_bands") or payload.get("data", {}).get(
                "target_bands", {}
            )
        else:
            channel_powers = getattr(payload, "channel_powers", {})
            target_bands = getattr(payload, "target_bands", {})

        if not channel_powers:
            return

        mode_data = self.neurofeedback_data[mode_name]
        mode_data["target_bands"] = target_bands

        latest_features = {}
        for channel, bands in channel_powers.items():
            for band_name, power_value in bands.items():
                feature_key = f"{channel}_{band_name}"
                latest_features[feature_key] = float(power_value)
                mode_data["feature_history"][feature_key].append(power_value)

        mode_data["latest_features"] = latest_features
        self._update_plots(mode_name)

    def _update_plots(self, mode_name: str) -> None:
        """Update the plots for a specific mode."""
        mode_data = self.neurofeedback_data[mode_name]
        ui = mode_data["ui"]
        graphics_widget = ui["widget"]
        latest_features = mode_data["latest_features"]
        feature_history = mode_data["feature_history"]

        if not graphics_widget or not latest_features:
            return

        organized_features = self._organize_features_by_channel(latest_features)

        self._update_organized_plots(
            mode_name, organized_features, graphics_widget, feature_history
        )

    def _organize_features_by_channel(
        self, features: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Organize features by channel for better layout."""
        organized = defaultdict(dict)

        for feature_key, value in features.items():
            if "_" in feature_key:
                parts = feature_key.rsplit("_", 1)
                if len(parts) == 2:
                    channel, band = parts
                    organized[channel][band] = value
                else:
                    organized[feature_key]["value"] = value
            else:
                organized[feature_key]["value"] = value

        return dict(organized)

    def _update_organized_plots(
        self,
        mode_name: str,
        organized_features: dict[str, dict[str, float]],
        graphics_widget: pg.GraphicsLayoutWidget,
        feature_history: dict,
    ) -> None:
        """Update plots: one per channel with bands overlaid, in a 2-column grid."""
        mode_data = self.neurofeedback_data[mode_name]
        plots = mode_data["ui"]["plots"]
        curves = mode_data["ui"]["curves"]

        channels = sorted(organized_features.keys())

        if len(channels) > 16:
            channels = channels[:16]
            if mode_name not in self._channel_limit_logged:
                self.logger.info(
                    f"Limiting neurofeedback display to first 16 channels for mode {mode_name}"
                )
                self._channel_limit_logged.add(mode_name)

        # Detect channel set changes -> rebuild grid
        current_channel_set = frozenset(channels)
        if current_channel_set != mode_data["last_channel_set"]:
            graphics_widget.clear()
            plots.clear()
            curves.clear()
            mode_data["last_plotted_data"].clear()
            mode_data["last_yrange"].clear()
            mode_data["last_channel_set"] = current_channel_set

        # Collect all bands across channels for consistent coloring
        all_bands: set[str] = set()
        for bands in organized_features.values():
            all_bands.update(bands.keys())
        band_list = sorted(all_bands)

        # Update legend if band set changed
        if band_list != mode_data["known_bands"]:
            mode_data["known_bands"] = band_list
            self._update_legend(mode_name, band_list)

        for i, channel in enumerate(channels):
            row = i // self.grid_cols
            col = i % self.grid_cols
            bands = organized_features[channel]

            self._create_or_update_channel_plot(
                mode_name, channel, bands, band_list, graphics_widget,
                plots, curves, feature_history, row, col,
            )

    def _create_or_update_channel_plot(
        self,
        mode_name: str,
        channel: str,
        bands: dict[str, float],
        band_list: list[str],
        graphics_widget: pg.GraphicsLayoutWidget,
        plots: dict,
        curves: dict,
        feature_history: dict,
        row: int,
        col: int,
    ) -> None:
        """Create or update a channel plot with one curve per band."""
        mode_data = self.neurofeedback_data[mode_name]

        # Create plot if it doesn't exist for this channel
        if channel not in plots:
            plot_item = graphics_widget.addPlot(row=row, col=col)
            style_plot_axes(plot_item)
            plot_item.setTitle(channel, color=TEXT_LABEL, size="9pt")
            plot_item.hideAxis("bottom")
            plot_item.setMouseEnabled(x=False, y=True)

            plots[channel] = plot_item
            curves[channel] = {}
            mode_data["last_yrange"][channel] = None

            self.logger.debug(f"Created channel plot for {channel} at ({row}, {col})")

        plot_item = plots[channel]

        # Ensure a curve exists for each band in this channel
        for band_name in bands:
            if band_name not in curves[channel]:
                band_idx = band_list.index(band_name) if band_name in band_list else 0
                color = NEUROFEEDBACK_BAND_COLORS[band_idx % len(NEUROFEEDBACK_BAND_COLORS)]
                curve = plot_item.plot(pen=pg.mkPen(color, width=1.5))
                curves[channel][band_name] = curve

        # Update each band's curve and collect all y-data for combined auto-range
        all_y_data = []
        for band_name in bands:
            feature_key = f"{channel}_{band_name}"
            curve = curves[channel][band_name]
            history = list(feature_history.get(feature_key, []))

            if not history:
                continue

            x_data = np.arange(len(history))
            y_data = np.array(history)

            last_data = mode_data["last_plotted_data"].get(feature_key)
            if last_data is None or not np.array_equal(last_data, y_data):
                curve.setData(x_data, y_data)
                mode_data["last_plotted_data"][feature_key] = y_data

            all_y_data.append(y_data)

        # Combined auto-range across all bands for this channel
        if all_y_data:
            combined = np.concatenate(all_y_data)
            self._auto_range_channel(plot_item, combined, channel, mode_data)

    def _auto_range_channel(
        self, plot_item: pg.PlotItem, data: np.ndarray, channel: str, mode_data: dict
    ) -> None:
        """Auto-range y-axis across all bands for a channel."""
        if len(data) == 0:
            return

        y_min, y_max = calculate_y_range(data)

        last_range = mode_data["last_yrange"].get(channel)
        if last_range is None or should_update_y_range(
            y_min, y_max, last_range[0], last_range[1], threshold=0.05
        ):
            plot_item.setYRange(y_min, y_max, padding=0)
            mode_data["last_yrange"][channel] = (y_min, y_max)

    def _update_legend(self, mode_name: str, band_list: list[str]) -> None:
        """Create or update a shared legend widget showing band->color mapping."""
        mode_data = self.neurofeedback_data[mode_name]
        ui = mode_data["ui"]
        tab_layout = ui["layout"]

        # Remove old legend if present
        if ui["legend_widget"] is not None:
            tab_layout.removeWidget(ui["legend_widget"])
            ui["legend_widget"].deleteLater()
            ui["legend_widget"] = None

        # No legend needed for single band
        if len(band_list) <= 1:
            return

        legend_widget = QtWidgets.QWidget()
        legend_layout = QtWidgets.QHBoxLayout(legend_widget)
        legend_layout.setContentsMargins(4, 2, 4, 2)
        legend_layout.setSpacing(12)

        for i, band_name in enumerate(band_list):
            color = NEUROFEEDBACK_BAND_COLORS[i % len(NEUROFEEDBACK_BAND_COLORS)]

            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 2px;")

            label = QtWidgets.QLabel(band_name)
            label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")

            legend_layout.addWidget(swatch)
            legend_layout.addWidget(label)

        legend_layout.addStretch()
        legend_widget.setFixedHeight(22)

        # Insert between title (index 0) and graphics widget (index 1)
        tab_layout.insertWidget(1, legend_widget, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        ui["legend_widget"] = legend_widget

    def clear_all_data(self) -> None:
        """Clear all neurofeedback data."""
        self.neurofeedback_data.clear()
        self.logger.info("Cleared all neurofeedback data")
