#!/usr/bin/env python
"""
Neurofeedback Plot Manager

Handles visualization of real-time neurofeedback features including band power
visualizations across multiple channels and frequency bands.
"""

import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    NEUROFEEDBACK_SIGNAL,
    style_plot_axes,
)
from dendrite.gui.styles.design_tokens import TEXT_LABEL


class NeurofeedbackPlotManager:
    """
    Manages neurofeedback feature plots for real-time band power visualization.

    Handles:
    - Multi-channel band power plots
    - Real-time line plots with history
    - Automatic layout and scaling
    - Multiple frequency bands per channel
    """

    def __init__(self, parent_widget: QtWidgets.QWidget):
        self.parent_widget = parent_widget
        self.logger = logging.getLogger(__name__)

        # Data storage for each mode
        self.neurofeedback_data: dict[str, dict[str, Any]] = {}

        # UI organization - mode_manager provides stack and navigation
        self.mode_manager = None

        self.max_history_points = 100
        self.max_features_per_group = 8
        self.max_cols = 1  # Single column vertical layout

        # Track modes that have already shown channel limit warning
        self._channel_limit_logged: set = set()

    def initialize_for_mode(self, mode_name: str, mode_manager) -> bool:
        """Initialize neurofeedback UI for a specific mode."""
        if mode_name in self.neurofeedback_data:
            return False  # Already initialized

        self.mode_manager = mode_manager

        # Initialize data structure
        self.neurofeedback_data[mode_name] = {
            "ui": {"widget": None, "layout": None, "plots": {}, "curves": {}},
            "latest_features": {},
            "feature_history": defaultdict(lambda: deque(maxlen=self.max_history_points)),
            "last_plotted_data": {},
            "last_yrange": {},
            "target_bands": {},
        }

        # Create content widget
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)

        title_label = QtWidgets.QLabel(f"{mode_name} - Neurofeedback Features")
        title_label.setStyleSheet(f"font-weight: 500; color: {TEXT_LABEL}; font-size: 11px;")
        tab_layout.addWidget(title_label)

        # Create graphics widget
        graphics_widget = pg.GraphicsLayoutWidget()
        graphics_widget.setBackground(None)
        graphics_widget.setMinimumHeight(100)
        graphics_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        tab_layout.addWidget(graphics_widget)

        # Store UI references
        self.neurofeedback_data[mode_name]["ui"]["widget"] = graphics_widget
        self.neurofeedback_data[mode_name]["ui"]["layout"] = tab_layout

        # Add to stack and update navigation via mode_manager
        mode_manager.add_external_tab(mode_name, tab)

        self.logger.info(f"Initialized neurofeedback UI for mode: {mode_name}")
        return True

    def update_features(self, mode_name: str, payload: Any) -> None:
        """Update neurofeedback features for a mode."""
        if mode_name not in self.neurofeedback_data:
            self.logger.warning(f"Mode {mode_name} not initialized in neurofeedback_data")
            return

        # Extract data from payload (dict from mode_manager routing)
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

        # Store latest data
        mode_data = self.neurofeedback_data[mode_name]
        mode_data["target_bands"] = target_bands

        # Flatten channel_powers for easier plotting
        latest_features = {}
        for channel, bands in channel_powers.items():
            for band_name, power_value in bands.items():
                feature_key = f"{channel}_{band_name}"
                latest_features[feature_key] = float(power_value)

                # Add to history
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

        # Organize features by channel and band
        organized_features = self._organize_features_by_channel(latest_features)

        # Update or create plots
        self._update_organized_plots(
            mode_name, organized_features, graphics_widget, feature_history
        )

    def _organize_features_by_channel(
        self, features: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Organize features by channel for better layout."""
        organized = defaultdict(dict)

        for feature_key, value in features.items():
            # Split feature_key like "ch0_alpha" -> channel="ch0", band="alpha"
            if "_" in feature_key:
                parts = feature_key.rsplit(
                    "_", 1
                )  # Split from right to handle multi-underscore names
                if len(parts) == 2:
                    channel, band = parts
                    organized[channel][band] = value
                else:
                    # Fallback for unexpected format
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
        """Update plots with organized feature data."""
        mode_data = self.neurofeedback_data[mode_name]
        plots = mode_data["ui"]["plots"]
        curves = mode_data["ui"]["curves"]

        # Calculate layout
        channels = sorted(organized_features.keys())

        # Limit number of channels if too many
        if len(channels) > 16:
            channels = channels[:16]
            if mode_name not in self._channel_limit_logged:
                self.logger.info(
                    f"Limiting neurofeedback display to first 16 channels for mode {mode_name}"
                )
                self._channel_limit_logged.add(mode_name)

        current_row = 0

        for i, channel in enumerate(channels):
            bands = organized_features[channel]

            if len(bands) == 1:
                # Single band per channel - one plot per channel
                band_name = list(bands.keys())[0]
                feature_key = f"{channel}_{band_name}"

                row = current_row + (i // self.max_cols)
                col = i % self.max_cols

                self._create_or_update_plot(
                    mode_name,
                    feature_key,
                    f"{channel}\n{band_name}",
                    graphics_widget,
                    plots,
                    curves,
                    feature_history,
                    row,
                    col,
                )
            else:
                # Multiple bands per channel - create subplots
                band_names = sorted(bands.keys())
                for j, band_name in enumerate(band_names):
                    feature_key = f"{channel}_{band_name}"
                    plot_title = f"{channel}\n{band_name}"

                    # Calculate position for multi-band layout
                    plot_index = i * len(band_names) + j
                    row = current_row + (plot_index // self.max_cols)
                    col = plot_index % self.max_cols

                    self._create_or_update_plot(
                        mode_name,
                        feature_key,
                        plot_title,
                        graphics_widget,
                        plots,
                        curves,
                        feature_history,
                        row,
                        col,
                    )

    def _create_or_update_plot(
        self,
        mode_name: str,
        feature_key: str,
        title: str,
        graphics_widget: pg.GraphicsLayoutWidget,
        plots: dict,
        curves: dict,
        feature_history: dict,
        row: int,
        col: int,
    ) -> None:
        """Create or update a single feature plot."""
        mode_data = self.neurofeedback_data[mode_name]

        # Create plot if it doesn't exist
        if feature_key not in plots:
            plot_item = graphics_widget.addPlot(row=row, col=col)
            style_plot_axes(plot_item)
            plot_item.setTitle(title, color=TEXT_LABEL, size="9pt")
            plot_item.setLabel("left", "Power", color=TEXT_LABEL, size="8pt")
            plot_item.setLabel("bottom", "Time", color=TEXT_LABEL, size="8pt")
            plot_item.showGrid(x=True, y=True, alpha=0.3)
            plot_item.setMouseEnabled(x=False, y=True)

            curve = plot_item.plot(pen=pg.mkPen(NEUROFEEDBACK_SIGNAL, width=2))

            plots[feature_key] = plot_item
            curves[feature_key] = curve

            # Initialize tracking data
            mode_data["last_plotted_data"][feature_key] = None
            mode_data["last_yrange"][feature_key] = None

            self.logger.debug(f"Created plot for {feature_key} at ({row}, {col})")

        # Update plot data
        plot_item = plots[feature_key]
        curve = curves[feature_key]
        history = list(feature_history.get(feature_key, []))

        if history:
            x_data = np.arange(len(history))
            y_data = np.array(history)

            # Only update if data changed
            last_data = mode_data["last_plotted_data"].get(feature_key)
            if last_data is None or not np.array_equal(last_data, y_data):
                curve.setData(x_data, y_data)
                mode_data["last_plotted_data"][feature_key] = y_data  # No need to copy

                # Auto-range y-axis
                self._auto_range_plot(plot_item, y_data, feature_key, mode_data)

    def _auto_range_plot(
        self, plot_item: pg.PlotItem, data: np.ndarray, feature_key: str, mode_data: dict
    ) -> None:
        """Automatically range the plot based on data."""
        if len(data) == 0:
            return

        min_val, max_val = np.min(data), np.max(data)
        data_range = max_val - min_val

        if data_range < 1e-9:  # Flat data
            padding = 0.5
            y_min, y_max = min_val - padding, max_val + padding
        else:
            padding = data_range * 0.1
            y_min, y_max = min_val - padding, max_val + padding

        # Only update y-range if it changed significantly
        last_range = mode_data["last_yrange"].get(feature_key)
        if (
            last_range is None
            or abs(last_range[0] - y_min) > data_range * 0.05
            or abs(last_range[1] - y_max) > data_range * 0.05
        ):
            plot_item.setYRange(y_min, y_max, padding=0)
            mode_data["last_yrange"][feature_key] = (y_min, y_max)

    def clear_all_data(self) -> None:
        """Clear all neurofeedback data."""
        self.neurofeedback_data.clear()
        self.logger.info("Cleared all neurofeedback data")
