#!/usr/bin/env python
"""
Modality Plot Manager

Handles initialization and updating of physiological signal plots (EMG, EOG, etc.).
"""

import numpy as np
import pyqtgraph as pg

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    MODALITY_COLORS,
    setup_signal_plot,
    update_plot_y_range_if_needed,
)


class ModalityPlotManager:
    """Manages physiological signal plots"""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget, data_manager):
        self.plot_widget = plot_widget
        self.data_manager = data_manager
        self.mod_curves: list[pg.PlotDataItem] = []
        self.range_padding = 0.1

        # Colors for different modality types
        self.colors = MODALITY_COLORS

        # Performance optimizations
        self.plot_items = []  # Cache plot item references
        self.last_y_ranges = []  # Track last Y ranges to avoid unnecessary updates
        self.range_change_threshold = 0.1  # Only update if range changes >10%

    def initialize_plots(self):
        """
        Initialize modality plots with multi-column layout.
        Each modality type gets its own column with channels stacked vertically.
        """
        if not self.data_manager.initialized:
            return

        self.plot_widget.clear()
        self.mod_curves = []
        self.plot_items = []
        self.last_y_ranges = []

        num_modalities = len(self.data_manager.modality_channel_labels)
        if num_modalities == 0:
            return

        # Each modality type gets its own column
        for col_idx, modality_type in enumerate(self.data_manager.modality_channel_labels):
            channel_labels = self.data_manager.modality_channel_names.get(modality_type, [])
            if not channel_labels:
                continue

            color = self.colors[col_idx % len(self.colors)]

            for row_idx, channel_label in enumerate(channel_labels):
                p, curve = setup_signal_plot(
                    self.plot_widget,
                    row_idx,
                    col_idx,
                    title=channel_label,
                    x_max=self.data_manager.time_axis[-1],
                    y_range=(-100, 100),
                    color=color,
                    max_height=150,
                )
                self.mod_curves.append(curve)
                self.plot_items.append(p)
                self.last_y_ranges.append((-100, 100))

    def update_plots(self):
        """Update modality plot data and auto-range axes with optimizations."""
        if not self.mod_curves or not self.data_manager.initialized:
            return

        for i, curve in enumerate(self.mod_curves):
            if i < len(self.data_manager.mod_buffers) and len(self.data_manager.mod_buffers[i]) > 1:
                buffer = self.data_manager.mod_buffers[i]
                data = np.array(buffer)

                # Update curve data
                curve.setData(self.data_manager.time_axis[-len(data) :], data)

                # Update Y range if it changed significantly
                update_plot_y_range_if_needed(
                    data,
                    i,
                    self.last_y_ranges,
                    self.plot_items,
                    self.range_padding,
                    self.range_change_threshold,
                )

    def clear_plots(self):
        """Clear all modality plots."""
        if self.plot_widget:
            self.plot_widget.clear()
        self.mod_curves = []
