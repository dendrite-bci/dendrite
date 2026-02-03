#!/usr/bin/env python
"""
EEG Plot Manager

Handles initialization and updating of EEG channel plots.
"""

import logging

import numpy as np
import pyqtgraph as pg

from dendrite.auxiliary.dashboard.backend.signal_quality import (
    ChannelQualityResult,
    QualityLevel,
)
from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    EEG_SIGNAL,
    setup_signal_plot,
    update_plot_y_range_if_needed,
)
from dendrite.gui.styles.design_tokens import STATUS_ERROR, STATUS_OK, STATUS_WARN, TEXT_LABEL

_QUALITY_TITLE_COLORS: dict[QualityLevel, str] = {
    QualityLevel.GOOD: STATUS_OK,
    QualityLevel.WARNING: STATUS_WARN,
    QualityLevel.BAD: STATUS_ERROR,
    QualityLevel.UNKNOWN: TEXT_LABEL,
}


class EEGPlotManager:
    """Manages EEG channel plots"""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget, data_manager):
        self.plot_widget = plot_widget
        self.data_manager = data_manager
        self.eeg_curves = []
        self.selected_channels = []
        self.max_displayed_channels = 32
        self.range_padding = 0.1

        # Performance optimizations
        self.plot_items = []  # Cache plot item references
        self.last_y_ranges = []  # Track last Y ranges to avoid unnecessary updates
        self.range_change_threshold = 0.1  # Only update if range changes >10%

    def initialize_plots(self, selected_channels: list[int] | None = None):
        """Initialize EEG plots based on selected channels."""
        if not self.data_manager.initialized:
            return

        self.plot_widget.clear()
        self.eeg_curves = []
        self.plot_items = []
        self.last_y_ranges = []

        # Use provided channels or default to first 16
        if selected_channels is not None:
            channels_to_show = selected_channels
        elif hasattr(self, "selected_channels") and self.selected_channels:
            channels_to_show = self.selected_channels
        else:
            channels_to_show = list(range(min(16, self.data_manager.num_eeg)))

        self.selected_channels = channels_to_show
        num_to_show = len(channels_to_show)

        if num_to_show == 0:
            logging.warning("No EEG channels selected for display")
            return

        cols = 4

        for plot_idx, channel_idx in enumerate(channels_to_show):
            row, col = divmod(plot_idx, cols)
            label = (
                self.data_manager.eeg_channel_labels[channel_idx]
                if channel_idx < len(self.data_manager.eeg_channel_labels)
                else f"EEG_{channel_idx}"
            )

            p, curve = setup_signal_plot(
                self.plot_widget,
                row,
                col,
                title=f"{channel_idx}: {label}",
                x_max=self.data_manager.time_axis[-1],
                y_range=(-100, 100),
                color=EEG_SIGNAL,
            )
            self.eeg_curves.append((curve, channel_idx))
            self.plot_items.append(p)
            self.last_y_ranges.append((-100, 100))

    def update_plots(self):
        """Update EEG plot data and auto-range axes with optimizations."""
        if not self.data_manager.initialized:
            return

        if not self.eeg_curves:
            return

        for plot_idx, (curve, channel_idx) in enumerate(self.eeg_curves):
            if (
                channel_idx < len(self.data_manager.eeg_buffers)
                and len(self.data_manager.eeg_buffers[channel_idx]) > 1
            ):
                buffer = self.data_manager.eeg_buffers[channel_idx]
                data = np.array(buffer)

                # Update curve data
                curve.setData(self.data_manager.time_axis[-len(data) :], data)

                # Update Y range if it changed significantly
                update_plot_y_range_if_needed(
                    data,
                    plot_idx,
                    self.last_y_ranges,
                    self.plot_items,
                    self.range_padding,
                    self.range_change_threshold,
                )

    def update_channel_selection(self, selected_channels: list[int]):
        """Update which channels are displayed."""
        self.selected_channels = selected_channels
        self.initialize_plots(selected_channels)

    def update_quality_indicators(self, results: list[ChannelQualityResult]):
        """Color EEG plot titles by signal quality level."""
        if not self.eeg_curves or not self.plot_items:
            return

        # Build channel_index -> QualityLevel lookup
        quality_map: dict[int, QualityLevel] = {r.channel_index: r.level for r in results}

        for plot_idx, (_, channel_idx) in enumerate(self.eeg_curves):
            if plot_idx >= len(self.plot_items):
                break

            level = quality_map.get(channel_idx, QualityLevel.UNKNOWN)
            color = _QUALITY_TITLE_COLORS.get(level, TEXT_LABEL)

            label = (
                self.data_manager.eeg_channel_labels[channel_idx]
                if channel_idx < len(self.data_manager.eeg_channel_labels)
                else f"EEG_{channel_idx}"
            )
            self.plot_items[plot_idx].setTitle(f"{channel_idx}: {label}", color=color, size="8pt")

    def clear_plots(self):
        """Clear all EEG plots."""
        if self.plot_widget:
            self.plot_widget.clear()
        self.eeg_curves = []
