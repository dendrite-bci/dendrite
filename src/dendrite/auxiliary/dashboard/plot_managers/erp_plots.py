#!/usr/bin/env python
"""
ERP Plot Manager

Handles initialization and updating of Event-Related Potential plots for synchronous modes.
Includes channel selection controls and averaging logic.
"""

import colorsys
import logging
import time
from collections import deque
from functools import partial
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.dashboard.backend.signal_quality import ChannelQualityResult, QualityLevel
from dendrite.auxiliary.dashboard.dialogs import ChannelSelectionDialog
from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    ERP_EVENT_COLORS,
    DotSample,
    style_legend,
    style_plot_axes,
)
from dendrite.auxiliary.dashboard.widgets.components import (
    DASHBOARD_CHECKBOX_STYLE,
    DASHBOARD_CONTROL_HEIGHT,
)
from dendrite.gui.styles.design_tokens import BORDER, TEXT_LABEL
from dendrite.gui.styles.widget_styles import WidgetStyles

# ERP configuration
ERP_HISTORY_LENGTH = 5
ERP_UPDATE_THROTTLE_S = 0.1  # Throttle ERP plot updates to 10Hz


class ERPPlotManager:
    """Manages ERP plots with channel selection controls"""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget, data_manager, sample_rate: int = 500):
        self.plot_widget = plot_widget
        self.data_manager = data_manager
        self.sample_rate = sample_rate

        # ERP data storage per mode
        self.erp_data: dict[str, dict[str, Any]] = {}

        # Plot state tracking
        self.plots: dict[str, Any] = {}
        self.selected_channels: set[int] = set(range(4))  # Default first 4 channels

        # Preprocessing state (baseline correction only)
        self.baseline_enabled = True
        self.exclude_bad: bool = True
        self._bad_channels: set[int] = set()

        # UI components
        self.selectors: dict[str, Any] = {}

        # Performance tracking
        self.erp_plot_states: dict[str, Any] = {}
        self.erp_last_update_times: dict[str, float] = {}

        # Base colors for different event types
        self.base_event_colors = ERP_EVENT_COLORS

    def create_controls_widget(self, mode_name: str) -> QtWidgets.QWidget:
        """Create single-row inline ERP controls - status bar style."""
        widget = QtWidgets.QWidget()
        widget.setFixedHeight(28)

        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        # Channel section: button opens channel selection dialog
        n_channels = len(self._get_eeg_labels())
        channel_btn = QtWidgets.QPushButton(
            f"Channels: {len(self.selected_channels)} of {n_channels}"
        )
        channel_btn.setStyleSheet(WidgetStyles.button(size="small", transparent=True, blend=True))
        channel_btn.setFixedHeight(DASHBOARD_CONTROL_HEIGHT)
        channel_btn.clicked.connect(partial(self._open_channel_dialog, mode_name))
        layout.addWidget(channel_btn)

        # Baseline checkbox (subtracts whole-epoch mean)
        baseline_check = QtWidgets.QCheckBox("Baseline")
        baseline_check.setChecked(self.baseline_enabled)
        baseline_check.setStyleSheet(DASHBOARD_CHECKBOX_STYLE)
        baseline_check.stateChanged.connect(partial(self._update_baseline_config, mode_name))
        layout.addWidget(baseline_check)

        # Exclude bad channels checkbox
        exclude_bad_check = QtWidgets.QCheckBox("Exclude bad")
        exclude_bad_check.setChecked(self.exclude_bad)
        exclude_bad_check.setStyleSheet(DASHBOARD_CHECKBOX_STYLE)
        exclude_bad_check.stateChanged.connect(partial(self._update_exclude_bad, mode_name))
        layout.addWidget(exclude_bad_check)

        layout.addStretch()

        self.selectors[mode_name] = {
            "channel_button": channel_btn,
            "baseline_check": baseline_check,
            "exclude_bad_check": exclude_bad_check,
        }

        return widget

    def _orient_eeg_data(self, eeg_data: np.ndarray) -> tuple:
        """
        Orient EEG data to (channels, samples) format.

        Determines the correct orientation based on expected channel count from
        data_manager if available, otherwise assumes smaller dimension is channels.

        Returns:
            Tuple of (oriented_data, num_channels, channel_indices)
        """
        expected_channels = None
        if self.data_manager.initialized:
            expected_channels = len(self.data_manager.eeg_channel_labels)

        # Check if data matches expected channel count in either dimension
        if expected_channels is not None:
            if eeg_data.shape[0] == expected_channels:
                return eeg_data, expected_channels, list(range(expected_channels))
            elif eeg_data.shape[1] == expected_channels:
                return eeg_data.T, expected_channels, list(range(expected_channels))

        # Fallback: assume smaller dimension is channels
        if eeg_data.shape[0] < eeg_data.shape[1]:
            num_channels = eeg_data.shape[0]
            return eeg_data, num_channels, list(range(num_channels))
        else:
            data_transposed = eeg_data.T
            num_channels = data_transposed.shape[0]
            return data_transposed, num_channels, list(range(num_channels))

    def handle_erp_data(self, mode_name: str, item: dict[str, Any]):
        """Handle incoming ERP data."""
        data = item.get("data", {})
        event_type = data.get("event_type", "UnknownEvent")
        eeg_data_raw = data.get("eeg_data")
        start_offset_ms = data.get("start_offset_ms", 0.0)
        payload_sample_rate = data.get("sample_rate", self.sample_rate)

        if eeg_data_raw is None:
            logging.warning(f"Received ERP payload for '{mode_name}' without 'eeg_data'.")
            return

        try:
            eeg_data = np.asarray(eeg_data_raw, dtype=float)
            if eeg_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {eeg_data.shape}")

            # Orient data to (channels, samples) format
            eeg_data, erp_channels, erp_channel_indices = self._orient_eeg_data(eeg_data)

            if mode_name not in self.erp_data:
                self.erp_data[mode_name] = {}
            if event_type not in self.erp_data[mode_name]:
                self.erp_data[mode_name][event_type] = {
                    "sum": np.zeros_like(eeg_data),
                    "count": 0,
                    "samples": eeg_data.shape[1],
                    "history": deque(maxlen=ERP_HISTORY_LENGTH),
                    "erp_channels": erp_channels,
                    "erp_channel_indices": erp_channel_indices,
                    "erp_channel_names": [],
                    "start_offset_ms": start_offset_ms,
                    "payload_sample_rate": payload_sample_rate,
                }

            # Update timing info (may change if mode config changes)
            self.erp_data[mode_name][event_type]["start_offset_ms"] = start_offset_ms
            self.erp_data[mode_name][event_type]["payload_sample_rate"] = payload_sample_rate

            erp_rec = self.erp_data[mode_name][event_type]

            # Validate consistency
            if eeg_data.shape[1] != erp_rec["samples"]:
                logging.warning(f"ERP sample count mismatch for {mode_name}/{event_type}")
                return
            if eeg_data.shape[0] != erp_rec["erp_channels"]:
                logging.warning(f"ERP channel count mismatch for {mode_name}/{event_type}")
                return

            # Update ERP data
            erp_rec["sum"] += eeg_data
            erp_rec["count"] += 1
            erp_rec["history"].append(eeg_data)

        except Exception as e:
            logging.error(
                f"Error processing ERP data for {mode_name}/{event_type}: {e}", exc_info=True
            )

    def update_plots(self, mode_name: str):
        """Update ERP plots for a mode."""
        if mode_name not in self.erp_data or not self.selected_channels:
            return

        # Throttle updates
        current_time = time.time()
        last_update_time = self.erp_last_update_times.get(mode_name, 0)
        if current_time - last_update_time < ERP_UPDATE_THROTTLE_S:
            return

        self.erp_last_update_times[mode_name] = current_time

        event_types = sorted(list(self.erp_data[mode_name].keys()))
        active_event_types = [et for et in event_types if self.erp_data[mode_name][et]["count"] > 0]

        # Structure change detection - only rebuild when plot structure truly changes
        # (new event types or channels), not just when data arrives after tab switch
        current_state = {
            "active_event_types": active_event_types,
            "selected_channels": tuple(sorted(self.selected_channels)),
            "exclude_bad": self.exclude_bad,
            "bad_channels": frozenset(self._bad_channels),
        }

        last_state = self.erp_plot_states.get(mode_name, {})
        needs_structural_rebuild = set(current_state["active_event_types"]) != set(
            last_state.get("active_event_types", [])
        )
        needs_curve_refresh = (
            current_state["selected_channels"] != last_state.get("selected_channels", ())
            or current_state["exclude_bad"] != last_state.get("exclude_bad", True)
            or current_state["bad_channels"] != last_state.get("bad_channels", frozenset())
        )

        # Structural rebuild: event types changed — clear entire widget and recreate plots
        if needs_structural_rebuild and mode_name in self.plots:
            self.plot_widget.clear()
            self.plots[mode_name] = {}
        elif mode_name not in self.plots:
            self.plots[mode_name] = {}

        # Always update state tracking
        self.erp_plot_states[mode_name] = current_state
        structure_changed = needs_structural_rebuild or mode_name not in self.plots

        # Process all event types
        for plot_row_idx, event_type in enumerate(active_event_types):
            erp_rec = self.erp_data[mode_name][event_type]
            event_type_idx = event_types.index(event_type)

            # Calculate cumulative average
            avg_erp = erp_rec["sum"] / erp_rec["count"]
            current_trial_count = erp_rec["count"]

            num_samples = erp_rec["samples"]
            start_offset_ms = erp_rec.get("start_offset_ms", 0.0)
            payload_sample_rate = erp_rec.get("payload_sample_rate", self.sample_rate)

            # Apply preprocessing (baseline correction)
            avg_erp = self._apply_preprocessing(avg_erp)

            # Time axis: account for start offset (pre-stimulus baseline) and use payload sample rate
            start_time_s = start_offset_ms / 1000.0
            end_time_s = start_time_s + (num_samples - 1) / payload_sample_rate
            erp_time_axis = np.linspace(start_time_s, end_time_s, num_samples)

            # Debug: verify data and time axis shapes match
            if avg_erp.shape[1] != len(erp_time_axis):
                logging.error(
                    f"ERP shape mismatch: data has {avg_erp.shape[1]} samples, "
                    f"time axis has {len(erp_time_axis)} points"
                )

            # Create plot if structure changed
            if structure_changed:
                plot_item = self.plot_widget.addPlot(row=plot_row_idx, col=0)
                legend = plot_item.addLegend(offset=(10, 10), sampleType=DotSample)
                style_legend(legend)
                plot_item.setLabel("left", "Amplitude", color=TEXT_LABEL)
                plot_item.setLabel("bottom", "Time from event (s)", color=TEXT_LABEL)
                style_plot_axes(plot_item)
                plot_item.addItem(
                    pg.InfiniteLine(
                        pos=0,
                        angle=90,
                        pen=pg.mkPen(BORDER, width=1, style=QtCore.Qt.PenStyle.DashLine),
                    )
                )
                plot_item.addItem(
                    pg.InfiniteLine(
                        pos=0,
                        angle=0,
                        pen=pg.mkPen(BORDER, width=1, style=QtCore.Qt.PenStyle.DashLine),
                    )
                )

                self.plots[mode_name][event_type] = {"plot": plot_item, "curves": {}}

            plot_config = self.plots[mode_name][event_type]
            plot_item = plot_config["plot"]

            new_title = f"{event_type} (n={current_trial_count})"
            if (
                structure_changed
                or not hasattr(plot_item, "_cached_title")
                or plot_item._cached_title != new_title
            ):
                plot_item.setTitle(new_title, color=TEXT_LABEL, size="9pt")
                plot_item._cached_title = new_title

            # Get channels, excluding bad if enabled
            base_color = self.base_event_colors[event_type_idx % len(self.base_event_colors)]
            selected_channels_list = sorted(list(self.selected_channels))
            if self.exclude_bad and self._bad_channels:
                selected_channels_list = [
                    ch for ch in selected_channels_list if ch not in self._bad_channels
                ]
            if not selected_channels_list:
                continue
            channel_colors = self._generate_color_variations(
                base_color, len(selected_channels_list)
            )

            # Curve refresh: remove stale curves and rebuild legend without clearing the plot
            if needs_curve_refresh and not structure_changed:
                expected_keys = {
                    f"ch_{ch_idx}" for ch_idx in selected_channels_list if ch_idx < avg_erp.shape[0]
                }
                stale_keys = [k for k in plot_config["curves"] if k not in expected_keys]
                for key in stale_keys:
                    plot_item.removeItem(plot_config["curves"].pop(key))
                # Rebuild legend for current curves
                legend = plot_item.legend
                if legend:
                    legend.clear()

            for color_idx, ch_idx in enumerate(selected_channels_list):
                if ch_idx < avg_erp.shape[0]:
                    curve_key = f"ch_{ch_idx}"
                    channel_name = f"Ch{ch_idx + 1}"

                    if self.data_manager.initialized and ch_idx < len(
                        self.data_manager.eeg_channel_labels
                    ):
                        channel_name = self.data_manager.eeg_channel_labels[ch_idx]

                    color = channel_colors[color_idx % len(channel_colors)]

                    if structure_changed or curve_key not in plot_config["curves"]:
                        curve = plot_item.plot(pen=pg.mkPen(color, width=1.5), name=channel_name)
                        plot_config["curves"][curve_key] = curve
                    else:
                        curve = plot_config["curves"][curve_key]
                        # Re-add to legend after refresh cleared it
                        if needs_curve_refresh and not structure_changed:
                            legend = plot_item.legend
                            if legend:
                                legend.addItem(curve, channel_name)

                    curve.setData(erp_time_axis, avg_erp[ch_idx, :])

    def _generate_color_variations(self, base_color: str, num_channels: int) -> list[str]:
        """Generate color variations for channels."""
        base_color = base_color.lstrip("#")
        r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (0, 2, 4))
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        variations = []
        for i in range(num_channels):
            # Single channel uses base color (ratio=0.5 -> offset=0)
            ratio = i / (num_channels - 1) if num_channels > 1 else 0.5
            new_s = max(0.3, min(1.0, s + (ratio - 0.5) * 0.3))
            new_v = max(0.4, min(1.0, v + (ratio - 0.5) * 0.2))
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            hex_color = f"#{int(new_r * 255):02x}{int(new_g * 255):02x}{int(new_b * 255):02x}"
            variations.append(hex_color)

        return variations

    def _get_eeg_labels(self) -> list[str]:
        """Get EEG channel labels dynamically from data_manager."""
        if self.data_manager.initialized:
            return self.data_manager.eeg_channel_labels
        return [f"Ch{i + 1}" for i in range(64)]

    def _open_channel_dialog(self, mode_name: str):
        """Open channel selection dialog."""
        labels = self._get_eeg_labels()

        # Find the parent widget for the dialog
        parent = None
        if mode_name in self.selectors:
            btn = self.selectors[mode_name].get("channel_button")
            if btn:
                parent = btn.window()

        accepted, new_selection = ChannelSelectionDialog.get_selection(
            labels, self.selected_channels, parent, self._bad_channels
        )
        if accepted:
            self.selected_channels = new_selection
            self._update_channel_button(mode_name)
            self.update_plots(mode_name)

    def _update_channel_button(self, mode_name: str):
        """Update channel button text with current selection count."""
        if mode_name not in self.selectors:
            return

        selectors = self.selectors[mode_name]
        all_labels = self._get_eeg_labels()
        n_channels = len(all_labels)

        # Clamp selected channels to valid range
        self.selected_channels = {ch for ch in self.selected_channels if ch < n_channels}

        # Update button text
        channel_btn = selectors.get("channel_button")
        if channel_btn:
            channel_btn.setText(f"Channels: {len(self.selected_channels)} of {n_channels}")

    def _update_baseline_config(self, mode_name: str, value):
        """Update baseline enabled state and replot."""
        self.baseline_enabled = value == QtCore.Qt.CheckState.Checked.value
        self._force_replot(mode_name)

    def update_quality(self, results: list[ChannelQualityResult]) -> None:
        """Update bad channel set from quality results and replot if needed."""
        bad = {r.channel_index for r in results if r.level == QualityLevel.BAD}
        if bad != self._bad_channels:
            self._bad_channels = bad
            if self.exclude_bad:
                for mode_name in list(self.erp_data.keys()):
                    self._force_replot(mode_name)

    def _update_exclude_bad(self, mode_name: str, value):
        """Update exclude-bad state and replot."""
        self.exclude_bad = value == QtCore.Qt.CheckState.Checked.value
        self._force_replot(mode_name)

    def _force_replot(self, mode_name: str):
        """Force replot by resetting the throttle timer."""
        self.erp_last_update_times[mode_name] = 0
        self.update_plots(mode_name)

    def _apply_preprocessing(self, erp_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to averaged ERP data (baseline correction only)."""
        data = erp_data.copy()

        if self.baseline_enabled:
            data = self._apply_baseline_correction(data)

        return data

    def _apply_baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """Subtract whole-epoch mean from each channel."""
        baseline = np.mean(data, axis=1, keepdims=True)
        return data - baseline

    def clear_plots(self, mode_name: str | None = None):
        """Clear ERP plots."""
        if mode_name:
            if mode_name in self.plots:
                del self.plots[mode_name]
            if mode_name in self.erp_data:
                del self.erp_data[mode_name]
        else:
            if self.plot_widget:
                self.plot_widget.clear()
            self.plots.clear()
            self.erp_data.clear()

        if mode_name:
            logging.info(f"ERP plots cleared for mode: {mode_name}")
        else:
            logging.info("ERP plots cleared")
