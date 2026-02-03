#!/usr/bin/env python
"""
PSD Plot Manager

Handles initialization and updating of real-time Power Spectral Density plots.
Uses Welch's method to compute PSD from EEG ring buffers.
"""

import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from scipy.signal import welch

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    PSD_CHANNEL_OVERLAY,
    PSD_MEAN_SIGNAL,
    style_plot_axes,
)
from dendrite.auxiliary.dashboard.widgets.components import (
    DASHBOARD_CHECKBOX_STYLE,
    LABEL_STYLES,
)
from dendrite.gui.styles.design_tokens import TEXT_LABEL, TEXT_MUTED

# Frequency band definitions: (name, low_hz, high_hz, rgba)
PSD_BANDS = [
    ("delta", 0.5, 4.0, (100, 100, 255, 25)),
    ("theta", 4.0, 8.0, (100, 200, 100, 25)),
    ("alpha", 8.0, 13.0, (200, 200, 100, 25)),
    ("beta", 13.0, 30.0, (200, 100, 100, 25)),
    ("gamma", 30.0, 50.0, (180, 100, 200, 25)),
]

# Welch parameters
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128
WELCH_NFFT = 512
FREQ_MIN = 0.5
FREQ_MAX = 50.0

# Throttle interval for PSD updates (seconds)
PSD_UPDATE_INTERVAL = 0.5


class PSDPlotManager:
    """Manages real-time PSD plots for EEG data."""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget, data_manager):
        self.plot_widget = plot_widget
        self.data_manager = data_manager

        self.plot_item: pg.PlotItem | None = None
        self.mean_curve: pg.PlotDataItem | None = None
        self.channel_curves: list[pg.PlotDataItem] = []
        self.band_regions: list[pg.LinearRegionItem] = []
        self.band_labels: list[pg.TextItem] = []
        self.selected_channels: list[int] = []

        self._show_channels = False
        self._show_bands = True
        self._last_update_time = 0.0

    def initialize_plots(self, selected_channels: list[int] | None = None):
        """Create the PSD plot with mean curve and band regions."""
        self.plot_widget.clear()
        self.channel_curves = []
        self.band_regions = []
        self.band_labels = []

        if selected_channels is not None:
            self.selected_channels = selected_channels

        self.plot_item = self.plot_widget.addPlot(row=0, col=0)
        self.plot_item.setMouseEnabled(x=False, y=True)
        self.plot_item.setTitle("Power Spectral Density", color=TEXT_LABEL, size="8pt")
        self.plot_item.setLabel("bottom", "Frequency (Hz)")
        self.plot_item.setLabel("left", "Power (dB)")
        self.plot_item.setXRange(FREQ_MIN, FREQ_MAX, padding=0)
        self.plot_item.setYRange(-20, 40, padding=0)
        self.plot_item.showGrid(x=True, y=True, alpha=0.15)
        self.plot_item.setMenuEnabled(False)

        style_plot_axes(self.plot_item)

        # Band shading regions
        self._create_band_regions()

        # Mean PSD curve (thick, on top)
        self.mean_curve = self.plot_item.plot(pen=pg.mkPen(PSD_MEAN_SIGNAL, width=2))

        self._last_update_time = 0.0

    def _create_band_regions(self):
        """Add shaded frequency band regions with labels."""
        for name, low, high, rgba in PSD_BANDS:
            region = pg.LinearRegionItem(
                values=(low, high),
                orientation="vertical",
                brush=pg.mkBrush(*rgba),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-10)
            region.setVisible(self._show_bands)
            self.plot_item.addItem(region)
            self.band_regions.append(region)

            label = pg.TextItem(name, color=TEXT_MUTED, anchor=(0.5, 1.0))
            label.setPos((low + high) / 2, -18)
            label.setVisible(self._show_bands)
            self.plot_item.addItem(label)
            self.band_labels.append(label)

    def update_plots(self):
        """Recompute PSD if enough time has passed since last update."""
        if self.plot_item is None or not self.data_manager.initialized:
            return

        now = time.monotonic()
        if now - self._last_update_time < PSD_UPDATE_INTERVAL:
            return
        self._last_update_time = now

        fs = self.data_manager.sample_rate
        if fs <= 0:
            return

        channels = self.selected_channels
        if not channels:
            channels = list(range(min(16, self.data_manager.num_eeg)))

        # Collect valid channel data
        channel_psds = []
        freqs = None

        for ch_idx in channels:
            if ch_idx >= len(self.data_manager.eeg_buffers):
                continue
            buf = self.data_manager.eeg_buffers[ch_idx]
            if len(buf) < WELCH_NPERSEG:
                continue

            data = np.array(buf)
            nperseg = min(WELCH_NPERSEG, len(data))
            noverlap = min(WELCH_NOVERLAP, nperseg - 1)

            f, pxx = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT)

            # Restrict to frequency range of interest
            mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
            f = f[mask]
            pxx = pxx[mask]

            # Convert to dB, clamp floor to avoid log(0)
            pxx_db = 10.0 * np.log10(np.maximum(pxx, 1e-20))

            if freqs is None:
                freqs = f
            channel_psds.append(pxx_db)

        if not channel_psds or freqs is None:
            return

        psd_array = np.array(channel_psds)
        mean_psd = np.mean(psd_array, axis=0)

        # Update mean curve
        self.mean_curve.setData(freqs, mean_psd)

        # Update per-channel overlay curves
        self._update_channel_overlays(freqs, psd_array)

        # Auto-range Y axis based on mean PSD
        y_min = float(np.min(mean_psd)) - 5
        y_max = float(np.max(mean_psd)) + 5
        self.plot_item.setYRange(y_min, y_max, padding=0)

        # Reposition band labels at bottom of visible range
        for label in self.band_labels:
            label.setPos(label.pos().x(), y_min + 1)

    def _update_channel_overlays(self, freqs: np.ndarray, psd_array: np.ndarray):
        """Update or create per-channel overlay curves."""
        n_channels = len(psd_array)

        # Remove excess curves
        while len(self.channel_curves) > n_channels:
            curve = self.channel_curves.pop()
            self.plot_item.removeItem(curve)

        # Add missing curves
        while len(self.channel_curves) < n_channels:
            curve = self.plot_item.plot(pen=pg.mkPen(*PSD_CHANNEL_OVERLAY, width=1))
            curve.setZValue(-5)
            self.channel_curves.append(curve)

        # Update data and visibility
        for i, curve in enumerate(self.channel_curves):
            curve.setData(freqs, psd_array[i])
            curve.setVisible(self._show_channels)

    def update_channel_selection(self, selected_channels: list[int]):
        """Sync displayed channels with EEG channel slider."""
        self.selected_channels = selected_channels
        self._last_update_time = 0.0  # Force immediate recompute

    def create_controls_widget(self) -> QtWidgets.QWidget:
        """Return a widget with PSD display controls."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        label = QtWidgets.QLabel("PSD:")
        label.setStyleSheet(LABEL_STYLES["subtitle"])
        layout.addWidget(label)

        channels_cb = QtWidgets.QCheckBox("Show Channels")
        channels_cb.setChecked(self._show_channels)
        channels_cb.setStyleSheet(DASHBOARD_CHECKBOX_STYLE)
        channels_cb.toggled.connect(self._on_show_channels_toggled)
        layout.addWidget(channels_cb)

        bands_cb = QtWidgets.QCheckBox("Band Regions")
        bands_cb.setChecked(self._show_bands)
        bands_cb.setStyleSheet(DASHBOARD_CHECKBOX_STYLE)
        bands_cb.toggled.connect(self._on_show_bands_toggled)
        layout.addWidget(bands_cb)

        layout.addStretch()
        return widget

    def _on_show_channels_toggled(self, checked: bool):
        self._show_channels = checked
        for curve in self.channel_curves:
            curve.setVisible(checked)

    def _on_show_bands_toggled(self, checked: bool):
        self._show_bands = checked
        for region in self.band_regions:
            region.setVisible(checked)
        for label in self.band_labels:
            label.setVisible(checked)

    def clear_plots(self):
        """Reset for reconnect."""
        if self.plot_widget:
            self.plot_widget.clear()
        self.plot_item = None
        self.mean_curve = None
        self.channel_curves = []
        self.band_regions = []
        self.band_labels = []
        self._last_update_time = 0.0
