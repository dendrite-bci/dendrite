#!/usr/bin/env python
"""
BMI Visualization Dashboard

Minimal coordinator that delegates to specialized managers.
"""

import logging
import queue
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.auxiliary.dashboard.backend import (
    DEFAULT_STREAM_NAME,
    DataBufferManager,
    ModeManager,
    OptimizedLSLReceiver,
    SignalQualityAnalyzer,
)
from dendrite.auxiliary.dashboard.plot_managers import (
    EEGPlotManager,
    ERPPlotManager,
    EventPlotManager,
    ModalityPlotManager,
    NeurofeedbackPlotManager,
    PerformancePlotManager,
    PSDPlotManager,
)
from dendrite.auxiliary.dashboard.plot_managers.async_plots import AsyncPlotManager
from dendrite.auxiliary.dashboard.widgets import (
    CHANNELS_PER_PAGE,
    ChannelConfigPopup,
    QualityStripWidget,
    StatusBar,
    UIHelpers,
)
from dendrite.constants import (
    DEFAULT_BUFFER_SIZE,
    MODE_ASYNCHRONOUS,
    MODE_NEUROFEEDBACK,
    MODE_SYNCHRONOUS,
)
from dendrite.gui.styles.widget_styles import WidgetStyles, apply_app_styles
from dendrite.gui.utils import set_app_icon

UI_UPDATE_INTERVAL_MS = 50  # 20 FPS for smooth real-time visualization
MAX_QUEUE_ITEMS_PER_UPDATE = 300
PERFORMANCE_HISTORY_LENGTH = 200
QUALITY_UPDATE_INTERVAL_MS = 2000  # Signal quality analysis at 0.5 Hz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Dashboard(QtWidgets.QWidget):
    """Minimal dashboard coordinator - delegates everything to managers"""

    reconnect_request_signal = QtCore.pyqtSignal()

    def __init__(
        self,
        plot_queue: queue.Queue,
        sample_rate: int,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        parent=None,
    ):
        super().__init__(parent)
        self.plot_queue = plot_queue
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        # Data managers
        self.data_manager = DataBufferManager(sample_rate, buffer_size)

        # Performance data storage
        self.sync_performance_data: dict[str, dict[str, Any]] = {}

        # Channel pagination state
        self._current_page = 0
        self._channels_per_page = CHANNELS_PER_PAGE
        self._active_preset_prefix = ""  # Empty = all channels
        self._filtered_channel_indices: list[int] = []  # Indices after preset filter
        self._total_channels = 0
        self._channel_labels: list[str] = []

        self._channel_update_timer = QtCore.QTimer()
        self._channel_update_timer.setSingleShot(True)
        self._channel_update_timer.setInterval(150)  # 150ms debounce
        self._channel_update_timer.timeout.connect(self._apply_channel_update)

        self._setup_ui()

        self._psd_controls_widget: QtWidgets.QWidget | None = None

        self.eeg_plot_manager = EEGPlotManager(self.eeg_plots, self.data_manager)
        self.psd_plot_manager = PSDPlotManager(self.psd_plots, self.data_manager)
        self.event_plot_manager = EventPlotManager(self.events_plots, self.data_manager)
        self.modality_plot_manager = ModalityPlotManager(self.mod_plots, self.data_manager)
        self.neurofeedback_plot_manager = NeurofeedbackPlotManager(self)

        # Signal quality analysis
        self.quality_analyzer = SignalQualityAnalyzer(sample_rate)
        self._quality_timer = QtCore.QTimer(self)
        self._quality_timer.timeout.connect(self._update_signal_quality)
        self._quality_timer.start(QUALITY_UPDATE_INTERVAL_MS)

        self.mode_manager = ModeManager(self.clf_stack, self.clf_tabs_container)
        self.performance_managers: dict[str, PerformancePlotManager] = {}
        self.erp_managers: dict[str, ERPPlotManager] = {}
        self.async_managers: dict[str, AsyncPlotManager] = {}  # Unified async managers

        self._perf_data_changed: dict[str, bool] = {}
        self._erp_data_changed: dict[str, bool] = {}

        # Register handlers with mode manager
        self.mode_manager.register_handlers(
            performance_handler=self._handle_performance_data,
            erp_handler=self._handle_erp_data,
            async_handler=self._handle_async_data,
            neurofeedback_handler=lambda mode_name,
            data: self.neurofeedback_plot_manager.update_features(mode_name, data),
        )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(UI_UPDATE_INTERVAL_MS)

        logging.info(
            f"Dashboard initialized: {buffer_size} samples, {UI_UPDATE_INTERVAL_MS}ms updates"
        )

    def _setup_ui(self):
        """Setup UI structure"""
        # Global styles already applied via apply_app_styles()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Status bar at top (always visible, includes reconnect button)
        self.status_bar = StatusBar()
        self.status_bar.reconnect_clicked.connect(self._reconnect_and_reset)
        layout.addWidget(self.status_bar)

        # Setup sections first
        eeg_group = self._setup_eeg_section()
        self.mod_group = self._setup_modality_section()
        self.events_group = self._setup_events_section()
        clf_group = self._setup_classifier_section()

        # Horizontal splitter: left (EEG/signals) | right (Neural Decoding)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        # Left panel: scroll area for EEG + nested sections
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setStyleSheet(WidgetStyles.scrollarea)
        left_container = QtWidgets.QWidget()
        left_container.setStyleSheet(WidgetStyles.container(bg="panel"))
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        left_layout.addWidget(eeg_group, stretch=3)
        left_layout.addWidget(self.events_group, stretch=0)
        left_layout.addWidget(self.mod_group, stretch=0)
        left_layout.addStretch(1)
        left_scroll.setWidget(left_container)

        # Right panel: Neural Decoding (scroll area isolates geometry from splitter)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right_scroll.setStyleSheet(WidgetStyles.scrollarea)
        right_container = QtWidgets.QWidget()
        right_container.setStyleSheet(WidgetStyles.container(bg="panel"))
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(clf_group, stretch=1)
        right_container.setMinimumWidth(250)
        right_scroll.setWidget(right_container)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([900, 300])  # 3:1 ratio (75%:25%)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _setup_eeg_section(self):
        """Setup EEG section with time-series and PSD plots."""
        eeg_group, self.eeg_plots_container = UIHelpers.create_collapsible_section(
            "EEG", None, start_expanded=True
        )

        controls_widget, self.channel_controls = UIHelpers.create_channel_pagination_controls()
        self.channel_controls["prev_btn"].clicked.connect(lambda: self._on_page_changed(-1))
        self.channel_controls["next_btn"].clicked.connect(lambda: self._on_page_changed(1))
        self.channel_controls["config_btn"].clicked.connect(self._show_channel_config_popup)
        self.eeg_plots_container.addWidget(controls_widget)

        self.quality_strip = QualityStripWidget()
        self.eeg_plots_container.addWidget(self.quality_strip)

        self.eeg_plots = pg.GraphicsLayoutWidget()
        self.eeg_plots.setBackground(None)
        self.eeg_plots.setMinimumHeight(100)
        self.eeg_plots.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.eeg_plots_container.addWidget(self.eeg_plots)

        # PSD plot below EEG time-series
        self.psd_plots = pg.GraphicsLayoutWidget()
        self.psd_plots.setBackground(None)
        self.psd_plots.setMinimumHeight(150)
        self.psd_plots.setMaximumHeight(250)
        self.psd_plots.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.eeg_plots_container.addWidget(self.psd_plots)

        return eeg_group

    def _setup_modality_section(self):
        """Setup modality section and return the group widget."""
        mod_group, self.mod_plots_container = UIHelpers.create_collapsible_section(
            "Additional Signals", None, start_expanded=False
        )
        self.mod_plots = pg.GraphicsLayoutWidget()
        self.mod_plots.setBackground(None)
        self.mod_plots.setMinimumHeight(100)
        self.mod_plots.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,  # Only use needed height
        )
        self.mod_plots_container.addWidget(self.mod_plots)
        return mod_group

    def _setup_events_section(self):
        """Setup events section and return the group widget."""
        events_group, self.events_plots_container = UIHelpers.create_collapsible_section(
            "Events", None, start_expanded=False
        )
        self.events_plots = pg.GraphicsLayoutWidget()
        self.events_plots.setBackground(None)
        self.events_plots.setMinimumHeight(50)
        self.events_plots.setMaximumHeight(120)
        self.events_plots.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,  # Only use needed height
        )
        self.events_plots_container.addWidget(self.events_plots)
        return events_group

    def _setup_classifier_section(self):
        """Setup classifier section with PillNavigation and stacked content, return group widget."""
        clf_group, self.clf_tabs_container = UIHelpers.create_collapsible_section(
            "Neural Decoding", None, start_expanded=True
        )
        # Navigation will be added dynamically by ModeManager when first mode registers
        # Stack widget holds the mode content panels
        self.clf_stack = QtWidgets.QStackedWidget()
        self.clf_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,  # Respects children's size hints
        )
        self.clf_tabs_container.addWidget(self.clf_stack, stretch=1)
        return clf_group

    def _process_queue(self):
        """Process items from queue"""
        count = 0
        try:
            while count < MAX_QUEUE_ITEMS_PER_UPDATE:
                item = self.plot_queue.get_nowait()
                # ModeOutputPacket uses 'type', raw_data uses 'type' via output_type key
                payload_type = item.get("output_type") or item.get("type")

                if payload_type == "raw_data":
                    if not self.data_manager.initialized:
                        self._initialize_from_data(item)
                    self.data_manager.append_raw_data(item)
                    # Record data received for stale detection
                    if self.status_bar:
                        self.status_bar.record_data_received()
                elif payload_type == "mode_history":
                    self._process_mode_history(item)
                elif payload_type:
                    self._process_mode_data(item)

                count += 1
        except queue.Empty:
            pass

    def _initialize_from_data(self, item: dict[str, Any]):
        """Initialize from first data packet"""
        if not self.data_manager.initialize_from_raw_data(item):
            return

        self.eeg_plot_manager.initialize_plots()
        self.psd_plot_manager.initialize_plots()
        if self._psd_controls_widget is not None:
            self.eeg_plots_container.removeWidget(self._psd_controls_widget)
            self._psd_controls_widget.deleteLater()
        self._psd_controls_widget = self.psd_plot_manager.create_controls_widget()
        self.eeg_plots_container.addWidget(self._psd_controls_widget)
        self.event_plot_manager.initialize_plots()
        self.modality_plot_manager.initialize_plots()

        # Update quality analyzer with actual stream sample rate
        self.quality_analyzer.sample_rate = self.data_manager.sample_rate

        # Hide modality section if no additional signals
        if self.data_manager.num_modalities == 0:
            self.mod_group.hide()

        # Expand events section if stream includes markers
        if "markers" in item.get("data", {}):
            button = self.events_group.layout().itemAt(0).widget()
            button.setChecked(True)

        # Initialize channel pagination
        if self.channel_controls:
            self._channel_labels = self.data_manager.eeg_channel_labels
            self._total_channels = len(self._channel_labels)
            self._filtered_channel_indices = list(range(self._total_channels))
            self._current_page = 0
            self._active_preset_prefix = ""
            self._update_pagination_ui()
            self._apply_channel_update()

    def _process_mode_history(self, history_payload: dict[str, Any]):
        """Process mode history packets"""
        mode_name = history_payload.get("mode_name", "unknown")
        mode_type = history_payload.get("mode_type", "unknown")
        packets = history_payload.get("packets", [])

        # Ensure mode is registered and UI initialized before processing history
        is_new = self.mode_manager.register_mode(mode_name, mode_type)
        if is_new:
            self._initialize_mode_ui(mode_name, mode_type)

        # Now process historical packets
        for packet in packets:
            if "mode_name" not in packet:
                packet["mode_name"] = mode_name
            if "mode_type" not in packet:
                packet["mode_type"] = mode_type
            # Route directly without registering again
            self.mode_manager.route_data(packet)

    def _process_mode_data(self, item: dict[str, Any]):
        """Process mode output data"""
        mode_name = item.get("mode_name")
        mode_type = item.get("mode_type")

        if not mode_name or not mode_type:
            return

        # Register new modes
        is_new = self.mode_manager.register_mode(mode_name, mode_type)
        if is_new:
            self._initialize_mode_ui(mode_name, mode_type)

        # Route to appropriate handler
        self.mode_manager.route_data(item)

    def _initialize_mode_ui(self, mode_name: str, mode_type: str):
        """Initialize UI for new mode"""
        if mode_type == MODE_NEUROFEEDBACK:
            self.neurofeedback_plot_manager.initialize_for_mode(mode_name, self.mode_manager)
        elif mode_type in [MODE_SYNCHRONOUS, MODE_ASYNCHRONOUS]:
            # Create managers BEFORE creating UI so controls can access them
            if mode_type == MODE_SYNCHRONOUS:
                # Create placeholder managers that will be initialized with widgets after tab creation
                num_eeg_channels = (
                    len(self.data_manager.eeg_channel_labels)
                    if self.data_manager.initialized
                    else 64
                )
                default_erp_channels = set(range(min(4, num_eeg_channels)))

                # Create ERP manager with placeholder widget (will be replaced)
                erp_manager = ERPPlotManager(None, self.data_manager, self.sample_rate)
                erp_manager.selected_channels = default_erp_channels
                self.erp_managers[mode_name] = erp_manager

                # Create performance manager with placeholder widget (will be replaced)
                perf_manager = PerformancePlotManager(None)
                self.performance_managers[mode_name] = perf_manager

            # Now create the tab - controls can access managers
            mode_ui = self.mode_manager.create_mode_tab(
                mode_name,
                mode_type,
                create_erp_controls=self._create_erp_controls
                if mode_type == MODE_SYNCHRONOUS
                else None,
                create_perf_controls=self._create_perf_controls
                if mode_type == MODE_SYNCHRONOUS
                else None,
            )

            # Update managers with actual widgets
            if mode_type == MODE_SYNCHRONOUS:
                perf_widget = mode_ui.get("Performance", {}).get("widget")
                erp_widget = mode_ui.get("ERP", {}).get("widget")
                if perf_widget and mode_name in self.performance_managers:
                    self.performance_managers[mode_name].plot_widget = perf_widget
                if erp_widget and mode_name in self.erp_managers:
                    self.erp_managers[mode_name].plot_widget = erp_widget
            elif mode_type == MODE_ASYNCHRONOUS:
                pred_widget = mode_ui.get("PredictionTrace", {}).get("widget")
                if pred_widget:
                    # Use new unified async manager
                    self.async_managers[mode_name] = AsyncPlotManager(pred_widget)
                    self.async_managers[mode_name].initialize_for_mode(mode_name)

    def _create_erp_controls(self, mode_name: str) -> QtWidgets.QWidget:
        """Create ERP controls widget"""
        if mode_name in self.erp_managers:
            return self.erp_managers[mode_name].create_controls_widget(mode_name)
        return QtWidgets.QWidget()

    def _create_perf_controls(self, mode_name: str) -> QtWidgets.QWidget:
        """Create performance metrics controls widget"""
        if mode_name in self.performance_managers:
            return self.performance_managers[mode_name].create_controls_widget(mode_name)
        return QtWidgets.QWidget()

    def _handle_performance_data(self, mode_name: str, item: dict[str, Any]):
        """Handle performance data"""
        if mode_name not in self.performance_managers:
            return

        payload_content = item.get("data", {})
        if not payload_content:
            logging.warning(f"Performance [{mode_name}]: No 'data' field in item")
            return

        # Get output type and name
        effective_output_type = item.get("type", "unknown")
        sync_mode_output_name = payload_content.get("output_name", "default_metric_set")
        metric_set_key = sync_mode_output_name

        if effective_output_type == "classification":
            metric_set_key = "async_classification_metrics"

        # Initialize performance data storage
        if mode_name not in self.sync_performance_data:
            self.sync_performance_data[mode_name] = {}

        if metric_set_key not in self.sync_performance_data[mode_name]:
            self.sync_performance_data[mode_name][metric_set_key] = {
                "indices": deque(maxlen=PERFORMANCE_HISTORY_LENGTH),
                "metrics": {},
                "processed_ids": set(),
            }

        rec = self.sync_performance_data[mode_name][metric_set_key]

        # Create unique packet ID to avoid duplicates
        packet_timestamp = item.get("timestamp", time.time())
        distinguishing_value = payload_content.get(
            "prediction", payload_content.get("value", time.time())
        )
        packet_id = f"{packet_timestamp}_{distinguishing_value}"

        if packet_id in rec["processed_ids"]:
            return

        rec["processed_ids"].add(packet_id)

        # Clear processed_ids when it exceeds limit (old IDs won't be seen again)
        if len(rec["processed_ids"]) > PERFORMANCE_HISTORY_LENGTH * 2:
            rec["processed_ids"].clear()

        next_idx = len(rec["indices"]) + 1
        rec["indices"].append(next_idx)

        metrics_to_plot = {}

        if effective_output_type == "performance":  # SynchronousMode
            # Extract all metrics from SyncMode_MetricsPayload (top-level fields)
            metric_fields = [
                "accuracy",
                "confidence",
                "chance_level",
                "adaptive_chance_level",
                "cohens_kappa",
            ]
            for field in metric_fields:
                if field in payload_content:
                    value = payload_content[field]
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        metrics_to_plot[field] = value

        elif effective_output_type == "classification":  # AsyncMode
            async_metrics = payload_content.get("classification_metrics", {})
            if "prequential_accuracy" in async_metrics:
                metrics_to_plot["prequential_accuracy"] = async_metrics["prequential_accuracy"]
            if "chance_level" in async_metrics:
                metrics_to_plot["chance_level"] = async_metrics["chance_level"]

        # Store metrics
        for metric_key, metric_val in metrics_to_plot.items():
            if metric_key not in rec["metrics"]:
                rec["metrics"][metric_key] = deque(maxlen=PERFORMANCE_HISTORY_LENGTH)
            rec["metrics"][metric_key].append(metric_val)

        # Mark data as changed for this mode
        self._perf_data_changed[mode_name] = True

    def _handle_erp_data(self, mode_name: str, item: dict[str, Any]):
        """Handle ERP data"""
        if mode_name in self.erp_managers:
            self.erp_managers[mode_name].handle_erp_data(mode_name, item)
            self._erp_data_changed[mode_name] = True

    def _handle_async_data(self, mode_name: str, item: dict[str, Any]):
        """Handle asynchronous mode data with unified async manager"""
        if mode_name in self.async_managers:
            self.async_managers[mode_name].handle_data(mode_name, item)

    def update_dashboard(self):
        """Main update loop with conditional updates for better performance"""
        self._process_queue()

        if not self.data_manager.initialized:
            return

        # Only update raw data plots if new data has arrived
        if self.data_manager.data_changed:
            self.eeg_plot_manager.update_plots()
            self.psd_plot_manager.update_plots()
            self.event_plot_manager.update_plots()
            self.modality_plot_manager.update_plots()
            self.data_manager.data_changed = False  # Reset flag

        # Update mode-specific plots only if data changed
        for mode_name, manager in self.performance_managers.items():
            if self._perf_data_changed.get(mode_name, False):
                if mode_name in self.sync_performance_data:
                    manager.update_plots(mode_name, self.sync_performance_data[mode_name])
                self._perf_data_changed[mode_name] = False

        for mode_name, manager in self.erp_managers.items():
            if self._erp_data_changed.get(mode_name, False):
                manager.update_plots(mode_name)
                self._erp_data_changed[mode_name] = False

        # Update async managers (unified system - has internal data-changed tracking)
        for mode_name, manager in self.async_managers.items():
            manager.update_plots(mode_name)

    def _on_page_changed(self, delta: int):
        """Handle page navigation."""
        if not self._filtered_channel_indices:
            return

        total_pages = self._get_total_pages()
        new_page = self._current_page + delta
        new_page = max(0, min(new_page, total_pages - 1))

        if new_page != self._current_page:
            self._current_page = new_page
            self._update_pagination_ui()
            self._channel_update_timer.start()

    def _show_channel_config_popup(self):
        """Show config popup near the config button."""
        popup = ChannelConfigPopup(self)
        popup.preset_selected.connect(self._on_preset_selected)
        popup.show_near(self.channel_controls["config_btn"])

    def _on_preset_selected(self, prefix: str):
        """Handle preset button click."""
        self._active_preset_prefix = prefix

        if prefix:
            # Filter to channels matching prefix
            self._filtered_channel_indices = [
                i
                for i, label in enumerate(self._channel_labels)
                if label.upper().startswith(prefix.upper())
            ]
        else:
            # "All" - reset to full channel list
            self._filtered_channel_indices = list(range(self._total_channels))

        # Reset to page 0 and update
        self._current_page = 0
        self._update_pagination_ui()
        self._apply_channel_update()

    def _get_total_pages(self) -> int:
        """Get total number of pages for current filter."""
        n = len(self._filtered_channel_indices)
        return max(1, (n + self._channels_per_page - 1) // self._channels_per_page)

    def _get_current_page_channels(self) -> list[int]:
        """Get channel indices for current page."""
        start = self._current_page * self._channels_per_page
        end = start + self._channels_per_page
        return self._filtered_channel_indices[start:end]

    def _update_pagination_ui(self):
        """Update pagination controls based on current state."""
        if not self.channel_controls:
            return

        total_pages = self._get_total_pages()
        page_channels = self._get_current_page_channels()

        # Update page label
        self.channel_controls["page_label"].setText(
            f"Page {self._current_page + 1} of {total_pages}"
        )

        # Update button states
        self.channel_controls["prev_btn"].setEnabled(self._current_page > 0)
        self.channel_controls["next_btn"].setEnabled(self._current_page < total_pages - 1)

        # Update summary label
        if page_channels:
            # Show channel range using 1-based indexing for display
            first_ch = page_channels[0] + 1
            last_ch = page_channels[-1] + 1
            total_filtered = len(self._filtered_channel_indices)

            if self._active_preset_prefix:
                self.channel_controls["summary_label"].setText(
                    f"Channels {first_ch}-{last_ch} ({total_filtered} {self._active_preset_prefix}* channels)"
                )
            else:
                self.channel_controls["summary_label"].setText(
                    f"Channels {first_ch}-{last_ch} of {self._total_channels}"
                )
        else:
            self.channel_controls["summary_label"].setText("No channels")

    def _apply_channel_update(self):
        """Apply the debounced channel update to plots."""
        selected_channels = self._get_current_page_channels()

        if self.eeg_plot_manager and self.data_manager.initialized and selected_channels:
            self.eeg_plot_manager.update_channel_selection(selected_channels)
            self.psd_plot_manager.update_channel_selection(selected_channels)

        if selected_channels:
            first_ch = selected_channels[0] + 1
            last_ch = selected_channels[-1] + 1
            logging.info(
                f"Updated EEG selection: channels {first_ch}-{last_ch} ({len(selected_channels)} channels)"
            )

    def _update_signal_quality(self):
        """Periodic signal quality analysis (called by _quality_timer)."""
        if not self.data_manager.initialized or not self.data_manager.eeg_buffers:
            return

        results = self.quality_analyzer.analyze(
            self.data_manager.eeg_buffers,
            self.data_manager.eeg_channel_labels,
        )
        self.quality_strip.update_quality(results)
        self.eeg_plot_manager.update_quality_indicators(results)
        for erp_manager in self.erp_managers.values():
            erp_manager.update_quality(results)

    def _reconnect_and_reset(self):
        """Reset dashboard and reconnect"""
        logging.info("Dashboard: Reconnect & Reset triggered")

        self.reconnect_request_signal.emit()

        # Reset pagination state
        self._current_page = 0
        self._active_preset_prefix = ""
        self._filtered_channel_indices = []
        self._total_channels = 0
        self._channel_labels = []

        self.data_manager.clear_all_buffers()
        self.sync_performance_data.clear()

        self.eeg_plot_manager.clear_plots()
        self.psd_plot_manager.clear_plots()
        self.event_plot_manager.clear_plots()
        self.modality_plot_manager.clear_plots()
        self.neurofeedback_plot_manager.clear_all_data()
        self.quality_strip.update_quality([])

        for manager in self.performance_managers.values():
            manager.clear_plots()
        for manager in self.erp_managers.values():
            manager.clear_plots()
        for manager in self.async_managers.values():
            manager.clear_plots()

        self.performance_managers.clear()
        self.erp_managers.clear()
        self.async_managers.clear()

        self.mode_manager.clear_all()

        while not self.plot_queue.empty():
            try:
                self.plot_queue.get_nowait()
            except queue.Empty:
                break

        logging.info("Dashboard: Reset complete")


class MainWindow(QtWidgets.QMainWindow):
    """Main window"""

    def __init__(self, stream_name=DEFAULT_STREAM_NAME):
        super().__init__()
        self.setWindowTitle("Dendrite Dashboard")
        self.setGeometry(50, 50, 1400, 900)

        # Main window background already handled by apply_app_styles()

        self.plot_queue = queue.Queue(maxsize=2000)
        # Initial sample rate (updated dynamically when data arrives)
        self.dashboard = Dashboard(self.plot_queue, 500, DEFAULT_BUFFER_SIZE)
        self.setCentralWidget(self.dashboard)

        self.receiver = OptimizedLSLReceiver(stream_name)
        self.receiver.new_payload_signal.connect(
            self._handle_payload, QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.dashboard.reconnect_request_signal.connect(self._handle_reconnect)

        # Connect receiver signals to dashboard status/notifications
        self.receiver.connection_changed_signal.connect(self._handle_connection_change)
        self.receiver.stream_info_signal.connect(self._handle_stream_info)

        self.qt_status_bar = self.statusBar()
        self.qt_status_bar.showMessage("Initializing...")

    def _handle_payload(self, payload: dict):
        """Handle new payload"""
        try:
            self.plot_queue.put_nowait(payload)
            qsize = self.plot_queue.qsize()
            if qsize > 1500:
                self.qt_status_bar.showMessage(f"Warning: Queue size {qsize}", 2000)
        except queue.Full:
            logging.warning("Plot queue full!")

    def _handle_reconnect(self):
        """Handle reconnect request"""
        if self.receiver:
            self.receiver.trigger_reconnect()
            self.qt_status_bar.showMessage("Reconnecting...", 3000)

    def _handle_connection_change(self, connected: bool):
        """Handle connection state change from receiver."""
        if self.dashboard.status_bar:
            self.dashboard.status_bar.set_connected(connected)

    def _handle_stream_info(self, info: str):
        """Handle stream info update from receiver."""
        if self.dashboard.status_bar:
            self.dashboard.status_bar.set_stream_info(info)

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Clean shutdown"""
        self.dashboard.timer.stop()
        self.dashboard._quality_timer.stop()
        self.receiver.stop()
        event.accept()


def main():
    """Entry point"""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Dendrite Dashboard")

    # Apply unified global styles
    apply_app_styles(app)

    try:
        set_app_icon(app, "icons/dashboard.svg")
    except ImportError:
        pass

    stream_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STREAM_NAME
    window = MainWindow(stream_name)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
