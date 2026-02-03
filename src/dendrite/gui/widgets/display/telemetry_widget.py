"""
Telemetry Widget

A clean, minimal telemetry panel showing key operational metrics.
Displays system status, resource usage, and mode performance without plots.

Resource Monitoring Strategy:
- Process caching: psutil.Process objects are cached by PID for accurate
  cpu_percent() readings. Fresh Process objects give unreliable values since
  cpu_percent(interval=None) measures delta from last call on same object.
- USS memory: Uses memory_full_info().uss (Unique Set Size) instead of RSS.
  USS shows actual unique memory per process, avoiding inflated values for
  forked processes that share memory pages with parent. Falls back to RSS
  if USS unavailable.
- CPU normalization: Raw cpu_percent() can exceed 100% on multi-core systems
  (200% = 2 cores). Normalized to system capacity for intuitive 0-100% display.
"""

import os
import time

import psutil
from PyQt6 import QtCore, QtWidgets

from dendrite.constants import METRIC_THRESHOLDS, STALE_DATA_THRESHOLD_SEC
from dendrite.gui.styles.design_tokens import (
    BG_PANEL,
    SEPARATOR_SUBTLE,
    STATUS_DANGER,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import WidgetStyles
from dendrite.gui.widgets.common.flow_layout import FlowLayout
from dendrite.gui.widgets.common.indeterminate_progress_bar import IndeterminateProgressBar
from dendrite.gui.widgets.common.latency_sparkline import LatencySparkline
from dendrite.gui.widgets.common.resource_progress_bar import CompactResourceBar
from dendrite.utils.logger_central import get_logger
from dendrite.utils.state_keys import (
    channel_quality_key,
    e2e_latency_key,
    mode_metric_key,
    stream_connected_key,
    stream_latency_key,
    stream_timestamp_key,
    streamer_metric_key,
    viz_bandwidth_key,
    viz_consumers_key,
)

logger = get_logger(__name__)


class BaseStreamCard(QtWidgets.QFrame):
    """Base card with sparkline and threshold-based coloring."""

    def __init__(self, low_threshold: float, high_threshold: float, parent=None):
        super().__init__(parent)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        # Subclasses must set self.value_label and self.sparkline

    def _create_sparkline(self, width: int, height: int = 20) -> LatencySparkline:
        return LatencySparkline(
            max_samples=30,
            width=width,
            height=height,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

    def _update_value(self, value: float, unit: str, decimals: int, font_size: int = 12):
        """Update displayed value and add to sparkline."""
        color = self._get_value_color(value)
        text = f"{int(value)}{unit}" if decimals == 0 else f"{value:.{decimals}f}{unit}"
        self.value_label.setText(text)
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: {font_size}px; font-weight: 500;"
        )
        self.sparkline.clear_force_color()
        self.sparkline.add_sample(value)

    def set_inactive(self, font_size: int = 12):
        """Show inactive state (no data)."""
        self.value_label.setText("--")
        self.value_label.setStyleSheet(f"color: {TEXT_DISABLED}; font-size: {font_size}px;")

    def set_idle(self, font_size: int = 11):
        """Show idle state (connected but no recent data)."""
        self.value_label.setText("IDLE")
        self.value_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {font_size}px;")

    def set_dropped(self, text: str = "DROPPED", font_size: int = 11):
        """Show dropped state (stream stopped sending data)."""
        self.value_label.setText(text)
        self.value_label.setStyleSheet(
            f"color: {STATUS_DANGER}; font-size: {font_size}px; font-weight: 600;"
        )
        self.sparkline.set_force_color(STATUS_DANGER)

    def clear(self):
        """Clear sparkline and reset value."""
        self.sparkline.clear()
        self.set_inactive()

    def _get_value_color(self, value: float) -> str:
        """Get color based on value thresholds."""
        if value >= self.high_threshold:
            return STATUS_DANGER
        elif value >= self.low_threshold:
            return STATUS_WARNING_ALT
        return STATUS_SUCCESS


class StreamHealthCard(BaseStreamCard):
    """Mini card showing stream name, value, and sparkline."""

    def __init__(
        self,
        label: str,
        unit: str = "ms",
        decimals: int = 1,
        low_threshold: float = 10,
        high_threshold: float = 30,
        parent=None,
    ):
        super().__init__(low_threshold, high_threshold, parent)
        self.unit = unit
        self.decimals = decimals
        self.setFixedWidth(120)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(4)

        label_widget = QtWidgets.QLabel(label)
        label_widget.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        top_row.addWidget(label_widget)

        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 12px; font-weight: 500;")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        top_row.addWidget(self.value_label)

        layout.addLayout(top_row)

        self.sparkline = self._create_sparkline(104)
        layout.addWidget(self.sparkline)

    def update_value(self, value: float) -> None:
        """Update displayed value and add to sparkline."""
        self._update_value(value, self.unit, self.decimals)

    def set_inactive(self):
        super().set_inactive(font_size=12)

    def set_dropped(self):
        super().set_dropped("DROPPED", font_size=11)


class StreamInfoCard(BaseStreamCard):
    """Mini card showing stream type, modality breakdown, rate, and latency sparkline."""

    def __init__(
        self,
        stream_type: str,
        modality_counts: dict,
        sample_rate: float,
        low_threshold: float = 10,
        high_threshold: float = 30,
        parent=None,
    ):
        super().__init__(low_threshold, high_threshold, parent)
        self.stream_type = stream_type
        self.sample_rate = sample_rate
        self.setFixedWidth(160)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(4)

        type_label = QtWidgets.QLabel(stream_type.upper())
        type_label.setStyleSheet(f"color: {TEXT_LABEL}; font-size: 11px; font-weight: bold;")
        top_row.addWidget(type_label)

        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        top_row.addWidget(self.value_label)

        layout.addLayout(top_row)

        parts = [f"{count} {mod}" for mod, count in sorted(modality_counts.items()) if count > 0]
        modality_str = ", ".join(parts) if parts else "0 ch"

        if sample_rate >= 1000:
            rate_str = f"{int(sample_rate / 1000)}kHz"
        elif sample_rate > 0:
            rate_str = f"{int(sample_rate)}Hz"
        else:
            rate_str = "irregular"

        details_label = QtWidgets.QLabel(f"{modality_str} @ {rate_str}")
        details_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(details_label)

        self.sparkline = self._create_sparkline(140)
        layout.addWidget(self.sparkline)

    def update_latency(self, latency_ms: float):
        """Update displayed latency and add to sparkline."""
        self._update_value(latency_ms, "ms", 0, font_size=11)

    def set_inactive(self):
        super().set_inactive(font_size=11)

    def set_idle(self):
        super().set_idle(font_size=11)

    def set_dropped(self):
        super().set_dropped("DROP", font_size=11)

    def set_stale(self, latency_ms: float):
        """Show stale latency (dimmed, data not fresh)."""
        self.value_label.setText(f"{int(latency_ms)}ms")
        self.value_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.sparkline.set_force_color(TEXT_MUTED)


class ResourceWorker(QtCore.QObject):
    """Background worker for expensive resource collection (psutil calls)."""

    finished = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._process_cache = {}
        self._gui_process = psutil.Process()
        self._mode_pids = {}
        self._system_pids = {}

    def set_pids(self, mode_pids: dict, system_pids: dict):
        """Store PIDs for next collection (called from main thread before signal)."""
        self._mode_pids = mode_pids
        self._system_pids = system_pids

    @QtCore.pyqtSlot()
    def collect(self):
        """Collect CPU/memory in background thread."""
        result = {"total_cpu": 0.0, "total_memory": 0.0, "breakdown": []}

        # GUI process
        try:
            result["total_cpu"] += self._gui_process.cpu_percent(interval=None)
            gui_mem = self._get_memory(self._gui_process)
            result["total_memory"] += gui_mem
            result["breakdown"].append(("GUI", gui_mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Mode processes
        for name, pid in self._mode_pids.items():
            cpu, mem = self._get_resources(pid)
            result["total_cpu"] += cpu
            result["total_memory"] += mem
            if mem > 0:
                result["breakdown"].append((name, mem))

        # System processes with children (the expensive part)
        for name, pid in self._system_pids.items():
            proc_mem = 0.0
            cpu, mem = self._get_resources(pid)
            result["total_cpu"] += cpu
            proc_mem += mem

            try:
                parent = self._get_cached_process(pid)
                if parent:
                    for child in parent.children(recursive=True):
                        cpu, mem = self._get_resources(child.pid)
                        result["total_cpu"] += cpu
                        proc_mem += mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            result["total_memory"] += proc_mem
            if proc_mem > 0:
                result["breakdown"].append((name.capitalize(), proc_mem))

        self.finished.emit(result)

    def _get_cached_process(self, pid: int):
        """Get or create cached Process for accurate cpu_percent."""
        if pid not in self._process_cache:
            try:
                self._process_cache[pid] = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return None
        proc = self._process_cache.get(pid)
        if proc and not proc.is_running():
            del self._process_cache[pid]
            return None
        return proc

    def _get_resources(self, pid: int) -> tuple:
        """Get CPU and memory usage for a specific process."""
        proc = self._get_cached_process(pid)
        if proc:
            try:
                return proc.cpu_percent(interval=None), self._get_memory(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return 0.0, 0.0

    def _get_memory(self, proc) -> float:
        """Get memory in MB, preferring USS over RSS."""
        try:
            return proc.memory_full_info().uss / (1024 * 1024)
        except (psutil.AccessDenied, AttributeError):
            return proc.memory_info().rss / (1024 * 1024)


class TelemetryWidget(QtWidgets.QWidget):
    """Telemetry panel displaying operational metrics in a clean, minimal layout."""

    UPDATE_INTERVAL = 1000  # Update every 1 second

    # Signal to trigger resource collection (cross-thread safe)
    _collect_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.start_time = None
        self.is_recording = False
        self.process = psutil.Process()
        self.system_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.cpu_count = os.cpu_count() or 1
        self.mode_pids = {}  # Maps mode instance name to process PID
        self.system_pids = {}  # Maps system process name to PID (DAQ, processor, saver, etc.)

        # Background worker for resource collection (avoids blocking GUI)
        self._cached_resources = {"total_cpu": 0, "total_memory": 0, "breakdown": []}
        self._resource_thread = QtCore.QThread()
        self._resource_worker = ResourceWorker()
        self._resource_worker.moveToThread(self._resource_thread)
        self._resource_worker.finished.connect(self._on_resources_ready)
        self._collect_requested.connect(self._resource_worker.collect)
        self._resource_thread.start()

        self.setup_ui()
        self.setup_update_timer()

    def setup_ui(self):
        """Set up compact telemetry UI."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(8)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(8)

        self.subtitle_label = QtWidgets.QLabel("Idle")
        self.subtitle_label.setStyleSheet(
            f"color: {TEXT_LABEL}; font-size: 13px; font-weight: 500;"
        )
        header_layout.addWidget(self.subtitle_label)

        header_layout.addStretch()

        self.elapsed_label = QtWidgets.QLabel("")
        self.elapsed_label.setStyleSheet(
            f"color: {TEXT_LABEL}; font-size: 13px; font-family: monospace;"
        )
        header_layout.addWidget(self.elapsed_label)

        main_layout.addLayout(header_layout)

        self.recording_bar = IndeterminateProgressBar()
        self.recording_bar.hide()
        main_layout.addWidget(self.recording_bar)

        main_layout.addSpacing(12)

        rec_header = QtWidgets.QLabel("STREAMS")
        rec_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        main_layout.addWidget(rec_header)

        # Stream info cards container
        self.stream_info_widget = QtWidgets.QWidget()
        self.stream_info_layout = FlowLayout(self.stream_info_widget, spacing=12)
        self.stream_info_cards = {}
        main_layout.addWidget(self.stream_info_widget)

        self.quality_label = QtWidgets.QLabel("")
        self.quality_label.setStyleSheet(f"color: {STATUS_WARNING_ALT}; font-size: 12px;")
        self.quality_label.hide()
        main_layout.addWidget(self.quality_label)

        main_layout.addSpacing(10)

        sys_header = QtWidgets.QLabel("SYSTEM")
        sys_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        main_layout.addWidget(sys_header)

        system_layout = QtWidgets.QHBoxLayout()
        system_layout.setSpacing(12)

        # Output stream cards (Dash, Preds, etc.)
        self.output_cards_widget = QtWidgets.QWidget()
        self.output_cards_layout = FlowLayout(self.output_cards_widget, spacing=6)
        self.output_cards = {}
        system_layout.addWidget(self.output_cards_widget, 1)

        # Resource bars
        resource_widget = QtWidgets.QWidget()
        resource_layout = QtWidgets.QVBoxLayout(resource_widget)
        resource_layout.setContentsMargins(0, 0, 0, 0)
        resource_layout.setSpacing(4)

        self.cpu_bar = CompactResourceBar("CPU")
        self.cpu_bar.set_thresholds(*METRIC_THRESHOLDS["cpu_percent"])
        resource_layout.addWidget(self.cpu_bar)

        self.mem_bar = CompactResourceBar("Mem")
        self.mem_bar.set_thresholds(*METRIC_THRESHOLDS["memory_percent"])
        self.mem_bar.maximum = self.system_ram_mb
        resource_layout.addWidget(self.mem_bar)

        system_layout.addWidget(resource_widget)
        main_layout.addLayout(system_layout)
        main_layout.addSpacing(14)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet(
            f"background: transparent; border-top: 1px solid {SEPARATOR_SUBTLE}; max-height: 1px;"
        )
        main_layout.addWidget(separator)

        main_layout.addSpacing(10)

        self.modes_header = QtWidgets.QLabel("MODES")
        self.modes_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        main_layout.addWidget(self.modes_header)

        main_layout.addSpacing(6)

        details_scroll = QtWidgets.QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setStyleSheet(WidgetStyles.scrollarea)

        details_widget = QtWidgets.QWidget()
        self.details_layout = FlowLayout(details_widget, spacing=8)

        details_scroll.setWidget(details_widget)
        main_layout.addWidget(details_scroll, 1)

        self.update_default_values()

    def setup_update_timer(self):
        """Set up timer for periodic updates."""
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(self.UPDATE_INTERVAL)

    def update_default_values(self):
        """Set default values when not recording."""
        self.subtitle_label.setText("Idle")
        self.elapsed_label.setText("")

        # Clear stream info cards
        self._clear_stream_info_cards()

        self.modes_header.setText("MODES")
        self.cpu_bar.set_value(0.0)
        self.mem_bar.set_value(0.0, "0 MB")

    def _clear_stream_info_cards(self):
        """Clear all stream info cards."""
        while self.stream_info_layout.count() > 0:
            item = self.stream_info_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.stream_info_cards.clear()

    def update_metrics(self):
        """Update all metrics based on current system state."""
        if not self.parent_window:
            return

        # Check if recording is active
        self.is_recording = self.parent_window.is_recording()

        if not self.is_recording:
            self.update_default_values()
            return

        # Update subtitle with study and recording info
        study_name, recording_name = self._get_study_info()
        self.subtitle_label.setText(f"{study_name} • {recording_name}")

        # Update separate sections
        self._update_session_section()
        self._update_data_section()
        self._update_channel_quality()
        self._update_stream_health()
        self._update_performance_section()

        # Update details section
        self._update_details()

    def _get_study_info(self) -> tuple:
        """Get study and recording names."""
        study_name = "Unknown"
        recording_name = "Unknown"
        if hasattr(self.parent_window, "general_params_widget"):
            study_name = (
                self.parent_window.general_params_widget.study_name_combo.currentText().strip()
                or "default_study"
            )
            recording_name = (
                self.parent_window.general_params_widget.file_name_input.text().strip()
                or "recording"
            )
        return study_name, recording_name

    def _update_session_section(self):
        """Update elapsed time in header."""
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def _update_data_section(self):
        """Update STREAMS section with per-stream info cards."""
        if not hasattr(self.parent_window, "stream_config_manager"):
            return

        streams = self.parent_window.stream_config_manager.get_streams()

        # Only rebuild if streams changed
        current_uids = set(streams.keys())
        cached_uids = set(self.stream_info_cards.keys())

        if current_uids != cached_uids:
            self._clear_stream_info_cards()

            low, high = METRIC_THRESHOLDS["stream_latency_ms"]

            for uid, stream in streams.items():
                # Count modalities within this stream
                modality_counts = {}
                for ch_type in stream.channel_types or []:
                    modality_counts[ch_type] = modality_counts.get(ch_type, 0) + 1

                card = StreamInfoCard(
                    stream_type=stream.type or "Unknown",
                    modality_counts=modality_counts,
                    sample_rate=stream.sample_rate or 0,
                    low_threshold=low,
                    high_threshold=high,
                )
                self.stream_info_cards[uid] = card
                self.stream_info_layout.addWidget(card)

        # Update latency for each card
        self._update_stream_latencies()

    def _update_channel_quality(self):
        """Show channel quality warning if bad channels detected."""
        shared_state = getattr(self.parent_window, "shared_state", None)
        quality = shared_state.get(channel_quality_key()) if shared_state else None
        bad_channels = quality.get("bad_channels_eeg", []) if quality else []

        if not bad_channels:
            self.quality_label.hide()
            return

        count = len(bad_channels)
        total_channels = quality.get("total_channels", 0)
        self.quality_label.setText(f"{count} bad channel{'s' if count > 1 else ''} detected")
        self.quality_label.setToolTip(
            f"Bad channels: {', '.join(map(str, bad_channels))}\nTotal EEG channels: {total_channels}"
        )
        self.quality_label.show()

    def _update_performance_section(self):
        """Update SYSTEM section with resources and mode count."""
        active_modes = 0
        if hasattr(self.parent_window, "mode_instance_badges"):
            active_modes = sum(
                1 for badge in self.parent_window.mode_instance_badges if badge.is_enabled()
            )

        self.modes_header.setText(f"MODES ({active_modes})")

        self._resource_worker.set_pids(self.mode_pids.copy(), self.system_pids.copy())
        self._collect_requested.emit()

        self._display_resources(self._cached_resources)

    def _on_resources_ready(self, resources: dict):
        """Handle resource data from background worker."""
        self._cached_resources = resources
        self._display_resources(resources)

    def _display_resources(self, resources: dict):
        """Update resource display with given data."""
        total_cpu = resources["total_cpu"]
        total_memory = resources["total_memory"]

        normalized_cpu = total_cpu / self.cpu_count
        self.cpu_bar.set_value(normalized_cpu)

        if total_memory >= 1024:
            mem_text = f"{total_memory / 1024:.1f} GB"
        else:
            mem_text = f"{total_memory:.0f} MB"
        self.mem_bar.set_value(total_memory, mem_text)

        tooltip_lines = ["Memory Usage"]
        for name, mem_mb in resources["breakdown"]:
            tooltip_lines.append(f"{name}: {mem_mb:.0f} MB")
        tooltip_lines.append(f"Total: {total_memory:.0f} MB")
        self.mem_bar.setToolTip("\n".join(tooltip_lines))

    def _update_stream_latencies(self):
        """Update latency values in stream info cards."""
        if not hasattr(self.parent_window, "shared_state") or not self.parent_window.shared_state:
            return

        sm = self.parent_window.shared_state

        for _uid, card in self.stream_info_cards.items():
            stream_type = card.stream_type.lower()
            latency = sm.get(stream_latency_key(stream_type))
            ts = sm.get(stream_timestamp_key(stream_type))
            connected = sm.get(stream_connected_key(stream_type), True)
            self._update_stream_card_state(card, latency, ts, connected)

    def _update_stream_card_state(
        self, card, latency: float | None, ts: float | None, connected: bool
    ) -> None:
        """Update stream card display state based on connection and data freshness."""
        if not connected:
            card.set_dropped()
            return

        has_timestamp = ts is not None and ts > 0
        is_fresh = has_timestamp and (time.time() - ts) <= STALE_DATA_THRESHOLD_SEC
        has_valid_latency = latency is not None and latency > 0

        if has_valid_latency and is_fresh:
            card.update_latency(latency)
            return

        if has_timestamp and not is_fresh:
            is_continuous = card.sample_rate > 0
            if is_continuous:
                card.set_dropped()
            elif has_valid_latency:
                card.set_stale(latency)
            else:
                card.set_inactive()
            return

        card.set_inactive()

    def _update_stream_health(self):
        """Update output stream health cards from SharedState."""
        if not hasattr(self.parent_window, "shared_state") or not self.parent_window.shared_state:
            return

        sm = self.parent_window.shared_state

        stream_labels = {
            "Viz": ("Dash", "bmi_visualization"),
            "LSL": ("Preds", "predictionstream"),
            "Socket": ("Socket", "socket"),
            "ZMQ": ("ZMQ", "zmq"),
            "ROS2": ("ROS2", "ros2"),
        }

        for key, (label, stream_prefix) in stream_labels.items():
            if key not in self.system_pids:
                continue

            bandwidth_key = streamer_metric_key(stream_prefix, "bandwidth_kbps")
            bandwidth = sm.get(bandwidth_key)

            if key == "Viz" and bandwidth is None:
                bandwidth = sm.get(viz_bandwidth_key())

            card_key = key.lower()
            self._get_or_create_card(
                self.output_cards,
                self.output_cards_layout,
                card_key,
                label,
                "KB/s",
                "output_bandwidth_kbps",
            )

            if bandwidth is not None:
                self.output_cards[card_key].update_value(bandwidth)
                tooltip_lines = [f"{label} Stream"]

                if key == "Viz":
                    consumers = sm.get(viz_consumers_key())
                    if consumers is not None:
                        tooltip_lines.append(
                            f"{consumers} connected client{'s' if consumers != 1 else ''}"
                        )

                tooltip_lines.append(f"{bandwidth:.1f} KB/s bandwidth")
                self.output_cards[card_key].setToolTip("\n".join(tooltip_lines))
            else:
                self.output_cards[card_key].set_inactive()

    def _format_rate(self, rate: float) -> str:
        """Format sample rate for display (500 -> '500', 1000 -> '1k')."""
        if rate == 0:
            return "—"
        elif rate >= 1000:
            return f"{int(rate / 1000)}k"
        else:
            return str(int(rate))

    def _update_details(self):
        """Update detailed mode instance information."""
        while self.details_layout.count() > 0:
            item = self.details_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not hasattr(self.parent_window, "mode_instance_badges"):
            return

        for badge in self.parent_window.mode_instance_badges:
            if not badge.is_enabled():
                continue

            mode_card = self._create_mode_detail_card(badge)
            self.details_layout.addWidget(mode_card)

    def _create_mode_detail_card(self, badge) -> QtWidgets.QWidget:
        """Create minimal mode display card."""
        container = QtWidgets.QFrame()
        container.setFixedWidth(220)
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_PANEL};
                border-radius: 6px;
            }}
        """)

        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(2)

        instance_name = badge.instance_data.get("name", "Unknown")
        mode_type = badge.instance_data.get("mode", "Unknown")

        name_label = QtWidgets.QLabel(f"{instance_name} • {mode_type.capitalize()}")
        name_label.setStyleSheet(f"color: {TEXT_LABEL}; font-size: 12px; font-weight: 500;")
        main_layout.addWidget(name_label)

        metrics_text = self._build_compact_metrics(instance_name)
        if metrics_text:
            metrics_label = QtWidgets.QLabel(metrics_text)
            metrics_label.setStyleSheet(WidgetStyles.label(size=11, color=TEXT_MUTED))
            main_layout.addWidget(metrics_label)

        return container

    def _format_metric_span(self, label: str, value: float, color: str, unit: str = "ms") -> str:
        """Format a metric as HTML span with label and colored value."""
        return f'<span style="color: {TEXT_MUTED};">{label}:</span> <span style="color: {color};">{value:.0f}{unit}</span>'

    def _get_or_create_card(
        self, cards_dict: dict, layout, key: str, label: str, unit: str, threshold_key: str
    ) -> StreamHealthCard:
        """Get existing card or create and register a new one."""
        if key not in cards_dict:
            low, high = METRIC_THRESHOLDS[threshold_key]
            card = StreamHealthCard(label, unit, low_threshold=low, high_threshold=high)
            cards_dict[key] = card
            layout.addWidget(card)
        return cards_dict[key]

    def _build_compact_metrics(self, instance_name: str) -> str:
        """Build single-line compact metrics (Internal, Inference, E2E, Accuracy)."""
        shared_state = getattr(self.parent_window, "shared_state", None)
        if not shared_state:
            return ""

        parts = []

        internal_ms = shared_state.get(mode_metric_key(instance_name, "internal_ms"))
        if internal_ms and internal_ms > 0:
            low, high = METRIC_THRESHOLDS["internal_ms"]
            color = self._get_level_color(self._get_resource_level(internal_ms, low, high))
            parts.append(self._format_metric_span("Int", internal_ms, color))

        inference_ms = shared_state.get(mode_metric_key(instance_name, "inference_ms"))
        if inference_ms and inference_ms > 0:
            low, high = METRIC_THRESHOLDS["inference_ms"]
            color = self._get_level_color(self._get_resource_level(inference_ms, low, high))
            parts.append(self._format_metric_span("Inf", inference_ms, color))

        e2e_ms = shared_state.get(e2e_latency_key())
        if e2e_ms and e2e_ms > 0:
            low, high = METRIC_THRESHOLDS["e2e_ms"]
            color = self._get_level_color(self._get_resource_level(e2e_ms, low, high))
            parts.append(self._format_metric_span("E2E", e2e_ms, color))

        gpu_mb = shared_state.get(mode_metric_key(instance_name, "gpu_mb"))
        if gpu_mb is not None and gpu_mb > 0:
            parts.append(self._format_metric_span("GPU", gpu_mb, TEXT_MAIN, "MB"))

        accuracy = shared_state.get(mode_metric_key(instance_name, "balanced_accuracy"))
        if accuracy is not None:
            if accuracy >= 0.7:
                color = STATUS_SUCCESS
            elif accuracy >= 0.5:
                color = STATUS_WARNING_ALT
            else:
                color = STATUS_DANGER
            parts.append(self._format_metric_span("Acc", accuracy * 100, color, "%"))

        return "  •  ".join(parts) if parts else ""

    def _get_resource_level(
        self, value: float, medium_threshold: float, high_threshold: float
    ) -> str:
        """Determine resource usage level."""
        if value >= high_threshold:
            return "high"
        elif value >= medium_threshold:
            return "medium"
        else:
            return "normal"

    def _get_level_color(self, level: str) -> str:
        """Get color for resource level."""
        if level == "high":
            return STATUS_DANGER
        elif level == "medium":
            return STATUS_WARNING_ALT
        else:
            return STATUS_SUCCESS

    def set_mode_pids(self, mode_pids: dict):
        """Update the mode PIDs dict for resource tracking."""
        self.mode_pids = mode_pids.copy()

    def set_system_pids(self, system_pids: dict):
        """Update the system PIDs dict for complete resource tracking."""
        self.system_pids = system_pids.copy()

    def on_recording_started(self):
        """Handle recording start event."""
        self.start_time = time.time()
        self.is_recording = True
        self.mode_pids = {}
        self.recording_bar.show()
        self.recording_bar.start()

    def on_recording_stopped(self):
        """Handle recording stop event."""
        self.start_time = None
        self.is_recording = False
        self.mode_pids = {}
        self.system_pids = {}
        self.recording_bar.stop()
        self.recording_bar.hide()
        for card in self.stream_info_cards.values():
            card.clear()
        for card in self.output_cards.values():
            card.clear()
        self.quality_label.hide()
        self.update_default_values()

    def cleanup(self):
        """Clean up resources (call before destroying widget)."""
        if self._resource_thread.isRunning():
            self._resource_thread.quit()
            self._resource_thread.wait(1000)
