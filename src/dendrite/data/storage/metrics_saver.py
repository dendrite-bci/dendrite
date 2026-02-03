"""Metrics saver for BMI mode outputs."""

import json
import logging
import queue
import signal
import sys
import time
from multiprocessing import Process
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

import h5py
import numpy as np

from dendrite.utils.logger_central import get_logger, setup_logger
from dendrite.utils.state_keys import e2e_latency_key, viz_consumers_key


class MetricsSaver(Process):
    """Metrics saver for saving metrics data from BMI modes to HDF5.

    This class handles metrics output from various BMI modes (Synchronous, Asynchronous, etc.)
    and saves them in a structured HDF5 format for analysis and visualization.
    """

    def __init__(
        self,
        filename: str,
        stop_event: Event,
        script_metadata: dict,
        mode_metrics_queues: dict[str, Queue] | None = None,
        shared_state=None,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.stop_event = stop_event
        self.script_metadata = script_metadata
        self.mode_metrics_queues = mode_metrics_queues or {}
        self.shared_state = shared_state

        # Telemetry sampling configuration
        self._telemetry_interval = 1.0  # Sample every 1 second
        self._last_telemetry_time = 0.0

        self.logger = get_logger("MetricsSaver")

    def run(self) -> None:
        """Main process entry point."""
        self.logger = setup_logger("MetricsSaver", level=logging.INFO)
        self.logger.info("MetricsSaver started")

        h5f = None

        def cleanup_handler(signum, frame):
            """Handle SIGTERM/SIGHUP by closing HDF5 file gracefully."""
            signal_name = signal.Signals(signum).name
            self.logger.warning(f"Received {signal_name}, closing metrics HDF5 file...")
            if h5f is not None:
                try:
                    h5f.close()
                    self.logger.info("Metrics HDF5 file closed successfully")
                except Exception as e:
                    self.logger.error(f"Error closing metrics HDF5 file: {e}")
            sys.exit(0)

        signal.signal(signal.SIGTERM, cleanup_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, cleanup_handler)

        try:
            with h5py.File(self.filename, "w") as h5f:
                self._save_script_metadata(h5f)
                self._create_mode_groups(h5f)
                self._create_telemetry_group(h5f)
                self._process_metrics_loop(h5f)

        except Exception as e:
            self.logger.error(f"Metrics saving error: {e}")
        finally:
            self.logger.info("Metrics data saver stopped")

    def _save_script_metadata(self, h5f: h5py.File) -> None:
        """Save script metadata to HDF5 file."""
        try:
            metadata_group = h5f.create_group("script_metadata")

            for key, value in self.script_metadata.items():
                if isinstance(value, (int, float, str, bool, np.integer, np.floating, np.bool_)):
                    metadata_group.attrs[key] = value
                elif isinstance(value, (list, dict)):
                    metadata_group.attrs[key] = json.dumps(value)
                else:
                    metadata_group.attrs[key] = str(value)

            self.logger.info("Script metadata saved")

        except Exception as e:
            self.logger.error(f"Error saving script metadata: {e}")

    def _create_mode_groups(self, h5f: h5py.File) -> dict:
        """Create groups for each BMI mode."""
        self.mode_groups = {}

        for mode_name in self.mode_metrics_queues.keys():
            group = h5f.create_group(mode_name)
            self.mode_groups[mode_name] = group
            group.attrs["mode_type"] = (
                "synchronous" if "sync" in mode_name.lower() else "asynchronous"
            )
            group.attrs["created_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info(f"Created {len(self.mode_groups)} mode groups")

    def _create_telemetry_group(self, h5f: h5py.File) -> None:
        """Create telemetry group for SharedState metrics if shared_state is provided."""
        if self.shared_state is None:
            self.telemetry_group = None
            return

        self.telemetry_group = h5f.create_group("telemetry")
        self.telemetry_group.attrs["created_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.telemetry_group.attrs["sample_interval_s"] = self._telemetry_interval
        self.logger.info("Created telemetry group for SharedState metrics")

    def _process_metrics_loop(self, h5f: h5py.File) -> None:
        """Main metrics processing loop."""
        while not self.stop_event.is_set():
            self._process_all_queues()
            self._sample_telemetry()
            time.sleep(0.1)

        self._process_all_queues(drain=True)

    def _sample_telemetry(self) -> None:
        """Sample SharedState metrics periodically."""
        if self.telemetry_group is None or self.shared_state is None:
            return

        current_time = time.time()
        if current_time - self._last_telemetry_time < self._telemetry_interval:
            return
        self._last_telemetry_time = current_time

        timestamp = current_time

        # Discover and sample stream latencies dynamically
        for key in self.shared_state.keys():
            if key.endswith("_latency_p50"):
                stream = key.replace("_latency_p50", "")
                latency = self.shared_state.get(key)
                if latency is not None:
                    self._save_individual_metric(
                        self.telemetry_group, f"{stream}_latency_ms", float(latency), timestamp
                    )

        # Sample E2E latency
        e2e = self.shared_state.get(e2e_latency_key())
        if e2e is not None:
            self._save_individual_metric(
                self.telemetry_group, "e2e_latency_ms", float(e2e), timestamp
            )

        # Sample visualization stream consumer count
        consumers = self.shared_state.get(viz_consumers_key())
        if consumers is not None:
            self._save_individual_metric(
                self.telemetry_group, "viz_stream_consumers", int(consumers), timestamp
            )

        # Sample mode metrics and streamer bandwidth (single pass)
        for key in self.shared_state.keys():
            if key.endswith(
                ("_internal_ms", "_inference_ms", "_gpu_mb", "_bandwidth_kbps", "_accuracy")
            ):
                value = self.shared_state.get(key)
                if value is not None:
                    self._save_individual_metric(self.telemetry_group, key, float(value), timestamp)

    def _process_all_queues(self, drain: bool = False) -> None:
        """Process all mode metrics queues."""
        for mode_name, queue_obj in self.mode_metrics_queues.items():
            group = self.mode_groups[mode_name]
            while True:
                try:
                    data = queue_obj.get_nowait()
                    self._save_metrics_data(group, data, mode_name)
                except queue.Empty:
                    break

    def _save_metrics_data(
        self, group: h5py.Group, data_packet: dict[str, Any], mode_name: str
    ) -> None:
        """Save metrics data to HDF5 group."""
        try:
            timestamp = self._extract_timestamp(data_packet)
            metrics_data = self._extract_metrics_data(data_packet)

            for key, value in metrics_data.items():
                self._save_individual_metric(group, key, value, timestamp)

        except Exception as e:
            self.logger.error(f"Error saving metrics data from {mode_name}: {e}")

    def _extract_timestamp(self, data_packet: dict[str, Any]) -> float:
        """Extract timestamp from data packet as epoch float (matches raw data format)."""
        timestamp_val = data_packet.get("data_timestamp")
        if isinstance(timestamp_val, (int, float)):
            return float(timestamp_val)
        return time.time()

    def _extract_metrics_data(self, data_packet: dict[str, Any]) -> dict[str, Any]:
        """Extract and flatten metrics data from mode output."""
        metrics_data = {}

        if data_packet.get("type"):
            metrics_data["packet_output_type"] = data_packet["type"]

        if data_packet.get("mode_name"):
            metrics_data["source_mode"] = data_packet["mode_name"]

        payload = data_packet.get("data", {})

        if hasattr(payload, "__dict__"):
            self._flatten_dict(payload.__dict__, metrics_data)
        elif isinstance(payload, dict):
            self._flatten_dict(payload, metrics_data)
        else:
            metrics_data.update(
                {
                    key: value
                    for key, value in data_packet.items()
                    if key not in ["timestamp", "data"]
                }
            )

        return metrics_data

    def _flatten_dict(self, data: dict, target: dict, prefix: str = "") -> None:
        """Recursively flatten nested dictionaries."""
        for key, value in data.items():
            full_key = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_dict(value, target, full_key)
            elif value is None:
                target[full_key] = float("nan")  # Use NaN for consistent numeric typing
            else:
                target[full_key] = value

    def _save_individual_metric(
        self, group: h5py.Group, key: str, value: Any, timestamp: float
    ) -> None:
        """Save individual metric to HDF5 group."""
        try:
            if not key or not isinstance(key, str):
                return

            safe_key = self._sanitize_key(key)
            dtype, formatted_value, value_shape = self._format_value_for_hdf5(value)

            if safe_key not in group:
                self._create_metric_dataset(
                    group, safe_key, formatted_value, dtype, value_shape, timestamp
                )
            else:
                self._append_to_metric_dataset(group, safe_key, formatted_value, timestamp)

        except Exception as e:
            self.logger.error(f"Error saving metric {key}: {e}")

    def _sanitize_key(self, key: str) -> str:
        """Sanitize key name for HDF5 compatibility."""
        safe_key = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        if safe_key and safe_key[0].isdigit():
            safe_key = f"_{safe_key}"
        return safe_key

    def _format_value_for_hdf5(self, value: Any) -> tuple[np.dtype, Any, tuple]:
        """Format value for HDF5 storage with shape information."""
        if isinstance(value, str):
            return h5py.string_dtype(encoding="utf-8"), value, ()
        elif isinstance(value, bool):
            return np.bool_, bool(value), ()
        elif isinstance(value, (int, np.integer)):
            return np.int64, int(value), ()
        elif isinstance(value, (float, np.floating)):
            return np.float64, float(value), ()
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return np.float64, 0.0, ()
            return np.float64, value.astype(np.float64), value.shape
        elif isinstance(value, (list, tuple)):
            try:
                arr = np.array(value, dtype=np.float64)
                return np.float64, arr, arr.shape
            except (ValueError, TypeError):
                return h5py.string_dtype(encoding="utf-8"), str(value), ()
        else:
            return h5py.string_dtype(encoding="utf-8"), str(value), ()

    def _create_metric_dataset(
        self,
        group: h5py.Group,
        key: str,
        value: Any,
        dtype: np.dtype,
        value_shape: tuple,
        timestamp: float,
    ) -> None:
        """Create a new metric dataset."""
        if value_shape:
            initial_shape = (1,) + value_shape
            maxshape = (None,) + value_shape
            chunk_shape = (1,) + value_shape

            dataset = group.create_dataset(
                key,
                shape=initial_shape,
                dtype=dtype,
                maxshape=maxshape,
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=1,
            )
            dataset[0] = value
        else:
            dataset = group.create_dataset(
                key,
                data=[value],
                dtype=dtype,
                maxshape=(None,),
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

        dataset.attrs["first_timestamp"] = timestamp
        dataset.attrs["value_type"] = str(type(value).__name__)
        if value_shape:
            dataset.attrs["original_shape"] = value_shape

        group.create_dataset(
            f"{key}_timestamps",
            data=[timestamp],
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression="gzip",
            compression_opts=1,
        )

    def _append_to_metric_dataset(
        self, group: h5py.Group, key: str, value: Any, timestamp: float
    ) -> None:
        """Append value to existing metric dataset."""
        dataset = group[key]
        timestamp_dataset = group[f"{key}_timestamps"]

        current_size = dataset.shape[0]
        new_size = current_size + 1

        dataset.resize((new_size,) + dataset.shape[1:])
        timestamp_dataset.resize((new_size,))

        try:
            dataset[current_size] = value
            timestamp_dataset[current_size] = timestamp
        except ValueError as e:
            self.logger.error(f"Shape mismatch when appending to {key}: {e}")
