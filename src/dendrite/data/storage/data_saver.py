"""Data saver for raw data streams to HDF5.

Reads timeseries from ring buffers and events from a small event queue.
"""

import json
import logging
import os
import queue
import time
from multiprocessing import Process
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

import h5py
import numpy as np

from dendrite import __version__
from dendrite.data.acquisition import EventRecord
from dendrite.data.shared_buffers import OverrunError, SharedRingBuffer
from dendrite.utils.component_state import ComponentState, ComponentStateMachine
from dendrite.utils.logger_central import setup_logger


class DataSaver(Process):
    """Data saver that reads from ring buffers and writes to HDF5."""

    EVENT_DTYPE = np.dtype(
        [
            ("event_id", np.int32),
            ("event_type", h5py.string_dtype(encoding="utf-8")),
            ("timestamp", np.float64),
            ("local_timestamp", np.float64),
            ("receive_timestamp", np.float64),
            ("extra_vars", h5py.string_dtype(encoding="utf-8")),
        ]
    )


    def __init__(
        self,
        filename: str,
        stop_event: Event,
        shared_state=None,
        ring_buffer_names: dict[str, str] | None = None,
        ring_buffer_channel_maps: dict[str, dict] | None = None,
        event_queue: Queue | None = None,
        global_metadata: dict | None = None,
        stream_configs: list | None = None,
        chunk_size: int = 100,
    ) -> None:
        super().__init__()
        self.filename = os.path.normpath(filename)
        self.stop_event = stop_event
        self.shared_state = shared_state
        self._ring_buffer_names = ring_buffer_names or {}
        self._ring_buffer_channel_maps = ring_buffer_channel_maps or {}
        self.event_queue = event_queue
        self.global_metadata = global_metadata
        self._stream_configs = {(c.stream_key or c.type): c for c in (stream_configs or [])}
        self.chunk_size = chunk_size

        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)

        self.stream_metadata: dict[str, dict] = {}
        self.datasets: dict[str, h5py.Dataset] = {}

        self.data_buffers: dict[str, dict[str, list]] = {}
        self.event_buffer: list[EventRecord] = []

        self.flush_interval = 2.0
        self.last_flush_time = time.time()
        self.event_chunk_size = max(10, chunk_size // 10)
        self._swmr_enabled = False

    def run(self) -> None:
        """Main process entry point."""
        self.logger = setup_logger("DataSaver", level=logging.INFO)
        self._state_machine = ComponentStateMachine("data_saver", self.shared_state)
        self.logger.info("Data saving process started")
        self._state_machine.transition(ComponentState.STARTING)

        h5f: h5py.File | None = None

        # Connect to ring buffers (must happen in child process)
        ring_buffers: dict[str, SharedRingBuffer] = {}
        read_positions: dict[str, int] = {}
        for stream_type, buf_name in self._ring_buffer_names.items():
            try:
                rb = SharedRingBuffer.connect(buf_name)
                ring_buffers[stream_type] = rb
                read_positions[stream_type] = rb.write_pos  # start from current
                self.logger.info(f"Connected to ring buffer: {buf_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to ring buffer '{buf_name}': {e}")

        try:
            h5f = h5py.File(self.filename, "w", libver="latest")
            self._initialize_file(h5f)
            self._create_all_datasets(h5f)
            if self.shared_state:
                self.shared_state.set("recording_file", self.filename)
            self._state_machine.transition(ComponentState.RUNNING)
            self._process_data_loop(h5f, ring_buffers, read_positions)
            self._flush_all_buffers(h5f)

        except Exception as e:
            self.logger.error(f"Data saving error: {e}")
            if self._state_machine.state not in (ComponentState.STOPPING, ComponentState.STOPPED):
                self._state_machine.set_error(str(e))
        finally:
            if h5f is not None:
                try:
                    h5f.close()
                    self.logger.info("HDF5 file closed")
                except Exception as e:
                    self.logger.error(f"Error closing HDF5 file: {e}")
            for rb in ring_buffers.values():
                rb.close()
            self._state_machine.finalize()
            self.logger.info("Data saver process stopped")

    SCHEMA_VERSION = 1

    def _initialize_file(self, h5f: h5py.File) -> None:
        h5f.attrs["created_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        h5f.attrs["created_by"] = f"Dendrite DataSaver v{__version__}"
        h5f.attrs["version"] = __version__
        h5f.attrs["schema_version"] = self.SCHEMA_VERSION
        if self.global_metadata:
            for key, value in self.global_metadata.items():
                self._safe_set_attribute(h5f.attrs, key, value)
        self.logger.info("HDF5 file initialized")

    def _create_all_datasets(self, h5f: h5py.File) -> None:
        """Create all timeseries and event datasets upfront from stream configs."""
        for stream_type in self._ring_buffer_names:
            config = self._stream_configs.get(stream_type)
            channel_map = self._ring_buffer_channel_maps.get(stream_type, {})

            if config:
                labels = list(config.labels or [])
                channel_types = list(config.channel_types or [])
                channel_count = config.channel_count
                sample_rate = config.sample_rate or 500.0
            else:
                labels = []
                channel_types = []
                channel_count = 0
                sample_rate = channel_map.get("sample_rate", 500.0)

            metadata = {
                "labels": labels,
                "channel_count": channel_count,
                "channel_format": "float32",
                "channel_types": channel_types,
                "sample_rate": sample_rate,
            }
            self.stream_metadata[stream_type] = metadata
            dataset = self._create_timeseries_dataset(h5f, stream_type)
            if dataset is not None:
                self.datasets[stream_type] = dataset

        # Always create Event dataset
        self.datasets["Event"] = self._create_event_dataset(h5f)

        # Write stream index: maps stream_key → stream_type for robust loading
        stream_index = {}
        for key in self._ring_buffer_names:
            cfg = self._stream_configs.get(key)
            stream_index[key] = cfg.type if cfg else key
        h5f.attrs["stream_index"] = json.dumps(stream_index)

        self.logger.info(f"Created {len(self.datasets)} datasets upfront")

    def _process_data_loop(
        self,
        h5f: h5py.File,
        ring_buffers: dict[str, SharedRingBuffer],
        read_positions: dict[str, int],
    ) -> None:
        sample_count = 0

        while not self.stop_event.is_set():
            drained = self._drain_ring_buffers(ring_buffers, read_positions)
            self._drain_event_queue()
            sample_count += drained

            # Write chunks if buffer large enough
            for stream_type in list(self.data_buffers.keys()):
                buf = self.data_buffers[stream_type]
                total_samples = sum(len(d) for d in buf["data"])
                if total_samples >= self.chunk_size:
                    self._write_timeseries_chunk(h5f, stream_type)

            if len(self.event_buffer) >= self.event_chunk_size:
                self._write_event_chunk(h5f)

            self._periodic_flush(h5f)

            if drained == 0:
                time.sleep(0.01)

        self.logger.info(f"Data processing completed. Total samples: {sample_count}")

    def _drain_ring_buffers(
        self,
        ring_buffers: dict[str, SharedRingBuffer],
        read_positions: dict[str, int],
    ) -> int:
        total = 0
        for stream_type, rb in ring_buffers.items():
            try:
                data, timestamps, local_ts, receive_ns, new_pos = rb.read_new(read_positions[stream_type])
            except OverrunError:
                read_positions[stream_type] = rb.write_pos
                self.logger.warning(f"Ring buffer overrun in {stream_type}, skipping ahead")
                continue
            except (FileNotFoundError, ValueError, OSError):
                continue

            if len(data) == 0:
                continue
            read_positions[stream_type] = new_pos

            # Save all raw channels (exclude the injected markers column at the end)
            raw_channels = rb.n_channels - 1
            mod_data = data[:, :raw_channels].astype(np.float32)

            receive_timestamps = receive_ns.astype(np.float64) / 1e9

            if stream_type not in self.data_buffers:
                self.data_buffers[stream_type] = {
                    "data": [], "timestamps": [], "local_timestamps": [], "receive_timestamps": [],
                }
            self.data_buffers[stream_type]["data"].append(mod_data)
            self.data_buffers[stream_type]["timestamps"].append(timestamps)
            self.data_buffers[stream_type]["local_timestamps"].append(local_ts)
            self.data_buffers[stream_type]["receive_timestamps"].append(receive_timestamps)

            total += len(data)
        return total

    def _drain_event_queue(self) -> None:
        if self.event_queue is None:
            return
        while True:
            try:
                record = self.event_queue.get_nowait()
                self.event_buffer.append(record)
            except queue.Empty:
                break

    def _periodic_flush(self, h5f: h5py.File) -> None:
        current_time = time.time()
        if current_time - self.last_flush_time > self.flush_interval:
            h5f.flush()
            self._enable_swmr_if_ready(h5f)
            self.last_flush_time = current_time

    def _enable_swmr_if_ready(self, h5f: h5py.File) -> None:
        """Enable SWMR mode once datasets exist."""
        if not self._swmr_enabled and self.datasets:
            h5f.swmr_mode = True
            self._swmr_enabled = True
            self.logger.info("SWMR mode enabled")

    def _create_timeseries_dataset(self, h5f: h5py.File, modality: str) -> h5py.Dataset | None:
        """Create a timeseries dataset for the given modality."""
        try:
            if modality in h5f:
                existing = h5f[modality]
                if isinstance(existing, h5py.Dataset):
                    self.logger.info(f"Dataset {modality} already exists, reusing")
                    return existing
                return None

            metadata = self.stream_metadata.get(modality)
            if not metadata:
                self.logger.error(f"No metadata available for {modality}")
                return None

            channel_labels = metadata.get("labels", [])
            channel_count = metadata.get("channel_count", len(channel_labels))

            if channel_count == 0:
                self.logger.error(f"No channels defined for {modality}")
                return None

            structured_dtype = self._build_structured_dtype(metadata)

            dataset = h5f.create_dataset(
                modality,
                shape=(0,),
                maxshape=(None,),
                dtype=structured_dtype,
                chunks=(self.chunk_size,),
            )

            dataset.attrs["field_names"] = list(structured_dtype.names or ())
            dataset.attrs["channel_labels"] = channel_labels
            dataset.attrs["sampling_frequency"] = metadata.get("sample_rate", 500.0)
            dataset.attrs["channel_format"] = metadata.get("channel_format", "float32")
            if "channel_types" in metadata:
                dataset.attrs["channel_types"] = metadata["channel_types"]

            # Write full stream config metadata
            stream_cfg = self._stream_configs.get(modality)
            if stream_cfg:
                skip = {"channel_count", "version", "labels", "sample_rate"}
                for key, value in stream_cfg.model_dump().items():
                    if key in skip or key in dataset.attrs:
                        continue
                    self._safe_set_attribute(dataset.attrs, key, value)

            self.logger.info(f"Created {modality} dataset with {channel_count} channels")
            return dataset

        except (ValueError, TypeError, OSError) as e:
            self.logger.error(f"Error creating dataset for {modality}: {e}")
            return None

    def _create_event_dataset(self, h5f: h5py.File) -> h5py.Dataset:
        """Create event dataset."""
        try:
            dataset = h5f.create_dataset(
                "Event",
                shape=(0,),
                maxshape=(None,),
                dtype=self.EVENT_DTYPE,
                chunks=(self.event_chunk_size,),
            )

            dataset.attrs["channel_labels"] = list(self.EVENT_DTYPE.names or ())
            dataset.attrs["description"] = "Event markers with timestamps and metadata"

            self.logger.info("Created Event dataset")
            return dataset

        except (ValueError, TypeError, OSError) as e:
            self.logger.error(f"Error creating event dataset: {e}")
            raise

    def _write_timeseries_chunk(self, h5f: h5py.File, stream_type: str) -> None:
        """Write a chunk of timeseries data to dataset."""
        buf = self.data_buffers.get(stream_type)
        if not buf or not buf["data"]:
            return

        try:
            data = np.concatenate(buf["data"])  # (n_samples, n_channels)
            timestamps = np.concatenate(buf["timestamps"])  # LSL clock-synced
            local_timestamps = np.concatenate(buf["local_timestamps"])  # LSL local_clock()
            receive_timestamps = np.concatenate(buf["receive_timestamps"])  # time.time_ns()

            dataset = self.datasets[stream_type]
            structured_dtype = dataset.dtype
            ts_fields = {"timestamp", "local_timestamp", "receive_timestamp"}
            field_names = [n for n in (structured_dtype.names or ()) if n not in ts_fields]

            n_samples = len(data)
            chunk_array: np.ndarray = np.zeros(n_samples, dtype=structured_dtype)

            for j, field_name in enumerate(field_names):
                chunk_array[field_name] = data[:, j]
            chunk_array["timestamp"] = timestamps
            chunk_array["local_timestamp"] = local_timestamps
            chunk_array["receive_timestamp"] = receive_timestamps

            current_size = dataset.shape[0]
            new_size = current_size + n_samples
            dataset.resize(new_size, axis=0)
            dataset[current_size:new_size] = chunk_array

            h5f.flush()
            for v in buf.values():
                v.clear()

        except (ValueError, TypeError, KeyError, OSError) as e:
            self.logger.error(f"Error writing timeseries chunk for {stream_type}: {e}")

    def _write_event_chunk(self, h5f: h5py.File) -> None:
        """Write a chunk of event data to dataset."""
        if not self.event_buffer:
            return

        try:
            dataset = self.datasets["Event"]

            chunk_data = []
            for record in self.event_buffer:
                # Event dict validated by EventData schema in DAQ
                normalized = record.sample

                event_id = int(normalized["event_id"])
                event_type = normalized["event_type"]

                extra_vars = {
                    k: v for k, v in normalized.items() if k not in ["event_id", "event_type"]
                }
                extra_vars_json = json.dumps(extra_vars)

                event_record = (
                    event_id,
                    event_type,
                    float(record.timestamp),
                    float(record.local_timestamp),
                    float(record.receive_timestamp),
                    extra_vars_json,
                )
                chunk_data.append(event_record)

            chunk_array = np.array(chunk_data, dtype=self.EVENT_DTYPE)
            current_size = dataset.shape[0]
            new_size = current_size + len(chunk_data)
            dataset.resize(new_size, axis=0)
            dataset[current_size:new_size] = chunk_array

            h5f.flush()
            self.event_buffer.clear()

        except (ValueError, TypeError, KeyError, OSError) as e:
            self.logger.error(f"Error writing event chunk: {e}")

    def _flush_all_buffers(self, h5f: h5py.File) -> None:
        try:
            for stream_type in list(self.data_buffers.keys()):
                if self.data_buffers[stream_type]["data"]:
                    self._write_timeseries_chunk(h5f, stream_type)

            if self.event_buffer:
                self._write_event_chunk(h5f)

            self.logger.info("All buffers flushed successfully")

        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")

    def _build_structured_dtype(self, metadata: dict) -> np.dtype:
        """Build a structured dtype for timeseries data."""
        channel_labels = metadata.get("labels", [])
        channel_count = metadata.get("channel_count", len(channel_labels))

        dtype_list = []

        for i in range(channel_count):
            if channel_labels and i < len(channel_labels):
                field_name = str(channel_labels[i])
                field_name = field_name.replace(" ", "_").replace("-", "_").replace(".", "_")
                field_name = field_name.replace("/", "_").replace("\\", "_")
                if field_name in [name for name, _ in dtype_list]:
                    field_name = f"{field_name}_{i}"
            else:
                field_name = f"ch{i + 1}"
            dtype_list.append((field_name, np.float32))

        dtype_list.append(("timestamp", np.float64))
        dtype_list.append(("local_timestamp", np.float64))
        dtype_list.append(("receive_timestamp", np.float64))

        return np.dtype(dtype_list)

    def _safe_set_attribute(self, attrs: h5py.AttributeManager, key: str, value: Any) -> None:
        try:
            if isinstance(value, (list, dict, tuple, set)):
                attrs[key] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool, np.integer, np.floating, np.bool_)):
                attrs[key] = value
            else:
                attrs[key] = str(value)
        except Exception as e:
            self.logger.warning(f"Could not store attribute {key}: {e}")
