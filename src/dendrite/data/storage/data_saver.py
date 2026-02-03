"""Data saver for raw data streams to HDF5."""

import json
import logging
import os
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

from dendrite.constants import APP_NAME, LOG_INTERVAL_SAMPLES, VERSION
from dendrite.data.acquisition import DataRecord
from dendrite.utils.logger_central import setup_logger


class DataSaver(Process):
    """Data saver for saving data streams to HDF5 format.

    Uses consistent error handling patterns similar to DataAcquisition.
    Includes chunk saving for improved performance.
    """

    EVENT_DTYPE = np.dtype(
        [
            ("event_id", np.int32),
            ("event_type", h5py.string_dtype(encoding="utf-8")),
            ("timestamp", np.float64),
            ("local_timestamp", np.float64),
            ("extra_vars", h5py.string_dtype(encoding="utf-8")),
        ]
    )

    DTYPE_MAP = {
        "float32": np.float32,
        "double64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "string": h5py.string_dtype(encoding="utf-8"),
    }

    # Keys to skip when storing metadata (channel_count redundant, version conflicts with app version)
    METADATA_SKIP_KEYS = {"channel_count", "version"}

    def __init__(
        self, filename: str, save_queue: Queue, stop_event: Event, chunk_size: int = 100
    ) -> None:
        super().__init__()
        self.filename = os.path.normpath(filename)
        self.save_queue = save_queue
        self.stop_event = stop_event
        self.chunk_size = chunk_size

        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)

        self.stream_metadata = {}
        self.datasets = {}

        self.data_buffers = {}
        self.event_buffer = []

        self.flush_interval = 5.0
        self.last_flush_time = time.time()
        self.event_chunk_size = max(10, chunk_size // 10)

    def run(self) -> None:
        """Main process entry point."""
        self.logger = setup_logger("DataSaver", level=logging.INFO)
        self.logger.info("Data saving process started")

        h5f = None

        def cleanup_handler(signum, frame) -> None:
            """Handle SIGTERM/SIGHUP by closing HDF5 file gracefully."""
            signal_name = signal.Signals(signum).name
            self.logger.warning(f"Received {signal_name}, closing HDF5 file...")
            if h5f is not None:
                try:
                    self._flush_all_buffers(h5f)
                    h5f.close()
                    self.logger.info("HDF5 file closed successfully")
                except Exception as e:
                    self.logger.error(f"Error closing HDF5 file: {e}")
            self.logger.info("Data saver process stopped")
            sys.exit(0)

        signal.signal(signal.SIGTERM, cleanup_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, cleanup_handler)

        try:
            with h5py.File(self.filename, "w") as h5f:
                self._initialize_file(h5f)
                self._process_data_loop(h5f)
                self._flush_all_buffers(h5f)

        except Exception as e:
            self.logger.error(f"Data saving error: {e}")
        finally:
            self.logger.info("Data saver process stopped")

    def _initialize_file(self, h5f: h5py.File) -> None:
        h5f.attrs["created_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        h5f.attrs["created_by"] = f"{APP_NAME} DataSaver v{VERSION}"
        h5f.attrs["version"] = VERSION
        self.logger.info("HDF5 file initialized")

    def _process_data_loop(self, h5f: h5py.File) -> None:
        record_count = 0

        while not self.stop_event.is_set() or not self.save_queue.empty():
            try:
                record = self.save_queue.get(timeout=0.1)
                record_count += 1

                self._process_record(h5f, record)

                if record_count % LOG_INTERVAL_SAMPLES == 0:
                    self.logger.info(f"Processed {record_count} records")

                self._periodic_flush(h5f)

            except queue.Empty:
                self._periodic_flush(h5f)
                continue
            except Exception as e:
                self.logger.error(f"Error processing record: {e}")
                continue

        self.logger.info(f"Data processing completed. Total records: {record_count}")

    def _process_record(self, h5f: h5py.File, record: DataRecord) -> None:
        """Process a single data record based on its modality."""
        modality = record.modality

        try:
            if modality == "Metadata":
                self._handle_global_metadata(h5f, record)
            elif modality.endswith("_Metadata"):
                self._handle_stream_metadata(h5f, record)
            elif modality == "Event":
                self._handle_event_data(h5f, record)
            else:
                self._handle_timeseries_data(h5f, record)

        except Exception as e:
            self.logger.error(f"Error processing {modality} record: {e}")

    def _periodic_flush(self, h5f: h5py.File) -> None:
        current_time = time.time()
        if current_time - self.last_flush_time > self.flush_interval:
            h5f.flush()
            self.last_flush_time = current_time

    def _handle_global_metadata(self, h5f: h5py.File, record: DataRecord) -> None:
        """Handle global metadata record."""
        try:
            metadata = json.loads(record.sample)

            for key, value in metadata.items():
                self._safe_set_attribute(h5f.attrs, key, value)

            self.logger.info("Global metadata processed")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid global metadata JSON: {e}")
        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error processing global metadata: {e}")

    def _handle_stream_metadata(self, h5f: h5py.File, record: DataRecord) -> None:
        """Handle stream-specific metadata."""
        try:
            stream_type = record.modality.split("_")[0]
            metadata_dict = json.loads(record.sample)

            if "labels" in metadata_dict:
                metadata_dict["labels"] = [str(label) for label in metadata_dict["labels"]]

            self.stream_metadata[stream_type] = metadata_dict
            self.logger.info(f"Stream metadata received for {stream_type}")

            if stream_type not in self.datasets and stream_type != "Events":
                dataset = self._create_timeseries_dataset(h5f, stream_type)
                if dataset is not None:
                    self.datasets[stream_type] = dataset

            if stream_type == "Events" and "Event" not in self.datasets:
                self.datasets["Event"] = self._create_event_dataset(h5f)

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid stream metadata JSON: {e}")
        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error processing stream metadata: {e}")

    def _handle_timeseries_data(self, h5f: h5py.File, record: DataRecord) -> None:
        """Handle timeseries data (EEG, EMG, etc.) with chunk buffering."""
        modality = record.modality

        if modality not in self.stream_metadata:
            self.logger.warning(f"No metadata for {modality}, buffering record...")
            try:
                self.save_queue.put_nowait(record)
            except queue.Full:
                pass
            return

        if modality not in self.datasets:
            dataset = self._create_timeseries_dataset(h5f, modality)
            if dataset is None:
                self.logger.error(f"Failed to create dataset for {modality}")
                return
            self.datasets[modality] = dataset

        if modality not in self.data_buffers:
            self.data_buffers[modality] = []

        self.data_buffers[modality].append(record)

        if len(self.data_buffers[modality]) >= self.chunk_size:
            self._write_timeseries_chunk(h5f, modality)

    def _handle_event_data(self, h5f: h5py.File, record: DataRecord) -> None:
        """Handle event data with chunk buffering."""
        if "Event" not in self.datasets:
            self.datasets["Event"] = self._create_event_dataset(h5f)

        self.event_buffer.append(record)

        if len(self.event_buffer) >= self.event_chunk_size:
            self._write_event_chunk(h5f)

    def _create_timeseries_dataset(self, h5f: h5py.File, modality: str) -> h5py.Dataset | None:
        """Create a timeseries dataset for the given modality."""
        try:
            if modality in h5f:
                self.logger.info(f"Dataset {modality} already exists, reusing")
                return h5f[modality]

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

            dataset.attrs["field_names"] = list(structured_dtype.names)

            for key, value in metadata.items():
                if key in self.METADATA_SKIP_KEYS:
                    continue
                # Translate LSL names to storage names
                if key == "labels":
                    key = "channel_labels"
                elif key == "sample_rate":
                    key = "sampling_frequency"
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

            dataset.attrs["channel_labels"] = list(self.EVENT_DTYPE.names)
            dataset.attrs["description"] = "Event markers with timestamps and metadata"

            self.logger.info("Created Event dataset")
            return dataset

        except (ValueError, TypeError, OSError) as e:
            self.logger.error(f"Error creating event dataset: {e}")
            raise

    def _write_timeseries_chunk(self, h5f: h5py.File, modality: str) -> None:
        """Write a chunk of timeseries data to dataset."""
        if modality not in self.data_buffers or not self.data_buffers[modality]:
            return

        try:
            records = self.data_buffers[modality]
            dataset = self.datasets[modality]
            structured_dtype = dataset.dtype

            field_names = [
                name
                for name in structured_dtype.names
                if name not in ["timestamp", "local_timestamp"]
            ]
            expected_channels = len(field_names)

            chunk_size = len(records)
            chunk_array = np.zeros(chunk_size, dtype=structured_dtype)

            for i, record in enumerate(records):
                if isinstance(record.sample, (list, np.ndarray)):
                    sample = np.asarray(record.sample)
                else:
                    sample = np.full(expected_channels, record.sample)

                if len(sample) != expected_channels:
                    sample = self._adjust_sample_size(sample, expected_channels, record.modality)

                for j, field_name in enumerate(field_names):
                    if j < len(sample):
                        chunk_array[i][field_name] = sample[j]

                chunk_array[i]["timestamp"] = record.timestamp
                chunk_array[i]["local_timestamp"] = record.local_timestamp

            current_size = dataset.shape[0]
            new_size = current_size + chunk_size
            dataset.resize(new_size, axis=0)
            dataset[current_size:new_size] = chunk_array

            h5f.flush()
            self.data_buffers[modality].clear()

        except (ValueError, TypeError, KeyError, OSError) as e:
            self.logger.error(f"Error writing timeseries chunk for {modality}: {e}")

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

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing event chunk: {e}")
        except (KeyError, OSError) as e:
            self.logger.error(f"Error writing event chunk: {e}")

    def _flush_all_buffers(self, h5f: h5py.File) -> None:
        try:
            for modality in list(self.data_buffers.keys()):
                if self.data_buffers[modality]:
                    self._write_timeseries_chunk(h5f, modality)

            if self.event_buffer:
                self._write_event_chunk(h5f)

            self.logger.info("All buffers flushed successfully")

        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")

    def _get_dtype_from_metadata(self, metadata: dict) -> np.dtype:
        channel_format = metadata.get("channel_format", "float64")
        return self.DTYPE_MAP.get(channel_format, np.float64)

    def _build_structured_dtype(self, metadata: dict) -> np.dtype:
        """Build a structured dtype for timeseries data."""
        channel_labels = metadata.get("labels", [])
        channel_count = metadata.get("channel_count", len(channel_labels))
        base_dtype = self._get_dtype_from_metadata(metadata)

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
            dtype_list.append((field_name, base_dtype))

        dtype_list.append(("timestamp", np.float64))
        dtype_list.append(("local_timestamp", np.float64))

        return np.dtype(dtype_list)

    def _adjust_sample_size(
        self, sample: np.ndarray, expected_channels: int, modality: str
    ) -> np.ndarray:
        """Adjust sample size to match expected channel count."""
        if len(sample) < expected_channels:
            padding = np.zeros(expected_channels - len(sample), dtype=sample.dtype)
            adjusted_sample = np.concatenate([sample, padding])
            self.logger.warning(
                f"Padded {modality} sample from {len(sample)} to {expected_channels} channels"
            )
        else:
            adjusted_sample = sample[:expected_channels]
            self.logger.warning(
                f"Truncated {modality} sample from {len(sample)} to {expected_channels} channels"
            )

        return adjusted_sample

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
