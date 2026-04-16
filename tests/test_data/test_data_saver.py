"""Tests for DataSaver HDF5 persistence."""

import json
import logging
import multiprocessing

import h5py
import numpy as np
import pytest

from dendrite.data.acquisition import EventRecord
from dendrite.data.storage.data_saver import DataSaver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saver(tmp_path):
    """DataSaver instance with logger injected — NOT started as a process."""
    filename = str(tmp_path / "test.h5")
    evt = multiprocessing.Event()
    ds = DataSaver(
        filename=filename,
        stop_event=evt,
        ring_buffer_names={},
        ring_buffer_channel_maps={},
        chunk_size=10,
    )
    ds.logger = logging.getLogger("test.DataSaver")
    yield ds


@pytest.fixture
def h5f(tmp_path):
    """Open HDF5 file for direct method testing."""
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as f:
        yield f


@pytest.fixture
def eeg_metadata():
    """Standard 3-channel EEG stream metadata."""
    return {
        "labels": ["Fp1", "Fp2", "Cz"],
        "channel_count": 3,
        "channel_format": "float32",
        "sample_rate": 256.0,
    }


# ---------------------------------------------------------------------------
# Structured dtype building
# ---------------------------------------------------------------------------


class TestBuildStructuredDtype:
    def test_with_labels(self, saver, eeg_metadata):
        dt = saver._build_structured_dtype(eeg_metadata)
        names = list(dt.names)
        assert names == ["Fp1", "Fp2", "Cz", "timestamp", "local_timestamp", "receive_timestamp"]
        assert dt["Fp1"] == np.float32
        assert dt["timestamp"] == np.float64

    def test_duplicate_labels_get_suffix(self, saver):
        meta = {"labels": ["ch", "ch", "other"], "channel_count": 3, "channel_format": "float32"}
        dt = saver._build_structured_dtype(meta)
        names = list(dt.names)
        # First "ch" keeps name, second gets "_1" suffix
        assert names[0] == "ch"
        assert names[1] == "ch_1"
        assert names[2] == "other"

    def test_special_chars_replaced(self, saver):
        meta = {"labels": ["F p1", "F-p2", "C.z"], "channel_count": 3, "channel_format": "float32"}
        dt = saver._build_structured_dtype(meta)
        names = list(dt.names)
        assert names[0] == "F_p1"
        assert names[1] == "F_p2"
        assert names[2] == "C_z"

    def test_fallback_channel_names(self, saver):
        meta = {"labels": [], "channel_count": 2, "channel_format": "float32"}
        dt = saver._build_structured_dtype(meta)
        names = list(dt.names)
        assert names[0] == "ch1"
        assert names[1] == "ch2"



# ---------------------------------------------------------------------------
# Timeseries chunk writing (array-based buffers)
# ---------------------------------------------------------------------------


class TestWriteTimeseriesChunk:
    def _setup_dataset(self, saver, h5f, eeg_metadata):
        """Register metadata and create dataset for EEG modality."""
        saver.stream_metadata["EEG"] = eeg_metadata
        dataset = saver._create_timeseries_dataset(h5f, "EEG")
        saver.datasets["EEG"] = dataset
        return dataset

    def _make_buf(self, data, timestamps, local_ts=None, receive_ts=None):
        """Build a data buffer dict with all required timestamp arrays."""
        if local_ts is None:
            local_ts = [t.copy() for t in timestamps]
        if receive_ts is None:
            receive_ts = [t.copy() for t in timestamps]
        return {"data": data, "timestamps": timestamps,
                "local_timestamps": local_ts, "receive_timestamps": receive_ts}

    def test_correct_data_written(self, saver, h5f, eeg_metadata):
        dataset = self._setup_dataset(saver, h5f, eeg_metadata)

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        timestamps = np.array([10.0, 20.0], dtype=np.float64)
        saver.data_buffers["EEG"] = self._make_buf([data], [timestamps])
        saver._write_timeseries_chunk(h5f, "EEG")

        assert dataset.shape[0] == 2
        assert float(dataset[0]["Fp1"]) == pytest.approx(1.0)
        assert float(dataset[1]["Cz"]) == pytest.approx(6.0)
        assert float(dataset[0]["timestamp"]) == pytest.approx(10.0)

    def test_multiple_chunks_concatenated(self, saver, h5f, eeg_metadata):
        dataset = self._setup_dataset(saver, h5f, eeg_metadata)

        chunk1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        chunk2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        ts1 = np.array([10.0], dtype=np.float64)
        ts2 = np.array([20.0], dtype=np.float64)
        saver.data_buffers["EEG"] = self._make_buf([chunk1, chunk2], [ts1, ts2])
        saver._write_timeseries_chunk(h5f, "EEG")

        assert dataset.shape[0] == 2
        assert float(dataset[0]["Fp1"]) == pytest.approx(1.0)
        assert float(dataset[1]["Fp1"]) == pytest.approx(4.0)

    def test_empty_buffer_noop(self, saver, h5f, eeg_metadata):
        self._setup_dataset(saver, h5f, eeg_metadata)
        saver.data_buffers["EEG"] = self._make_buf([], [])
        saver._write_timeseries_chunk(h5f, "EEG")
        assert saver.datasets["EEG"].shape[0] == 0


# ---------------------------------------------------------------------------
# Event chunk writing
# ---------------------------------------------------------------------------


class TestWriteEventChunk:
    def test_correct_data_written(self, saver, h5f):
        saver.datasets["Event"] = saver._create_event_dataset(h5f)

        event_sample = {"event_id": 7, "event_type": "left_hand", "duration": 1.5}
        saver.event_buffer = [EventRecord(event_sample, 100.0, 100.0, 100.0)]

        saver._write_event_chunk(h5f)

        dataset = saver.datasets["Event"]
        assert dataset.shape[0] == 1
        assert int(dataset[0]["event_id"]) == 7
        assert dataset[0]["event_type"].decode() == "left_hand"
        assert float(dataset[0]["timestamp"]) == pytest.approx(100.0)
        extra = json.loads(dataset[0]["extra_vars"].decode())
        assert extra["duration"] == 1.5


# ---------------------------------------------------------------------------
# Upfront dataset creation
# ---------------------------------------------------------------------------


class TestCreateAllDatasets:
    def test_creates_datasets_from_stream_configs(self, tmp_path):
        from dendrite.data.stream_schemas import StreamMetadata

        filename = str(tmp_path / "test_upfront.h5")
        evt = multiprocessing.Event()
        config = StreamMetadata(
            name="TestEEG", type="EEG", channel_count=3,
            sample_rate=256.0, labels=["Fp1", "Fp2", "Cz"],
            channel_types=["eeg", "eeg", "eeg"],
        )
        ds = DataSaver(
            filename=filename,
            stop_event=evt,
            ring_buffer_names={"EEG": "test_rb_eeg"},
            ring_buffer_channel_maps={"EEG": {"sample_rate": 256.0}},
            stream_configs=[config],
            chunk_size=10,
        )
        ds.logger = logging.getLogger("test.DataSaver")

        with h5py.File(filename, "w") as h5f:
            ds._create_all_datasets(h5f)
            assert "EEG" in ds.datasets
            assert "Event" in ds.datasets
            assert ds.datasets["EEG"].dtype.names is not None
            assert "Fp1" in ds.datasets["EEG"].dtype.names


# ---------------------------------------------------------------------------
# Global metadata
# ---------------------------------------------------------------------------


class TestGlobalMetadata:
    def test_global_metadata_written_to_attrs(self, tmp_path):
        filename = str(tmp_path / "test_meta.h5")
        evt = multiprocessing.Event()
        ds = DataSaver(
            filename=filename,
            stop_event=evt,
            ring_buffer_names={},
            ring_buffer_channel_maps={},
            global_metadata={"study_name": "Motor Imagery", "version": "2.0"},
            chunk_size=10,
        )
        ds.logger = logging.getLogger("test.DataSaver")

        with h5py.File(filename, "w") as h5f:
            ds._initialize_file(h5f)

        with h5py.File(filename, "r") as h5f:
            assert h5f.attrs["study_name"] == "Motor Imagery"


# ---------------------------------------------------------------------------
# Buffer clearing after write
# ---------------------------------------------------------------------------


class TestBufferClear:
    """Verify all buffer arrays are cleared after chunk write."""

    def test_all_buffer_keys_cleared(self, saver, h5f, eeg_metadata):
        """After writing a chunk, all 4 buffer arrays should be empty."""
        saver.stream_metadata["EEG"] = eeg_metadata
        saver.datasets["EEG"] = saver._create_timeseries_dataset(h5f, "EEG")

        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ts = np.array([10.0], dtype=np.float64)
        saver.data_buffers["EEG"] = {
            "data": [data], "timestamps": [ts],
            "local_timestamps": [ts.copy()], "receive_timestamps": [ts.copy()],
        }
        saver._write_timeseries_chunk(h5f, "EEG")

        for key, arr_list in saver.data_buffers["EEG"].items():
            assert len(arr_list) == 0, f"Buffer '{key}' not cleared after write"


# ---------------------------------------------------------------------------
# Three-timestamp verification
# ---------------------------------------------------------------------------


class TestSwmrConcurrentReadWrite:
    """Verify SWMR: writer writes chunks, concurrent reader sees them after refresh."""

    def test_reader_sees_chunks_after_refresh(self, tmp_path, eeg_metadata):
        """Write 5 chunks with SWMR enabled, reader refreshes and sees growing dataset."""
        filename = str(tmp_path / "swmr_test.h5")
        evt = multiprocessing.Event()
        saver = DataSaver(
            filename=filename, stop_event=evt,
            ring_buffer_names={}, ring_buffer_channel_maps={},
            chunk_size=100,
        )
        saver.logger = logging.getLogger("test.DataSaver")

        with h5py.File(filename, "w", libver="latest") as h5f:
            saver.stream_metadata["EEG"] = eeg_metadata
            saver.datasets["EEG"] = saver._create_timeseries_dataset(h5f, "EEG")
            saver.datasets["Event"] = saver._create_event_dataset(h5f)

            # Enable SWMR (must happen after dataset creation)
            h5f.swmr_mode = True

            # Open concurrent reader
            with h5py.File(filename, "r", swmr=True) as reader:
                r_eeg = reader["EEG"]
                r_ev = reader["Event"]

                # Write 5 chunks of 100 samples, verify reader sees each
                for chunk_idx in range(5):
                    n = 100
                    data = np.random.randn(n, 3).astype(np.float32)
                    ts = np.arange(n, dtype=np.float64) + chunk_idx * n
                    saver.data_buffers["EEG"] = {
                        "data": [data], "timestamps": [ts],
                        "local_timestamps": [ts.copy()],
                        "receive_timestamps": [ts.copy()],
                    }
                    saver._write_timeseries_chunk(h5f, "EEG")
                    h5f.flush()

                    r_eeg.refresh()
                    expected = (chunk_idx + 1) * n
                    assert r_eeg.shape[0] == expected, (
                        f"Chunk {chunk_idx}: reader sees {r_eeg.shape[0]}, expected {expected}"
                    )

                # Write 3 events, verify reader sees them
                for i in range(3):
                    saver.event_buffer.append(
                        EventRecord(
                            {"event_id": i, "event_type": f"ev_{i}"},
                            float(i * 10), float(i * 10), float(i * 10),
                        )
                    )
                saver._write_event_chunk(h5f)
                h5f.flush()

                r_ev.refresh()
                assert r_ev.shape[0] == 3
                assert int(r_ev[0]["event_id"]) == 0
                assert r_ev[2]["event_type"].decode() == "ev_2"

    def test_raw_h5_loader_reads_swmr_file(self, tmp_path, eeg_metadata):
        """RawH5Loader can read a file written with SWMR after writer closes."""
        from dendrite.data.loaders.raw_h5_loader import RawH5Loader

        filename = str(tmp_path / "swmr_loader_test.h5")
        evt = multiprocessing.Event()
        saver = DataSaver(
            filename=filename, stop_event=evt,
            ring_buffer_names={}, ring_buffer_channel_maps={},
            chunk_size=100,
        )
        saver.logger = logging.getLogger("test.DataSaver")

        with h5py.File(filename, "w", libver="latest") as h5f:
            saver.stream_metadata["EEG"] = eeg_metadata
            saver.datasets["EEG"] = saver._create_timeseries_dataset(h5f, "EEG")
            saver.datasets["Event"] = saver._create_event_dataset(h5f)

            # Write attrs so loader can find sample rate
            h5f["EEG"].attrs["sampling_frequency"] = 256.0

            h5f.swmr_mode = True

            # Write 200 samples
            for _ in range(2):
                data = np.random.randn(100, 3).astype(np.float32)
                ts = np.arange(100, dtype=np.float64)
                saver.data_buffers["EEG"] = {
                    "data": [data], "timestamps": [ts],
                    "local_timestamps": [ts.copy()],
                    "receive_timestamps": [ts.copy()],
                }
                saver._write_timeseries_chunk(h5f, "EEG")

            # Write 2 events
            for i in range(2):
                saver.event_buffer.append(
                    EventRecord(
                        {"event_id": i + 1, "event_type": f"stim_{i}"},
                        float(i), float(i), float(i),
                    )
                )
            saver._write_event_chunk(h5f)
            h5f.flush()

        # Now load with RawH5Loader (file closed, no SWMR needed)
        loaded = RawH5Loader(filename).load()
        assert loaded.data.shape[0] == 3  # 3 EEG channels (no timestamps)
        assert loaded.data.shape[1] == 200
        assert loaded.sample_rate == 256.0
        assert len(loaded.events) == 2
        assert loaded.channel_names == ["Fp1", "Fp2", "Cz"]


class TestThreeTimestamps:
    """Verify all 3 timestamps are written to timeseries dataset."""

    def test_timestamps_distinct(self, saver, h5f, eeg_metadata):
        saver.stream_metadata["EEG"] = eeg_metadata
        saver.datasets["EEG"] = saver._create_timeseries_dataset(h5f, "EEG")

        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        lsl_ts = np.array([100.0], dtype=np.float64)
        local_ts = np.array([99.5], dtype=np.float64)
        receive_ts = np.array([1711234567.123], dtype=np.float64)
        saver.data_buffers["EEG"] = {
            "data": [data], "timestamps": [lsl_ts],
            "local_timestamps": [local_ts], "receive_timestamps": [receive_ts],
        }
        saver._write_timeseries_chunk(h5f, "EEG")

        ds = saver.datasets["EEG"]
        assert float(ds[0]["timestamp"]) == pytest.approx(100.0)
        assert float(ds[0]["local_timestamp"]) == pytest.approx(99.5)
        assert float(ds[0]["receive_timestamp"]) == pytest.approx(1711234567.123)
