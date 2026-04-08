"""Tests for DataService signal preview and event summary methods."""

import h5py
import numpy as np
import pytest

from dendrite.web.services.data_service import DataService


@pytest.fixture
def svc(tmp_path):
    """DataService with an isolated temp database."""
    db_path = str(tmp_path / "test.db")
    return DataService(db_path=db_path)


def _add_recording(svc, h5_path: str) -> int:
    """Helper: create a study + recording pointing at the given H5 path."""
    study = svc.studies.get_or_create("test_study")
    return svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path=h5_path,
    )


def _create_eeg_h5(path, n_samples=1000, n_channels=8, sfreq=500.0):
    """Create a minimal H5 file with an EEG dataset matching DataSaver layout."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_samples, n_channels)).astype(np.float64)
    labels = [f"Ch{i}" for i in range(n_channels)]

    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("EEG", data=data)
        ds.attrs["channel_labels"] = labels
        ds.attrs["sampling_frequency"] = sfreq
    return data, labels


def _create_event_h5(path, events):
    """Create an H5 file with a structured Event dataset."""
    dt = np.dtype([
        ("event_id", np.int32),
        ("event_type", h5py.string_dtype(encoding="utf-8")),
        ("timestamp", np.float64),
        ("local_timestamp", np.float64),
        ("extra_vars", h5py.string_dtype(encoding="utf-8")),
    ])
    arr = np.array(events, dtype=dt)

    with h5py.File(str(path), "a") as f:
        f.create_dataset("Event", data=arr)


# --- Signal Preview ---


class TestGetSignalPreview:
    def test_no_recording(self, svc):
        assert svc.get_signal_preview(9999) is None

    def test_missing_file(self, svc, tmp_path):
        rid = _add_recording(svc, "/nonexistent/data.h5")
        with pytest.raises(FileNotFoundError):
            svc.get_signal_preview(rid)

    def test_returns_downsampled(self, svc, tmp_path):
        h5_path = tmp_path / "eeg.h5"
        data, labels = _create_eeg_h5(h5_path, n_samples=1000, n_channels=8, sfreq=500.0)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_signal_preview(rid, max_points=100)

        assert "EEG" in result
        eeg = result["EEG"]
        assert eeg["sample_rate"] == 500.0
        assert eeg["total_samples"] == 1000
        # Downsampled: step = 1000 // 100 = 10, so display = 100
        assert eeg["display_samples"] == 100
        assert len(eeg["time"]) == 100
        assert len(eeg["channels"]) == 8
        assert eeg["channels"][0]["label"] == "Ch0"
        assert len(eeg["channels"][0]["data"]) == 100

    def test_respects_max_channels(self, svc, tmp_path):
        h5_path = tmp_path / "eeg.h5"
        _create_eeg_h5(h5_path, n_samples=200, n_channels=16)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_signal_preview(rid, max_channels=4)

        assert len(result["EEG"]["channels"]) == 4

    def test_multiple_modalities(self, svc, tmp_path):
        h5_path = tmp_path / "multi.h5"
        rng = np.random.default_rng(0)

        with h5py.File(str(h5_path), "w") as f:
            eeg = f.create_dataset("EEG", data=rng.standard_normal((500, 4)))
            eeg.attrs["channel_labels"] = ["Fp1", "Fz", "Cz", "O1"]
            eeg.attrs["sampling_frequency"] = 250.0

            emg = f.create_dataset("EMG", data=rng.standard_normal((500, 2)))
            emg.attrs["channel_labels"] = ["EMG1", "EMG2"]
            emg.attrs["sampling_frequency"] = 1000.0

        rid = _add_recording(svc, str(h5_path))
        result = svc.get_signal_preview(rid)

        assert "EEG" in result
        assert "EMG" in result
        assert result["EEG"]["sample_rate"] == 250.0
        assert result["EMG"]["sample_rate"] == 1000.0

    def test_skips_structured_arrays(self, svc, tmp_path):
        """Event datasets (structured arrays) should not appear in signal preview."""
        h5_path = tmp_path / "with_events.h5"
        _create_eeg_h5(h5_path, n_samples=200, n_channels=4)
        _create_event_h5(h5_path, [(1, "left", 0.1, 0.1, "{}")])
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_signal_preview(rid)

        assert "EEG" in result
        assert "Event" not in result

    def test_small_data_no_downsampling(self, svc, tmp_path):
        """When data is smaller than max_points, return all samples."""
        h5_path = tmp_path / "small.h5"
        _create_eeg_h5(h5_path, n_samples=50, n_channels=2)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_signal_preview(rid, max_points=15000)

        assert result["EEG"]["display_samples"] == 50
        assert result["EEG"]["total_samples"] == 50


# --- Event Summary ---


class TestGetEventSummary:
    def test_no_recording(self, svc):
        assert svc.get_event_summary(9999) is None

    def test_missing_file(self, svc, tmp_path):
        rid = _add_recording(svc, "/nonexistent/data.h5")
        with pytest.raises(FileNotFoundError):
            svc.get_event_summary(rid)

    def test_no_event_dataset(self, svc, tmp_path):
        """H5 exists but has no Event dataset — return empty summary."""
        h5_path = tmp_path / "no_events.h5"
        _create_eeg_h5(h5_path)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_event_summary(rid)

        assert result["total_count"] == 0
        assert result["event_types"] == {}
        assert result["events"] == []

    def test_with_events(self, svc, tmp_path):
        h5_path = tmp_path / "events.h5"
        _create_eeg_h5(h5_path, n_samples=500)
        _create_event_h5(h5_path, [
            (1, "left_hand", 1.0, 1.0, "{}"),
            (2, "right_hand", 2.0, 2.0, "{}"),
            (3, "left_hand", 3.0, 3.0, "{}"),
            (4, "left_hand", 4.0, 4.0, "{}"),
            (5, "right_hand", 5.0, 5.0, "{}"),
        ])
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_event_summary(rid)

        assert result["total_count"] == 5
        assert result["event_types"]["left_hand"] == 3
        assert result["event_types"]["right_hand"] == 2
        assert len(result["events"]) == 5

    def test_many_events_returned(self, svc, tmp_path):
        """All events should be returned regardless of count."""
        h5_path = tmp_path / "many_events.h5"
        _create_eeg_h5(h5_path)
        events = [
            (i, "stim", float(i), float(i), "{}")
            for i in range(300)
        ]
        _create_event_h5(h5_path, events)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_event_summary(rid)

        assert result["total_count"] == 300
        assert len(result["events"]) == 300


# --- Metrics H5 helpers ---


def _create_metrics_h5(path, telemetry=None, modes=None):
    """Create a metrics H5 file with optional telemetry and mode groups.

    telemetry: dict of {key: (data_array, timestamps_array)} for the telemetry/ group
    modes: dict of {mode_name: {ds_key: (data_array, timestamps_array)}} for mode groups
    """
    with h5py.File(str(path), "w") as f:
        # Always include script_metadata (marker for metrics files)
        meta = f.create_group("script_metadata")
        meta.attrs["sample_rate"] = 500.0

        if telemetry:
            tg = f.create_group("telemetry")
            for key, (data, timestamps) in telemetry.items():
                tg.create_dataset(key, data=np.asarray(data, dtype=np.float64))
                if timestamps is not None:
                    tg.create_dataset(
                        f"{key}_timestamps",
                        data=np.asarray(timestamps, dtype=np.float64),
                    )

        if modes:
            for mode_name, datasets in modes.items():
                mg = f.create_group(mode_name)
                for ds_key, (data, timestamps) in datasets.items():
                    mg.create_dataset(ds_key, data=np.asarray(data, dtype=np.float64))
                    if timestamps is not None:
                        mg.create_dataset(
                            f"{ds_key}_timestamps",
                            data=np.asarray(timestamps, dtype=np.float64),
                        )


def _add_recording_with_metrics(svc, tmp_path, monkeypatch):
    """Create a recording with study paths pointing to tmp_path.

    Returns (recording_id, raw_dir, metrics_dir, file_id).
    """
    import os
    from pathlib import Path

    study_name = "test_study_metrics"
    file_id = "task-rec1_run-01_20240101_120000"
    paths = {
        "config": tmp_path / "config",
        "raw": tmp_path / "raw",
        "metrics": tmp_path / "metrics",
        "decoders": tmp_path / "decoders",
        "logs": tmp_path / "logs",
    }
    for d in paths.values():
        os.makedirs(d, exist_ok=True)

    monkeypatch.setattr(
        "dendrite.web.services.data_service.get_study_paths",
        lambda name: paths if name == study_name else {},
    )

    raw_path = str(paths["raw"] / f"{file_id}_raw.h5")
    study = svc.studies.get_or_create(study_name)
    rid = svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000_m",
        hdf5_file_path=raw_path,
        file_identifier=file_id,
    )
    return rid, paths, file_id


# --- Session Summary ---


class TestGetSessionSummary:
    def test_no_recording(self, svc):
        assert svc.get_session_summary(9999) is None

    def test_raw_only(self, svc, tmp_path):
        """Raw H5 without metrics — basic summary, no modes."""
        h5_path = tmp_path / "eeg.h5"
        _create_eeg_h5(h5_path, n_samples=1000, n_channels=8, sfreq=500.0)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_session_summary(rid)

        assert result["duration_seconds"] == pytest.approx(2.0)  # 1000/500
        assert result["sample_rate"] == 500.0
        assert result["channels"] == 8
        assert "EEG" in result["datasets"]
        assert result["modes"] == []
        assert result["has_metrics"] is False

    def test_with_metrics(self, svc, tmp_path, monkeypatch):
        """Raw + metrics H5 — includes modes and has_metrics flag."""
        rid, paths, file_id = _add_recording_with_metrics(svc, tmp_path, monkeypatch)
        _create_eeg_h5(paths["raw"] / f"{file_id}_raw.h5", n_samples=500, n_channels=4, sfreq=250.0)
        _create_metrics_h5(
            paths["metrics"] / f"{file_id}_metrics.h5",
            modes={"my_mode": {"accuracy": (np.ones(10), np.arange(10, dtype=float))}},
        )

        result = svc.get_session_summary(rid)

        assert result["duration_seconds"] == pytest.approx(2.0)  # 500/250
        assert result["has_metrics"] is True
        assert "my_mode" in result["modes"]


# --- Telemetry ---


class TestGetTelemetry:
    def test_no_metrics(self, svc, tmp_path):
        """No metrics file → empty structure."""
        h5_path = tmp_path / "eeg.h5"
        _create_eeg_h5(h5_path)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_telemetry(rid)

        assert result == {"latencies": {}, "mode_metrics": {}, "bandwidth": {}}

    def test_with_data(self, svc, tmp_path, monkeypatch):
        """Metrics H5 with telemetry data → verify latencies dict."""
        rid, paths, file_id = _add_recording_with_metrics(svc, tmp_path, monkeypatch)
        _create_eeg_h5(paths["raw"] / f"{file_id}_raw.h5")

        latency_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        latency_ts = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        _create_metrics_h5(
            paths["metrics"] / f"{file_id}_metrics.h5",
            telemetry={"eeg_latency_ms": (latency_data, latency_ts)},
        )

        result = svc.get_telemetry(rid)

        assert "eeg_latency_ms" in result["latencies"]
        metric = result["latencies"]["eeg_latency_ms"]
        assert metric["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        # Time should be relative (offset from first timestamp)
        assert metric["time"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_no_recording(self, svc):
        assert svc.get_telemetry(9999) is None


# --- Mode Performance ---


class TestGetModePerformance:
    def test_no_metrics(self, svc, tmp_path):
        """No metrics file → empty dict."""
        h5_path = tmp_path / "eeg.h5"
        _create_eeg_h5(h5_path)
        rid = _add_recording(svc, str(h5_path))

        result = svc.get_mode_performance(rid)

        assert result == {}

    def test_with_data(self, svc, tmp_path, monkeypatch):
        """Metrics H5 with mode data → verify returned time-series."""
        rid, paths, file_id = _add_recording_with_metrics(svc, tmp_path, monkeypatch)
        _create_eeg_h5(paths["raw"] / f"{file_id}_raw.h5")

        acc_data = np.array([0.5, 0.6, 0.7, 0.8])
        acc_ts = np.array([10.0, 11.0, 12.0, 13.0])
        conf_data = np.array([0.9, 0.85, 0.88, 0.92])
        conf_ts = np.array([10.0, 11.0, 12.0, 13.0])
        _create_metrics_h5(
            paths["metrics"] / f"{file_id}_metrics.h5",
            modes={
                "sync_mode_1": {
                    "accuracy": (acc_data, acc_ts),
                    "confidence": (conf_data, conf_ts),
                }
            },
        )

        result = svc.get_mode_performance(rid)

        assert "sync_mode_1" in result
        mode = result["sync_mode_1"]
        assert "accuracy" in mode
        assert "confidence" in mode
        assert mode["accuracy"]["values"] == [0.5, 0.6, 0.7, 0.8]
        assert mode["accuracy"]["time"] == [0.0, 1.0, 2.0, 3.0]
        assert mode["confidence"]["values"] == [0.9, 0.85, 0.88, 0.92]

    def test_nan_values_become_null(self, svc, tmp_path, monkeypatch):
        """NaN/Inf in H5 data should become None (JSON null), not crash."""
        rid, paths, file_id = _add_recording_with_metrics(svc, tmp_path, monkeypatch)
        _create_eeg_h5(paths["raw"] / f"{file_id}_raw.h5")

        data_with_nan = np.array([0.5, float("nan"), 0.7, float("inf")])
        ts = np.array([10.0, 11.0, 12.0, 13.0])
        _create_metrics_h5(
            paths["metrics"] / f"{file_id}_metrics.h5",
            modes={"m1": {"accuracy": (data_with_nan, ts)}},
        )

        result = svc.get_mode_performance(rid)

        vals = result["m1"]["accuracy"]["values"]
        assert vals[0] == 0.5
        assert vals[1] is None  # NaN → None
        assert vals[2] == 0.7
        assert vals[3] is None  # Inf → None

    def test_no_recording(self, svc):
        assert svc.get_mode_performance(9999) is None
