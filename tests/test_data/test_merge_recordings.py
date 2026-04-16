"""Tests for merge_recordings — shape validation, label remapping, single passthrough."""

import numpy as np
import pytest

from dendrite.data.loaders._training_data import merge_recordings
from dendrite.data.loaders._types import EpochedData

# ---- Helpers ---- #

def _make_study_data(
    n_epochs: int = 10,
    n_channels: int = 4,
    n_times: int = 100,
    class_names: list | None = None,
    label_map: dict | None = None,
    event_id: dict | None = None,
    recording_id: int = 1,
) -> EpochedData:
    class_names = class_names or ["left", "right"]
    label_map = label_map or {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)
    X = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)
    y = np.array([i % n_classes for i in range(n_epochs)], dtype=np.int64)
    return EpochedData(
        X=X, y=y,
        metadata={
            "paradigm": "Recording",
            "class_names": class_names,
            "class_counts": {name: int((y == i).sum()) for i, name in enumerate(class_names)},
            "label_map": label_map,
            "n_channels": n_channels,
            "n_times": n_times,
            "recording_id": recording_id,
            "event_id": event_id or {},
        },
        source="recording", source_id=str(recording_id),
        sample_rate=250.0,
        channel_names=[f"Ch{i}" for i in range(n_channels)],
        channel_types=["eeg"] * n_channels,
    )


class FakeRecordingsRepo:
    def __init__(self, records: dict[int, dict]):
        self._records = records

    def get_by_id(self, rid: int):
        return self._records.get(rid)


class FakeDataService:
    def __init__(self, records: dict[int, dict]):
        self.recordings = FakeRecordingsRepo(records)


def _records(*ids):
    return {rid: {"recording_id": rid, "recording_name": f"rec_{rid}",
                  "subject_id": f"S{rid:02d}", "hdf5_file_path": f"/fake/{rid}.h5"}
            for rid in ids}


# ---- Tests ---- #

class TestSingleRecording:
    def test_passthrough(self, monkeypatch):
        """Single recording returns the dataset directly with metadata additions."""
        sd = _make_study_data(recording_id=1)
        ds = FakeDataService(_records(1))

        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: sd)

        result = merge_recordings([1], {}, ds)
        assert result is sd
        assert result.metadata["n_recordings"] == 1
        assert result.metadata["subject_breakdown"] == {"S01": 1}


class TestMultipleRecordings:
    def test_same_labels_concatenate(self, monkeypatch):
        """Two recordings with same label maps are concatenated directly."""
        sd1 = _make_study_data(n_epochs=10, recording_id=1)
        sd2 = _make_study_data(n_epochs=8, recording_id=2)
        ds = FakeDataService(_records(1, 2))

        datas = iter([sd1, sd2])
        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: next(datas))

        result = merge_recordings([1, 2], {}, ds)
        assert result.X.shape[0] == 18
        assert result.y.shape[0] == 18
        assert result.metadata["n_recordings"] == 2
        assert result.metadata["subject_breakdown"] == {"S01": 1, "S02": 1}

    def test_different_labels_remap(self, monkeypatch):
        """Two recordings with different label maps get unified encoding."""
        sd1 = _make_study_data(
            n_epochs=6, recording_id=1,
            class_names=["left", "right"],
            label_map={"left": 0, "right": 1},
        )
        sd2 = _make_study_data(
            n_epochs=6, recording_id=2,
            class_names=["feet", "right"],
            label_map={"feet": 0, "right": 1},
        )
        ds = FakeDataService(_records(1, 2))

        datas = iter([sd1, sd2])
        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: next(datas))

        result = merge_recordings([1, 2], {}, ds)
        assert result.X.shape[0] == 12
        assert sorted(result.metadata["class_names"]) == ["feet", "left", "right"]
        assert set(result.y.tolist()).issubset({0, 1, 2})

    def test_shape_mismatch_raises(self, monkeypatch):
        """Mismatched spatial shapes raise ValueError."""
        sd1 = _make_study_data(n_epochs=5, n_channels=4, recording_id=1)
        sd2 = _make_study_data(n_epochs=5, n_channels=8, recording_id=2)
        ds = FakeDataService(_records(1, 2))

        datas = iter([sd1, sd2])
        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: next(datas))

        with pytest.raises(ValueError, match="Shape mismatch"):
            merge_recordings([1, 2], {}, ds)

    def test_event_id_merged(self, monkeypatch):
        """Event IDs from all recordings are merged."""
        sd1 = _make_study_data(recording_id=1, event_id={"left": 1, "right": 2})
        sd2 = _make_study_data(recording_id=2, event_id={"right": 2, "feet": 3})
        ds = FakeDataService(_records(1, 2))

        datas = iter([sd1, sd2])
        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: next(datas))

        result = merge_recordings([1, 2], {}, ds)
        assert result.metadata["event_id"] == {"left": 1, "right": 2, "feet": 3}


class TestBroadcastStep:
    def test_step_callback_called(self, monkeypatch):
        """broadcast_step is called for each recording."""
        sd = _make_study_data(recording_id=1)
        ds = FakeDataService(_records(1, 2))

        import dendrite.data.loaders._training_data as mod
        monkeypatch.setattr(mod, "load_epochs", lambda cfg, path, **kw: sd)

        steps = []
        merge_recordings([1, 2], {}, ds, broadcast_step=steps.append)
        assert len(steps) == 2
        assert "1/2" in steps[0]
        assert "2/2" in steps[1]
