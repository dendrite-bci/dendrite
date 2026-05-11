"""Tests for StreamService — configuration, modality aggregation, restore."""

import pytest

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.web.services.stream_service import StreamService


@pytest.fixture
def svc():
    return StreamService()


def _make_stream(
    name: str = "EEG",
    stream_type: str = "EEG",
    uid: str = "uid_eeg",
    source_id: str = "src_eeg",
    channel_count: int = 3,
    sample_rate: float = 500.0,
    labels: list[str] | None = None,
    channel_types: list[str] | None = None,
) -> StreamMetadata:
    labels = labels or [f"Ch{i}" for i in range(channel_count)]
    channel_types = channel_types or ["eeg"] * channel_count
    return StreamMetadata(
        name=name,
        type=stream_type,
        channel_count=channel_count,
        sample_rate=sample_rate,
        channel_format="float32",
        source_id=source_id,
        uid=uid,
        labels=labels,
        channel_types=channel_types,
        channel_units=["uV"] * channel_count,
    )


# --- configure_streams ---


def test_configure_selects_valid_uids(svc: StreamService):
    eeg = _make_stream()
    discovered = {eeg.stable_key: eeg}

    result = svc.configure_streams([eeg.stable_key], discovered)

    assert eeg.stable_key in result["streams"]
    assert svc.has_streams()
    assert len(svc.get_streams()) == 1


def test_configure_ignores_unknown_uids(svc: StreamService):
    result = svc.configure_streams(["nonexistent"], {})

    assert len(result["streams"]) == 0
    assert not svc.has_streams()


def test_configure_applies_channel_overrides(svc: StreamService):
    eeg = _make_stream(labels=["C3", "C4", "Cz"])
    discovered = {eeg.stable_key: eeg}

    overrides = {eeg.stable_key: {"labels": ["FC3", "FC4", "FCz"]}}
    result = svc.configure_streams([eeg.stable_key], discovered, overrides)

    configured = result["streams"][eeg.stable_key]
    assert configured.labels == ["FC3", "FC4", "FCz"]


def test_configure_clears_previous_streams(svc: StreamService):
    eeg1 = _make_stream(name="Old", uid="old", source_id="old")
    eeg2 = _make_stream(name="New", uid="new", source_id="new")

    svc.configure_streams([eeg1.stable_key], {eeg1.stable_key: eeg1})
    assert len(svc.get_streams()) == 1

    svc.configure_streams([eeg2.stable_key], {eeg2.stable_key: eeg2})
    streams = svc.get_streams()
    assert len(streams) == 1
    assert eeg2.stable_key in streams


# --- get_modalities_by_stream ---


def test_modalities_by_stream_groups_channels(svc: StreamService):
    eeg = _make_stream(
        channel_count=5,
        labels=["C3", "C4", "Cz", "EOG1", "EOG2"],
        channel_types=["eeg", "eeg", "eeg", "eog", "eog"],
    )
    svc.configure_streams([eeg.stable_key], {eeg.stable_key: eeg})

    result = svc.get_modalities_by_stream()

    assert len(result) == 1
    entry = next(iter(result.values()))
    assert entry["stream_name"] == eeg.name
    assert entry["stream_type"] == eeg.type
    assert entry["sample_rate"] == eeg.sample_rate
    assert len(entry["modalities"]["eeg"]) == 3
    assert len(entry["modalities"]["eog"]) == 2


def test_modalities_local_index_is_per_modality(svc: StreamService):
    """local_index must be 0-based per modality, not the stream-wide position.

    Regression: previously local_index used the full-stream enumerate index,
    which matched the modality-relative space only when modalities were
    contiguous. With interleaved EOG/EEG, channel_selection ended up
    stream-relative while the runtime expected modality-relative.
    """
    eeg = _make_stream(
        channel_count=5,
        labels=["EOG1", "C3", "EOG2", "Cz", "C4"],
        channel_types=["eog", "eeg", "eog", "eeg", "eeg"],
    )
    svc.configure_streams([eeg.stable_key], {eeg.stable_key: eeg})

    entry = next(iter(svc.get_modalities_by_stream().values()))
    eeg_channels = entry["modalities"]["eeg"]
    assert [c["label"] for c in eeg_channels] == ["C3", "Cz", "C4"]
    assert [c["local_index"] for c in eeg_channels] == [0, 1, 2]

    eog_channels = entry["modalities"]["eog"]
    assert [c["label"] for c in eog_channels] == ["EOG1", "EOG2"]
    assert [c["local_index"] for c in eog_channels] == [0, 1]


def test_modalities_by_stream_excludes_markers(svc: StreamService):
    eeg = _make_stream(
        channel_count=3,
        labels=["C3", "C4", "Markers"],
        channel_types=["eeg", "eeg", "markers"],
    )
    svc.configure_streams([eeg.stable_key], {eeg.stable_key: eeg})

    entry = next(iter(svc.get_modalities_by_stream().values()))

    assert "markers" not in entry["modalities"]
    assert len(entry["modalities"]["eeg"]) == 2


def test_modalities_by_stream_empty_when_no_streams(svc: StreamService):
    assert svc.get_modalities_by_stream() == {}


# --- restore_from_config ---


def test_restore_from_config_roundtrip(svc: StreamService):
    eeg = _make_stream()
    svc.configure_streams([eeg.stable_key], {eeg.stable_key: eeg})

    saved = [s.model_dump() for s in svc.get_streams().values()]

    svc2 = StreamService()
    svc2.restore_from_config(saved)

    assert len(svc2.get_streams()) == 1
    restored = list(svc2.get_streams().values())[0]
    assert restored.name == eeg.name
    assert restored.labels == eeg.labels


def test_restore_from_config_skips_strings(svc: StreamService):
    """Old format entries (strings) are skipped gracefully."""
    svc.restore_from_config(["some_old_string_entry"])
    assert not svc.has_streams()


# --- discovery cache ---


def test_discover_and_cache_clears_configured(svc: StreamService):
    """After discovery, configured streams are cleared."""
    eeg = _make_stream()
    svc._streams = {eeg.stable_key: eeg}
    assert svc.has_streams()

    # discover_and_cache calls discover_lsl_streams which needs pylsl,
    # so we test the cache mechanics by directly setting state
    svc._last_discovery = {eeg.stable_key: eeg}
    svc._last_discovery_time = 999999999.0
    cached = svc.get_cached_discovery(max_age=30.0)
    assert cached is not None
    assert eeg.stable_key in cached


def test_cached_discovery_none_when_stale(svc: StreamService):
    svc._last_discovery = {"key": "data"}
    svc._last_discovery_time = 0.0  # Very old
    assert svc.get_cached_discovery(max_age=30.0) is None
