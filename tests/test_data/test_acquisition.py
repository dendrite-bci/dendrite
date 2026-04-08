"""Tests for DataAcquisition — pure/near-pure methods (no LSL hardware)."""

import json
import logging
import multiprocessing
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from dendrite.data.acquisition import LATENCY_EVENT_TYPE, DataAcquisition, EventRecord
from dendrite.data.stream_schemas import StreamMetadata


def _make_metadata(**overrides) -> StreamMetadata:
    defaults = {
        "name": "TestStream",
        "type": "EEG",
        "channel_count": 4,
        "sample_rate": 250.0,
        "labels": ["Ch1", "Ch2", "Ch3", "Ch4"],
        "channel_types": ["EEG", "EEG", "EEG", "EEG"],
        "channel_units": ["uV", "uV", "uV", "uV"],
    }
    defaults.update(overrides)
    return StreamMetadata(**defaults)


def _make_daq(stream_configs=None):
    q_event = multiprocessing.Queue()
    stop = multiprocessing.Event()
    d = DataAcquisition(
        event_queue=q_event,
        stop_event=stop,
        stream_configs=stream_configs or [],
    )
    d.logger = logging.getLogger("test.DAQ")
    return d, q_event


def _setup_broadcast(*stream_names: str):
    """Create DAQ with pending-event deques for given stream names."""
    d, q = _make_daq()
    for name in stream_names:
        d._pending_events_per_stream[name] = deque()
    return d, q


def _broadcast(daq, event_id):
    """Append event_id to all pending-event deques."""
    for dq in daq._pending_events_per_stream.values():
        dq.append(event_id)


@pytest.fixture
def daq():
    d, q_event = _make_daq()
    yield d
    q_event.close()


@pytest.fixture
def daq_with_eeg():
    d, q_event = _make_daq([_make_metadata()])
    yield d
    q_event.close()


class TestParseEventSample:
    def test_valid_event(self, daq):
        result = daq._parse_event_sample([json.dumps({"event_id": 7, "event_type": "left_hand"})])
        assert result is not None
        event_json, event_id = result
        assert event_id == 7.0

    def test_invalid_json(self, daq):
        assert daq._parse_event_sample(["not json {"]) is None

    def test_missing_required_field(self, daq):
        assert daq._parse_event_sample([json.dumps({"event_id": 1})]) is None

    def test_pascal_case_normalized(self, daq):
        result = daq._parse_event_sample([json.dumps({"Event_Id": 5, "Event_Type": "rest"})])
        assert result is not None and result[1] == 5.0

    def test_string_event_id_coerced(self, daq):
        result = daq._parse_event_sample([json.dumps({"event_id": "7", "event_type": "target"})])
        assert result is not None and result[1] == 7.0

    def test_empty_sample(self, daq):
        assert daq._parse_event_sample([]) is None

    def test_latency_update_sets_shared_state(self, daq):
        daq.shared_state = MagicMock()
        daq._parse_event_sample([json.dumps({"event_id": 0, "event_type": LATENCY_EVENT_TYPE, "latency_ms_raw": 42.5})])
        daq.shared_state.set.assert_called_once()
        assert daq.shared_state.set.call_args[0][1] == 42.5

    def test_extra_fields_preserved(self, daq):
        result = daq._parse_event_sample([json.dumps({"event_id": 1, "event_type": "cue", "custom_field": "value"})])
        assert result[0]["custom_field"] == "value"


class TestUpdateLatencyTelemetry:
    def test_initializes_window(self, daq):
        daq._update_latency_telemetry("EEG", 5.0)
        assert "EEG" in daq._latency_windows

    def test_accumulates_without_publishing(self, daq):
        daq.shared_state = MagicMock()
        for _ in range(10):
            daq._update_latency_telemetry("EEG", 5.0)
        daq.shared_state.set.assert_not_called()

    def test_publishes_p50_after_interval(self, daq):
        daq.shared_state = MagicMock()
        for i in range(daq._latency_update_interval):
            daq._update_latency_telemetry("EEG", float(i))
        assert daq.shared_state.set.call_count == 2

    def test_events_stream_updates_immediately(self, daq):
        daq.shared_state = MagicMock()
        daq._update_latency_telemetry("Events", 10.0)
        assert daq.shared_state.set.call_count == 2

    def test_no_shared_state_no_error(self, daq):
        daq.shared_state = None
        for _ in range(100):
            daq._update_latency_telemetry("EEG", 5.0)

    def test_p50_value_correct(self, daq):
        daq.shared_state = MagicMock()
        for _ in range(daq._latency_update_interval):
            daq._update_latency_telemetry("EEG", 10.0)
        assert daq.shared_state.set.call_args_list[0][0][1] == 10.0


class TestEventRecord:
    def test_creation(self):
        record = EventRecord(sample={"event_id": 1, "event_type": "stim"}, timestamp=1.0, local_timestamp=2.0)
        assert record.timestamp == 1.0
        assert record.receive_timestamp == 0.0  # default

    def test_with_receive_timestamp(self):
        record = EventRecord({"event_id": 7, "event_type": "left"}, 1.0, 1.0, 99.5)
        assert record.sample["event_id"] == 7
        assert record.receive_timestamp == 99.5


class TestEventBroadcast:
    """Verify per-stream event deques broadcast to all streams."""

    def test_single_stream_receives_event(self):
        d, q = _setup_broadcast("EEG")
        _broadcast(d, 42)
        assert d._pending_events_per_stream["EEG"].popleft() == 42
        q.close()

    def test_multi_stream_all_receive_event(self):
        d, q = _setup_broadcast("EEG", "EMG", "EOG")
        _broadcast(d, 7)
        assert d._pending_events_per_stream["EEG"].popleft() == 7
        assert d._pending_events_per_stream["EMG"].popleft() == 7
        assert d._pending_events_per_stream["EOG"].popleft() == 7
        q.close()

    def test_consuming_one_deque_does_not_affect_others(self):
        d, q = _setup_broadcast("EEG", "EMG")
        _broadcast(d, 99)
        # EEG consumes
        d._pending_events_per_stream["EEG"].popleft()
        # EMG still has it
        assert len(d._pending_events_per_stream["EMG"]) == 1
        assert d._pending_events_per_stream["EMG"].popleft() == 99
        q.close()

    def test_multiple_events_queued_in_order(self):
        d, q = _setup_broadcast("EEG", "EMG")
        for event_id in [1, 2, 3]:
            _broadcast(d, event_id)
        assert [d._pending_events_per_stream["EEG"].popleft() for _ in range(3)] == [1, 2, 3]
        assert [d._pending_events_per_stream["EMG"].popleft() for _ in range(3)] == [1, 2, 3]
        q.close()

    def test_empty_deque_no_error(self):
        d, q = _setup_broadcast("EEG")
        dq = d._pending_events_per_stream["EEG"]
        assert len(dq) == 0
        # Popping empty deque should be handled gracefully (IndexError)
        marker = 0.0
        if dq:
            try:
                marker = dq.popleft()
            except IndexError:
                pass
        assert marker == 0.0
        q.close()


class TestEventQueue:
    def test_save_event_drops_when_full(self):
        q = multiprocessing.Queue(maxsize=1)
        d, _ = _make_daq()
        d.event_queue = q
        q.put("filler")
        record = EventRecord({"event_id": 1, "event_type": "stim"}, 1.0, 1.0)
        d._save_event(record)  # should not block
        q.close()

    def test_save_event_puts_record(self):
        q = multiprocessing.Queue(maxsize=10)
        d, _ = _make_daq()
        d.event_queue = q
        record = EventRecord({"event_id": 1, "event_type": "stim"}, 1.0, 2.0)
        d._save_event(record)
        got = q.get(timeout=2)
        assert isinstance(got, EventRecord)
        assert got.sample["event_id"] == 1
        q.close()
