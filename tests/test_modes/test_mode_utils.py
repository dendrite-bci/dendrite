"""Tests for Buffer and utility functions in mode_utils.py."""

import numpy as np
import pytest

from dendrite.processing.modes.mode_utils import (
    Buffer,
    extract_event_code,
    extract_event_mapping,
    generate_label_mapping,
)

from .conftest import fill_buffer


# ---------------------------------------------------------------------------
# Buffer.__init__
# ---------------------------------------------------------------------------


class TestBufferInit:
    def test_stores_modalities_and_size(self, make_buffer):
        buf = make_buffer(modalities=["eeg", "emg"], buffer_size=100)
        assert buf.modalities == ["eeg", "emg"]
        assert buf.buffer_size == 100

    def test_creates_markers_deque(self, make_buffer):
        buf = make_buffer(buffer_size=50)
        assert "markers" in buf.buffers
        assert buf.buffers["markers"].maxlen == 50

    def test_empty_rings(self, make_buffer):
        buf = make_buffer()
        assert buf._rings == {}

    def test_counters_zero(self, make_buffer):
        buf = make_buffer()
        assert buf._write_pos == 0
        assert buf._sample_count == 0
        assert buf.samples_since_last_step == 0


# ---------------------------------------------------------------------------
# Buffer.add_sample
# ---------------------------------------------------------------------------


class TestBufferAddSample:
    def test_returns_true(self, make_buffer, make_sample):
        buf = make_buffer()
        assert buf.add_sample(make_sample()) is True

    def test_lazy_inits_ring(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=20)
        buf.add_sample(make_sample(n_channels=8))
        assert "eeg" in buf._rings
        assert buf._rings["eeg"].shape == (8, 20)

    def test_increments_counters(self, make_buffer, make_sample):
        buf = make_buffer()
        buf.add_sample(make_sample())
        assert buf._sample_count == 1
        assert buf.samples_since_last_step == 1
        buf.add_sample(make_sample())
        assert buf._sample_count == 2
        assert buf.samples_since_last_step == 2

    def test_writes_correct_position(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        buf.add_sample(make_sample(n_channels=2, value=7.0))
        np.testing.assert_array_equal(buf._rings["eeg"][:, 0], [7.0, 7.0])

    def test_wraps_write_pos(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        for _ in range(5):
            buf.add_sample(make_sample())
        assert buf._write_pos == 0  # wrapped

    def test_appends_marker(self, make_buffer, make_sample):
        buf = make_buffer()
        buf.add_sample(make_sample(marker=42))
        assert list(buf.buffers["markers"]) == [42]

    def test_skips_missing_modality(self, make_buffer):
        buf = make_buffer(modalities=["eeg", "emg"])
        # Sample has only eeg, no emg — should not crash
        sample = {"eeg": np.ones((2, 1), dtype=np.float32), "_receive_ns": 0}
        buf.add_sample(sample)
        assert "eeg" in buf._rings
        assert "emg" not in buf._rings

    def test_records_timestamp(self, make_buffer, make_sample):
        buf = make_buffer()
        buf.add_sample(make_sample(timestamp=123456))
        assert list(buf.timestamps) == [123456]

    def test_multi_modality(self, make_buffer):
        buf = make_buffer(modalities=["eeg", "emg"], buffer_size=10)
        sample = {
            "eeg": np.ones((4, 1), dtype=np.float32),
            "emg": np.ones((2, 1), dtype=np.float32) * 2,
            "_receive_ns": 0,
        }
        buf.add_sample(sample)
        assert "eeg" in buf._rings
        assert "emg" in buf._rings
        assert buf._rings["eeg"].shape == (4, 10)
        assert buf._rings["emg"].shape == (2, 10)


# ---------------------------------------------------------------------------
# Buffer._is_full
# ---------------------------------------------------------------------------


class TestBufferIsFull:
    def test_not_full_initially(self, make_buffer):
        buf = make_buffer(buffer_size=10)
        assert buf._is_full() is False

    def test_full_at_buffer_size(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        assert buf._is_full() is True

    def test_full_after_overflow(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 8)
        assert buf._is_full() is True


# ---------------------------------------------------------------------------
# Buffer.extract_window
# ---------------------------------------------------------------------------


class TestBufferExtractWindow:
    def test_none_when_not_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 3)
        assert buf.extract_window() is None

    def test_returns_dict_when_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        result = buf.extract_window()
        assert isinstance(result, dict)
        assert "eeg" in result

    def test_window_shape(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5, n_channels=4)
        result = buf.extract_window()
        assert result["eeg"].shape == (4, 5)

    def test_window_correct_data(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5, n_channels=1)
        result = buf.extract_window()
        # Values should be 0,1,2,3,4 (from fill_buffer)
        np.testing.assert_array_equal(result["eeg"][0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_resets_step_counter(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        assert buf.samples_since_last_step == 5
        buf.extract_window()
        assert buf.samples_since_last_step == 0

    def test_window_after_wrap(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        # Fill 8 samples: values 0..7, last 5 are 3,4,5,6,7
        fill_buffer(buf, make_sample, 8, n_channels=1)
        result = buf.extract_window()
        np.testing.assert_array_equal(result["eeg"][0], [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_multi_modality(self, make_buffer):
        buf = make_buffer(modalities=["eeg", "emg"], buffer_size=3)
        for i in range(3):
            sample = {
                "eeg": np.full((2, 1), float(i), dtype=np.float32),
                "emg": np.full((1, 1), float(i * 10), dtype=np.float32),
                "_receive_ns": 0,
            }
            buf.add_sample(sample)
        result = buf.extract_window()
        assert "eeg" in result
        assert "emg" in result


# ---------------------------------------------------------------------------
# Buffer.extract_epoch_at_event
# ---------------------------------------------------------------------------


class TestBufferExtractEpoch:
    def test_none_when_not_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 3)
        assert buf.extract_epoch_at_event(0, 5) is None

    def test_none_when_out_of_bounds(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 10)
        # start_offset so negative that epoch_start < 0
        result = buf.extract_epoch_at_event(-15, 5)
        assert result is None

    def test_none_when_exceeds_buffer(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 10)
        # epoch_end > buffer_size
        result = buf.extract_epoch_at_event(0, 20)
        assert result is None

    def test_correct_epoch_at_end(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 10, n_channels=1)
        # Event at end (default), extract 3 samples ending at event
        result = buf.extract_epoch_at_event(-2, 3)
        assert result is not None
        assert result["eeg"].shape == (1, 3)

    def test_epoch_with_offset(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 10, n_channels=1)
        # Pre-event window: start_offset=-3, length=3
        result = buf.extract_epoch_at_event(-3, 3)
        assert result is not None
        assert result["eeg"].shape == (1, 3)

    def test_epoch_shape(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=20)
        fill_buffer(buf, make_sample, 20, n_channels=8)
        result = buf.extract_epoch_at_event(-5, 5)
        assert result is not None
        assert result["eeg"].shape == (8, 5)

    def test_with_event_position_from_end(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 10, n_channels=1)
        # Event 3 samples before end
        result = buf.extract_epoch_at_event(-2, 3, event_position_from_end=3)
        assert result is not None
        assert result["eeg"].shape == (1, 3)


# ---------------------------------------------------------------------------
# Buffer.is_ready_for_step
# ---------------------------------------------------------------------------


class TestBufferIsReadyForStep:
    def test_not_ready_empty_modalities(self, logger):
        buf = Buffer(modalities=[], buffer_size=10, logger=logger)
        assert buf.is_ready_for_step(1) is False

    def test_not_ready_before_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 3)
        assert buf.is_ready_for_step(1) is False

    def test_not_ready_insufficient_steps(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        buf.samples_since_last_step = 2
        assert buf.is_ready_for_step(5) is False

    def test_ready_when_step_reached(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        # After fill, samples_since_last_step == 5
        assert buf.is_ready_for_step(5) is True


# ---------------------------------------------------------------------------
# Buffer.get_status
# ---------------------------------------------------------------------------


class TestBufferGetStatus:
    def test_status_empty(self, make_buffer):
        buf = make_buffer(buffer_size=10)
        status = buf.get_status()
        assert status == {
            "buffer_size": 10,
            "current_size": 0,
            "samples_since_last_step": 0,
        }

    def test_status_partially_filled(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 3)
        status = buf.get_status()
        assert status["current_size"] == 3

    def test_status_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 8)  # overflow
        status = buf.get_status()
        assert status["current_size"] == 5  # capped


# ---------------------------------------------------------------------------
# Buffer._extract_slice
# ---------------------------------------------------------------------------


class TestExtractSlice:
    def test_none_for_markers(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        assert buf._extract_slice("markers", 0, 5) is None

    def test_none_for_unknown_modality(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        assert buf._extract_slice("unknown", 0, 5) is None

    def test_none_when_not_full(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=10)
        fill_buffer(buf, make_sample, 3)
        assert buf._extract_slice("eeg", 0, 3) is None

    def test_contiguous_slice(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5, n_channels=1)
        result = buf._extract_slice("eeg", 1, 4)
        assert result is not None
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_wrapped_slice(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        # Fill 7: values 0..6, buffer has [5,6,2,3,4], write_pos=2
        # Oldest is at write_pos=2, so logical order: 2,3,4,5,6
        fill_buffer(buf, make_sample, 7, n_channels=1)
        result = buf._extract_slice("eeg", 0, 5)
        assert result is not None
        np.testing.assert_array_equal(result[0], [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_zero_length_none(self, make_buffer, make_sample):
        buf = make_buffer(buffer_size=5)
        fill_buffer(buf, make_sample, 5)
        assert buf._extract_slice("eeg", 2, 2) is None


# ---------------------------------------------------------------------------
# extract_event_code
# ---------------------------------------------------------------------------


class TestExtractEventCode:
    def test_none_markers(self):
        assert extract_event_code({}) == -1

    def test_missing_key(self):
        assert extract_event_code({"eeg": [1, 2]}) == -1

    def test_int_marker(self):
        assert extract_event_code({"markers": 5}) == 5

    def test_numpy_scalar(self):
        assert extract_event_code({"markers": np.array([3])}) == 3

    def test_numpy_2d(self):
        assert extract_event_code({"markers": np.array([[5]])}) == 5

    def test_invalid_type(self):
        assert extract_event_code({"markers": "not_a_number"}) == -1

    def test_zero_is_valid(self):
        assert extract_event_code({"markers": 0}) == 0


# ---------------------------------------------------------------------------
# extract_event_mapping
# ---------------------------------------------------------------------------


class TestExtractEventMapping:
    def test_string_keys_converted(self):
        config = {"event_mapping": {"1": "left", "2": "right"}}
        result = extract_event_mapping(config)
        assert result == {1: "left", 2: "right"}

    def test_empty(self):
        assert extract_event_mapping({"event_mapping": {}}) == {}

    def test_missing_key(self):
        assert extract_event_mapping({}) == {}


# ---------------------------------------------------------------------------
# generate_label_mapping
# ---------------------------------------------------------------------------


class TestGenerateLabelMapping:
    def test_empty_mapping(self):
        label, reverse, idx_to_code = generate_label_mapping({})
        assert label == {}
        assert reverse == {}
        assert idx_to_code == {}

    def test_two_classes(self):
        label, reverse, idx_to_code = generate_label_mapping({7: "left", 8: "right"})
        assert label == {"left": 0, "right": 1}
        assert reverse == {0: "left", 1: "right"}
        assert idx_to_code == {0: 7, 1: 8}

    def test_sorted_by_name(self):
        label, _, idx_to_code = generate_label_mapping({10: "c", 2: "a", 5: "b"})
        assert label == {"a": 0, "b": 1, "c": 2}
        assert idx_to_code == {0: 2, 1: 5, 2: 10}

    def test_grouped_events_contiguous_labels(self):
        label, reverse, idx_to_code = generate_label_mapping(
            {0: "rest", 1: "action", 2: "action"},
        )
        assert label == {"action": 0, "rest": 1}
        assert reverse == {0: "action", 1: "rest"}
        assert idx_to_code[0] == 1  # first event code for "action"
        assert idx_to_code[1] == 0  # event code for "rest"


