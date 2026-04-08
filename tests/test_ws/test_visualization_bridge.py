"""Tests for visualization bridge serialization."""

import numpy as np
import pytest

from dendrite.web.ws.visualization_bridge import (
    _read_with_sleep,
    _serialize_mode_data,
)

from dendrite.data.shared_buffers import OverrunError, SharedRingBuffer


# --- _serialize_mode_data ---


def test_serialize_mode_data_numpy_to_binary():
    data = {"predictions": np.array([0.1, 0.9]), "label": "left"}
    result = _serialize_mode_data(data)
    assert "bytes" in result["predictions"]
    assert result["predictions"]["shape"] == [2]
    reconstructed = np.frombuffer(result["predictions"]["bytes"], dtype=np.float32)
    np.testing.assert_array_almost_equal(reconstructed, [0.1, 0.9], decimal=5)
    assert result["label"] == "left"


def test_serialize_mode_data_converts_numpy_scalars():
    data = {"class_index": np.int64(2), "confidence": np.float32(0.95)}
    result = _serialize_mode_data(data)
    assert result["class_index"] == 2
    assert isinstance(result["class_index"], int)
    assert isinstance(result["confidence"], float)


def test_serialize_mode_data_nested():
    data = {
        "outer": {
            "inner": np.array([1, 2, 3]),
            "list": [np.float64(1.0), "text"],
        }
    }
    result = _serialize_mode_data(data)
    assert "bytes" in result["outer"]["inner"]
    assert result["outer"]["list"] == [1.0, "text"]


def test_serialize_mode_data_passthrough():
    data = {"a": 1, "b": "hello", "c": [1, 2, 3], "d": None}
    assert _serialize_mode_data(data) == data


# --- _read_with_sleep overrun recovery ---


def test_read_with_sleep_overrun_skips_ahead():
    """OverrunError should return empty data and jump to current write_pos."""
    rb = SharedRingBuffer.create("test_viz_overrun", n_channels=2, max_samples=10, sample_rate=100)
    try:
        # Write enough to overrun a stale reader
        for i in range(15):
            rb.write(np.array([float(i), float(i)], np.float32), float(i))
        # Stale read_pos=0 would cause OverrunError, _read_with_sleep should recover
        data, ts, new_pos = _read_with_sleep(rb, 0)
        assert len(data) == 0
        assert new_pos == rb.write_pos
    finally:
        rb.close()
        rb.unlink()
