"""Tests for SharedRingBuffer."""

import numpy as np
import pytest

from dendrite.data.shared_buffers import OverrunError, SharedRingBuffer, compute_max_samples


@pytest.fixture
def rb():
    buf = SharedRingBuffer.create("test_rb", n_channels=4, max_samples=100, sample_rate=500.0)
    yield buf
    buf.close()
    buf.unlink()


class TestSharedRingBuffer:
    def test_create_and_connect(self, rb):
        rb2 = SharedRingBuffer.connect("test_rb")
        assert rb2.n_channels == 4 and rb2.max_samples == 100 and rb2.write_pos == 0
        rb2.close()

    def test_write_and_read(self, rb):
        rb.write(np.array([1, 2, 3, 4], np.float32), 10.0, 5.0, 999)
        data, ts, local_ts, receive_ns, pos = rb.read_new(0)
        assert pos == 1
        np.testing.assert_array_almost_equal(data[0], [1, 2, 3, 4])
        np.testing.assert_array_almost_equal(ts, [10.0])
        np.testing.assert_array_almost_equal(local_ts, [5.0])
        assert receive_ns[0] == 999

    def test_write_2d(self, rb):
        rb.write(np.array([[1], [2], [3], [4]], np.float32), 1.0)
        data, _, _, _, _ = rb.read_new(0)
        np.testing.assert_array_almost_equal(data[0], [1, 2, 3, 4])

    def test_read_empty(self, rb):
        d, t, lt, ns, pos = rb.read_new(0)
        assert len(d) == 0 and pos == 0

    def test_incremental(self, rb):
        for i in range(10):
            rb.write(np.full(4, float(i), np.float32), float(i), float(i) + 0.1, i * 1000)
        _, _, _, _, p1 = rb.read_new(0)
        for i in range(10, 15):
            rb.write(np.full(4, float(i), np.float32), float(i), float(i) + 0.1, i * 1000)
        d2, _, _, ns2, p2 = rb.read_new(p1)
        assert p2 == 15 and len(d2) == 5
        assert ns2[0] == 10000

    def test_wrap_around(self, rb):
        for i in range(150):
            rb.write(np.full(4, float(i), np.float32), float(i), float(i), i)
        data, _, _, receive_ns, pos = rb.read_new(120)
        assert pos == 150 and len(data) == 30
        np.testing.assert_array_almost_equal(data[-1], [149] * 4)
        assert receive_ns[0] == 120

    def test_overrun(self, rb):
        for _ in range(150):
            rb.write(np.full(4, 0.0, np.float32), 0.0)
        with pytest.raises(OverrunError):
            rb.read_new(0)

    def test_two_readers(self, rb):
        for i in range(20):
            rb.write(np.full(4, float(i), np.float32), float(i))
        _, _, _, _, pa = rb.read_new(0)
        db, _, _, _, pb = rb.read_new(10)
        assert pa == 20 and pb == 20 and len(db) == 10

    def test_markers_in_last_column(self, rb):
        rb.write(np.array([1, 2, 3, 7.0], np.float32), 1.0)
        rb.write(np.array([1, 2, 3, 0.0], np.float32), 2.0)
        data, _, _, _, _ = rb.read_new(0)
        assert data[0, 3] == 7.0 and data[1, 3] == 0.0

    def test_receive_ns_latency_tracking(self, rb):
        """receive_ns enables internal latency: time.time_ns() - receive_ns."""
        import time
        now = time.time_ns()
        rb.write(np.full(4, 1.0, np.float32), 1.0, 1.0, now)
        _, _, _, receive_ns, _ = rb.read_new(0)
        latency_ms = (time.time_ns() - int(receive_ns[0])) / 1_000_000.0
        assert 0 <= latency_ms < 100  # should be <1ms but allow slack

    def test_local_ts_round_trip(self, rb):
        """local_timestamp (LSL local_clock) survives round-trip."""
        rb.write(np.full(4, 1.0, np.float32), 100.0, 99.5)
        _, ts, local_ts, _, _ = rb.read_new(0)
        assert ts[0] == pytest.approx(100.0)
        assert local_ts[0] == pytest.approx(99.5)

    def test_is_valid_when_live(self, rb):
        assert rb.is_valid is True

    def test_is_valid_after_close(self, rb):
        rb2 = SharedRingBuffer.connect("test_rb")
        rb2.close()
        assert rb2.is_valid is False


class TestHelpers:
    def test_compute_max_samples(self):
        assert compute_max_samples(500.0, 30) == 15000

    def test_orphan_cleanup(self):
        r1 = SharedRingBuffer.create("test_orphan", 2, 50, 100.0)
        r1.close()
        r2 = SharedRingBuffer.create("test_orphan", 2, 50, 100.0)
        assert r2.write_pos == 0
        r2.close()
        r2.unlink()
