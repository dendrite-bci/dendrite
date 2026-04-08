"""Tests for FanOutQueue multi-consumer distribution."""

import multiprocessing

import pytest

from dendrite.processing.modes.mode_utils import FanOutQueue


@pytest.fixture
def queues():
    """Three bounded multiprocessing queues for testing."""
    qs = [multiprocessing.Queue(maxsize=2) for _ in range(3)]
    yield qs
    for q in qs:
        q.close()
        q.join_thread()


class TestFanOutQueueDistribution:
    def test_put_distributes_to_all_queues(self, queues):
        fan = FanOutQueue(queues)
        fan.put("item1")

        for q in queues:
            assert q.get(timeout=1) == "item1"

    def test_put_single_queue(self):
        q = multiprocessing.Queue(maxsize=2)
        try:
            fan = FanOutQueue([q])
            fan.put("item1")
            assert q.get(timeout=1) == "item1"
        finally:
            q.close()


class TestFanOutQueueFullBehavior:
    def test_full_queue_drops_silently(self):
        q1 = multiprocessing.Queue(maxsize=1)
        q2 = multiprocessing.Queue(maxsize=2)
        try:
            fan = FanOutQueue([q1, q2])
            q1.put_nowait("fill")  # Fill q1

            fan.put("overflow")  # q1 full, q2 still works

            # q2 still got the item
            assert q2.get(timeout=1) == "overflow"
        finally:
            q1.close()
            q2.close()

    def test_all_full_does_not_raise(self):
        q1 = multiprocessing.Queue(maxsize=1)
        q2 = multiprocessing.Queue(maxsize=1)
        try:
            fan = FanOutQueue([q1, q2])
            q1.put_nowait("fill")
            q2.put_nowait("fill")

            fan.put("overflow")  # Both full — should not raise
        finally:
            q1.close()
            q2.close()
