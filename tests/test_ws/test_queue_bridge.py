"""Tests for QueueBridge — the core multiprocessing-to-asyncio bridge."""

import asyncio

import pytest

from dendrite.web.ws.bridge import QueueBridge


@pytest.fixture
def bridge():
    return QueueBridge()


async def test_subscribe_and_broadcast(bridge: QueueBridge):
    """Broadcast an item; the subscriber should receive it."""
    q, history = bridge.subscribe("test_channel")
    assert history == []
    await bridge.broadcast("test_channel", {"msg": "hello"})
    item = await asyncio.wait_for(q.get(), timeout=1.0)
    assert item == {"msg": "hello"}


async def test_multiple_subscribers(bridge: QueueBridge):
    """All subscribers on a channel receive the broadcast."""
    q1, _ = bridge.subscribe("ch")
    q2, _ = bridge.subscribe("ch")
    await bridge.broadcast("ch", "payload")
    assert await asyncio.wait_for(q1.get(), timeout=1.0) == "payload"
    assert await asyncio.wait_for(q2.get(), timeout=1.0) == "payload"


async def test_slow_consumer_drops(bridge: QueueBridge):
    """A full queue should not block broadcast to other subscribers."""
    slow, _ = bridge.subscribe("ch", maxsize=1)
    fast, _ = bridge.subscribe("ch", maxsize=100)

    # Fill the slow subscriber's queue
    await bridge.broadcast("ch", "first")

    # This should not block — slow consumer drops, fast still gets it
    await bridge.broadcast("ch", "second")

    # Fast got both
    assert await asyncio.wait_for(fast.get(), timeout=1.0) == "first"
    assert await asyncio.wait_for(fast.get(), timeout=1.0) == "second"

    # Slow only got the first (second was dropped)
    assert await asyncio.wait_for(slow.get(), timeout=1.0) == "first"
    assert slow.empty()


async def test_unsubscribe(bridge: QueueBridge):
    """After unsubscribe, the queue no longer receives broadcasts."""
    q, _ = bridge.subscribe("ch")
    bridge.unsubscribe("ch", q)
    await bridge.broadcast("ch", "after_unsub")
    assert q.empty()


async def test_broadcast_to_nonexistent_channel(bridge: QueueBridge):
    """Broadcasting to a channel with no subscribers is a no-op."""
    await bridge.broadcast("nonexistent", "data")  # Should not raise


async def test_shutdown_clears_state(bridge: QueueBridge):
    """Shutdown cancels tasks and clears subscribers."""
    bridge.subscribe("ch")
    await bridge.shutdown()
    assert len(bridge._subscribers) == 0


async def test_get_stats_counts_broadcasts(bridge: QueueBridge):
    """get_stats returns broadcast/drop/subscriber counts per channel."""
    bridge.subscribe("viz", maxsize=1)
    bridge.subscribe("viz", maxsize=1)
    await bridge.broadcast("viz", "a")
    await bridge.broadcast("viz", "b")  # second will drop for both (queue full)

    stats = bridge.get_stats()
    assert stats["viz"]["broadcasts"] == 2
    assert stats["viz"]["subscribers"] == 2
    assert stats["viz"]["drops"] >= 2  # At least 2 drops (both subs full on 2nd)


async def test_history_returned_as_snapshot(bridge: QueueBridge):
    """subscribe() returns full session history without replaying into the queue."""
    bridge.enable_history("ch")
    await bridge.broadcast("ch", "a")
    await bridge.broadcast("ch", "b")
    await bridge.broadcast("ch", "c")

    q, history = bridge.subscribe("ch")
    assert history == ["a", "b", "c"]
    # Queue should be empty — history is NOT replayed into it
    assert q.empty()


async def test_history_cleared_on_stop(bridge: QueueBridge):
    """clear_history resets the buffer; new subscribers get empty history."""
    bridge.enable_history("ch")
    await bridge.broadcast("ch", "old")
    bridge.clear_history("ch")

    _, history = bridge.subscribe("ch")
    assert history == []


async def test_get_stats_empty(bridge: QueueBridge):
    """get_stats returns empty dict when nothing has happened."""
    assert bridge.get_stats() == {}
