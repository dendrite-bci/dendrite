"""Tests for WebSocket handler _relay — disconnect detection and subscriber cleanup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dendrite.web.ws.bridge import QueueBridge
from dendrite.web.ws.handlers import _relay


class FakeWebSocket:
    """Minimal WebSocket mock for testing _relay."""

    def __init__(self, pre_disconnected: bool = False):
        self._disconnect_event = asyncio.Event()
        self._sent: list = []
        if pre_disconnected:
            self._disconnect_event.set()

    async def receive(self):
        """Block until disconnect() is called."""
        await self._disconnect_event.wait()
        return {"type": "websocket.disconnect"}

    def disconnect(self):
        """Simulate client disconnect."""
        self._disconnect_event.set()

    async def send_json(self, data):
        self._sent.append(("json", data))

    async def send_bytes(self, data):
        self._sent.append(("bytes", data))


@pytest.fixture
def bridge():
    return QueueBridge()


async def test_relay_cleanup_on_disconnect(bridge: QueueBridge):
    """Subscriber is removed when client disconnects during idle channel."""
    ws = FakeWebSocket()

    relay_task = asyncio.create_task(_relay(ws, bridge, "test_ch"))

    # Give relay time to subscribe
    await asyncio.sleep(0.05)
    assert bridge.get_stats().get("test_ch", {}).get("subscribers", 0) == 1

    # Disconnect
    ws.disconnect()
    await asyncio.wait_for(relay_task, timeout=2.0)

    # Subscriber should be cleaned up
    subs = bridge._subscribers.get("test_ch", set())
    assert len(subs) == 0


async def test_relay_delivers_json_data(bridge: QueueBridge):
    """Data flows correctly through relay for JSON channel."""
    ws = FakeWebSocket()

    relay_task = asyncio.create_task(_relay(ws, bridge, "json_ch"))
    await asyncio.sleep(0.05)

    await bridge.broadcast("json_ch", {"msg": "hello"})
    await bridge.broadcast("json_ch", {"msg": "world"})
    await asyncio.sleep(0.05)

    ws.disconnect()
    await asyncio.wait_for(relay_task, timeout=2.0)

    assert ("json", {"msg": "hello"}) in ws._sent
    assert ("json", {"msg": "world"}) in ws._sent


async def test_relay_delivers_binary_data(bridge: QueueBridge):
    """Data flows correctly through relay for binary channel."""
    ws = FakeWebSocket()

    relay_task = asyncio.create_task(_relay(ws, bridge, "bin_ch", binary=True))
    await asyncio.sleep(0.05)

    await bridge.broadcast("bin_ch", {"val": 42})
    await asyncio.sleep(0.05)

    ws.disconnect()
    await asyncio.wait_for(relay_task, timeout=2.0)

    assert len(ws._sent) == 1
    assert ws._sent[0][0] == "bytes"


async def test_relay_no_subscriber_leak_on_multiple_cycles(bridge: QueueBridge):
    """Multiple connect/disconnect cycles don't leak subscribers."""
    for _ in range(5):
        ws = FakeWebSocket()
        task = asyncio.create_task(_relay(ws, bridge, "cycle_ch"))
        await asyncio.sleep(0.02)
        ws.disconnect()
        await asyncio.wait_for(task, timeout=2.0)

    subs = bridge._subscribers.get("cycle_ch", set())
    assert len(subs) == 0


async def test_relay_cleanup_on_immediate_disconnect(bridge: QueueBridge):
    """Relay cleans up even if disconnect happens before any data."""
    ws = FakeWebSocket(pre_disconnected=True)

    task = asyncio.create_task(_relay(ws, bridge, "fast_ch"))
    await asyncio.wait_for(task, timeout=2.0)

    subs = bridge._subscribers.get("fast_ch", set())
    assert len(subs) == 0
