"""
QueueBridge

Pub/sub fan-out: producers call broadcast(), WebSocket handlers subscribe()
to receive items via asyncio.Queues. Slow consumers get frames dropped.
"""

import asyncio
from collections import deque
from typing import Any

from dendrite.utils.logger_central import get_logger


class QueueBridge:
    """Multi-channel pub/sub with optional history replay.

    Producers (visualization bridge, telemetry poller, ML service) call
    broadcast() to fan out items. Each WebSocket client subscribes to a
    channel and receives items via an asyncio.Queue.
    """

    _viz_drain_task_count: int = 0

    def __init__(self):
        self.logger = get_logger("QueueBridge")
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._broadcast_counts: dict[str, int] = {}
        self._drop_counts: dict[str, int] = {}
        self._history: dict[str, deque] = {}

    def enable_history(self, channel: str, maxlen: int = 500) -> None:
        """Enable history buffer for a channel. New subscribers receive a snapshot."""
        self._history[channel] = deque(maxlen=maxlen)
        self.logger.info(f"History enabled for '{channel}' (maxlen={maxlen})")

    def clear_history(self, channel: str | None = None) -> None:
        """Clear history buffer(s). If channel is None, clear all."""
        if channel:
            if channel in self._history:
                self._history[channel].clear()
        else:
            for buf in self._history.values():
                buf.clear()

    def _fan_out(self, channel: str, item: Any) -> None:
        """Distribute item to all subscribers, appending to history if enabled."""
        if channel in self._history:
            self._history[channel].append(item)
        for sub_q in self._subscribers.get(channel, set()):
            try:
                sub_q.put_nowait(item)
            except asyncio.QueueFull:
                self._drop_counts[channel] = self._drop_counts.get(channel, 0) + 1

    def subscribe(self, channel: str, maxsize: int = 100) -> tuple[asyncio.Queue, list]:
        """Subscribe to a channel.

        Returns (queue, history_snapshot). The queue receives live items.
        The history snapshot is a list of buffered items for the caller to
        send asynchronously — nothing is replayed into the queue.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        history = list(self._history.get(channel, []))

        self._subscribers.setdefault(channel, set()).add(q)
        self.logger.debug(
            f"Subscriber added to '{channel}' "
            f"(total: {len(self._subscribers[channel])}, history: {len(history)})"
        )
        return q, history

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        """Remove a subscriber from a channel."""
        subs = self._subscribers.get(channel)
        if subs:
            subs.discard(q)

    async def broadcast(self, channel: str, item: Any) -> None:
        """Broadcast an item to all subscribers of a channel."""
        self._broadcast_counts[channel] = self._broadcast_counts.get(channel, 0) + 1
        self._fan_out(channel, item)

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return per-channel broadcast/drop/subscriber counts."""
        channels = set(self._broadcast_counts) | set(self._drop_counts) | set(self._subscribers)
        stats = {}
        for ch in channels:
            stats[ch] = {
                "broadcasts": self._broadcast_counts.get(ch, 0),
                "drops": self._drop_counts.get(ch, 0),
                "subscribers": len(self._subscribers.get(ch, set())),
            }
        return stats

    async def shutdown(self) -> None:
        """Clear all subscribers and history."""
        self._subscribers.clear()
        self._history.clear()
        self.logger.info("QueueBridge shut down")
