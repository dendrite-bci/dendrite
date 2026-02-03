"""
Queue utilities for Dendrite system.

Provides queue wrapper classes for multi-consumer data distribution.
"""

import logging
import multiprocessing
import queue
from typing import Any


class FanOutQueue:
    """
    Queue wrapper that distributes items to multiple consumer queues.

    Used for routing mode outputs to multiple destinations (metrics saver,
    visualization, monitoring) without blocking the producer.

    All queues use non-blocking puts with EAFP pattern for real-time priority.
    """

    def __init__(
        self,
        primary_queue: "multiprocessing.Queue[Any]",
        secondary_queue: "multiprocessing.Queue[Any]",
        monitoring_queue: "multiprocessing.Queue[Any] | None" = None,
    ) -> None:
        """Initialize with target queues.

        Args:
            primary_queue: Main queue for critical data (e.g., metrics saver)
            secondary_queue: Secondary queue for visualization
            monitoring_queue: Optional queue for GUI monitoring/telemetry
        """
        self.primary_queue = primary_queue
        self.secondary_queue = secondary_queue
        self.monitoring_queue = monitoring_queue

    def put(self, item, block=True, timeout=None):
        """
        Put item to all queues using non-blocking EAFP pattern.

        Real-time priority: drops items rather than blocking if queue is full.

        Args:
            item: The item to distribute
            block: Ignored (always non-blocking for real-time)
            timeout: Ignored (always non-blocking for real-time)
        """
        # Primary queue - critical path (metrics saving)
        try:
            self.primary_queue.put_nowait(item)
        except queue.Full:
            logging.warning("Primary queue full, dropping metrics item")

        # Secondary queue - visualization (best-effort)
        try:
            self.secondary_queue.put_nowait(item)
        except queue.Full:
            logging.debug("Visualization queue full, dropping item")

        # Monitoring queue - telemetry (optional, silent drop)
        if self.monitoring_queue is not None:
            try:
                self.monitoring_queue.put_nowait(item)
            except queue.Full:
                pass  # Silently drop monitoring data

    def empty(self):
        """Check if the primary queue is empty."""
        return self.primary_queue.empty()
