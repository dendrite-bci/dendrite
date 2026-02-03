"""
Cross-Process Shared State

Uses multiprocessing.Manager().dict() for cross-process state sharing.
Manager handles Windows spawn mode, process-safe reads/writes, and
automatic serialization of any picklable type.
"""

import logging
import time
from collections.abc import KeysView
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from typing import Any

logger = logging.getLogger(__name__)


class SharedState:
    """Cross-process shared state container using Manager().dict().

    Create in main process, pass to child processes. Any process can
    set/get values. Type is auto-detected (floats, dicts, lists all work).

    Example:
        # Main process
        state = SharedState()

        # DAQ process - set metrics and channel info
        state.set('eeg_latency_p50', 5.2)
        state.set('channel_info', {'labels': ['Fp1', 'Fp2'], 'types': ['EEG', 'EEG']})

        # Processor process - read channel info (with optional wait)
        info = state.wait_for('channel_info', timeout=60)

        # GUI process - read metrics
        latency = state.get('eeg_latency_p50')  # 5.2
    """

    def __init__(self):
        self._manager: SyncManager = Manager()
        self._data: dict[str, Any] = self._manager.dict()

    def __getstate__(self) -> dict[str, Any]:
        """Prepare for pickling - only pickle the DictProxy, not the manager."""
        return {"_data": self._data}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle - DictProxy maintains connection to manager."""
        self._data = state["_data"]
        self._manager = None  # Not available in child processes

    def __enter__(self) -> "SharedState":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False

    def set(self, key: str, value: float | int | dict | list | Any) -> None:
        """Set a value. Any serializable type supported."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key. Returns default if not set."""
        return self._data.get(key, default)

    def keys(self) -> KeysView[str]:
        """Return all keys."""
        return self._data.keys()

    def wait_for(self, key: str, timeout: float = 60.0, poll_interval: float = 0.1) -> Any | None:
        """Wait for a key to be set, with timeout.

        Args:
            key: Key to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds

        Returns:
            Value if found, None if timeout
        """
        start = time.monotonic()
        while (time.monotonic() - start) < timeout:
            result = self.get(key)
            if result is not None:
                return result
            time.sleep(poll_interval)
        return None

    def clear(self, key: str) -> None:
        """Remove a key from shared state."""
        try:
            del self._data[key]
        except KeyError:
            pass

    def cleanup(self) -> None:
        """Clean up resources. Call when done.

        Only the creator process should call this.
        """
        try:
            self._data.clear()
            if self._manager is not None:
                self._manager.shutdown()
        except (OSError, RuntimeError):
            pass
