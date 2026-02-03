"""In-memory data cache for avoiding repeated disk reads."""

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class DataCache:
    """LRU cache for loaded data.

    Thread-safe in-memory cache to avoid repeated disk reads.
    Uses OrderedDict for LRU eviction.
    """

    def __init__(self, max_size: int = 100):
        """Initialize cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get_or_load(
        self,
        key: str,
        loader_fn: Callable[[], Any],
    ) -> Any:
        """Return cached data or load and cache.

        Args:
            key: Cache key (e.g., "dataset_subject_block")
            loader_fn: Function to call if key not in cache

        Returns:
            Cached or newly loaded data
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

            # Load and cache
            self._misses += 1
            data = loader_fn()
            self._cache[key] = data

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            return data

    def get(self, key: str) -> Any | None:
        """Get item from cache without loading.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item into cache.

        Args:
            key: Cache key
            value: Data to cache
        """
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)

            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def has(self, key: str) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached items."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> tuple[int, int, float]:
        """Return (hits, misses, hit_rate)."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return self._hits, self._misses, hit_rate


# Global cache instance
_global_cache: DataCache | None = None


def get_global_cache() -> DataCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache
