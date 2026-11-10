"""In-process LRU cache used when Redis is unavailable.

What goes here:
    - ``LRUCache(max_items=1000, ttl_seconds)`` with the same get/set interface
      as RedisExactCache (OrderedDict + timestamps).

Tested by: tests/unit/test_memory_cache.py (eviction order, TTL expiry).
Phase: 1
"""

import time
from collections import OrderedDict
from collections.abc import Callable

from codesearch.schemas import Answer


class LRUCache:
    def __init__(
        self,
        max_items: int = 1000,
        ttl_seconds: float = 24 * 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items <= 0 or ttl_seconds <= 0:
            raise ValueError(
                f"max_items and ttl_seconds must be positive, got {max_items}, {ttl_seconds}"
            )

        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.lru: OrderedDict[str, tuple[float, Answer]] = OrderedDict()

    def get(self, key: str) -> Answer | None:
        if key not in self.lru:
            return None
        expires_at, answer = self.lru[key]
        if self.clock() >= expires_at:
            del self.lru[key]
            return None

        self.lru.move_to_end(key)
        return answer

    def set(self, key: str, answer: Answer) -> None:
        self.lru[key] = (self.clock() + self.ttl_seconds, answer)
        self.lru.move_to_end(key)

        if len(self.lru) > self.max_items:
            self.lru.popitem(last=False)

    def clear(self) -> None:
        self.lru = OrderedDict()

    def __len__(self) -> int:
        return len(self.lru)
