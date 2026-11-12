"""Combines the cache tiers and reports which one hit.

What goes here:
    - ``TieredCache(exact, semantic, fallback)``:
        * ``lookup(query, versions, top1_source) -> tuple[Answer | None, tier]``
          where tier is "exact", "semantic" or None
        * ``store(query, vector, answer, versions)``
        * on any Redis error: log once, switch to ``fallback`` (memory.py), and
          report ``degraded`` health until Redis responds again
    - ``hit_rate()`` per tier for monitoring.

Tested by: tests/unit/test_tiered_cache.py (exact first, semantic second,
Redis failure -> fallback without raising).
Phase: 1
"""

import time
from collections.abc import Callable
from typing import Any

from codesearch.cache.semantic import SemanticCache
from codesearch.errors import CacheUnavailableError
from codesearch.interfaces import Cache
from codesearch.schemas import Answer


class TieredCache:
    def __init__(
        self,
        exact: Cache,
        fallback: Cache,
        semantic: SemanticCache | None = None,
        *,
        retry_after_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retry_after_s <= 0:
            raise ValueError

        self.exact = exact
        self.fallback = fallback
        self.semantic = semantic
        self.retry_after_s = retry_after_s
        self.clock = clock
        self._retry_at: float | None = None
        self._counts: dict[str, int] = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    @property
    def degraded(self) -> bool:
        return self._retry_at is not None

    def _user_redis(self) -> bool:
        return self._retry_at is None or self.clock() >= self._retry_at

    def _mark_failed(self) -> None:
        """Redis just failed: use memory for the next retry_after_s seconds."""
        self._retry_at = self.clock() + self.retry_after_s

    def get(self, key: str) -> Answer | None:
        if self._user_redis():
            try:
                answer = self.exact.get(key)
                self._retry_at = None
            except CacheUnavailableError:
                self._mark_failed()
                answer = self.fallback.get(key)
        else:
            answer = self.fallback.get(key)

        if answer is not None:  # count only real hits
            self._counts["exact_hits"] += 1
        return answer

    def set(self, key: str, answer: Answer) -> None:
        if self._user_redis():
            try:
                self.exact.set(key, answer)
            except CacheUnavailableError:
                self._mark_failed()
                self.fallback.set(key, answer)
        else:
            self.fallback.set(key, answer)

    def lookup_similar(self, query: str, **kwargs: Any) -> Answer | None:
        answer = self.semantic.lookup(query, **kwargs) if self.semantic is not None else None
        self._counts["semantic_hits" if answer is not None else "misses"] += 1
        return answer

        return answer

    def add_similar(self, query: str, answer: Answer, **kwargs: Any) -> None:
        if self.semantic is None:
            return None

        self.semantic.add(query, answer, **kwargs)

    def stats(self) -> dict[str, int]:
        return self._counts
