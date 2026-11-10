"""Redis exact-match tier. The only module that imports redis.

What goes here:
    - ``RedisExactCache(url, ttl_hours, prefix="codesearch")``:
        * ``get(key) -> Answer | None`` / ``set(key, answer)`` with TTL (JSON)
        * hit / miss counters at ``codesearch:stats:hits`` / ``:misses``
        * ``clear()`` using ``scan_iter`` (never KEYS, never FLUSHDB)
        * ``ping() -> float`` latency in ms, for /readyz
    - Short socket timeouts so a slow Redis can't stall requests.

What it must do:
    - Raise a clear error on connection failure so tiered.py can fall back.

Tested by: tests/unit/test_exact_cache.py using fakeredis.
Phase: 1
"""

import time

import redis
from pydantic import ValidationError

from codesearch.cache.keys import KEY_PREFIX
from codesearch.errors import CacheUnavailableError
from codesearch.schemas import Answer

STATS_HITS = "codesearch:stats:hits"
STATS_MISSES = "codesearch:stats:misses"


class RedisExactCache:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl_hours: float = 24,
        *,
        client: redis.Redis | None = None,
        timeout_s: float = 0.5,
    ) -> None:
        if ttl_hours <= 0:
            raise ValueError(f"ttl_hours must be positive, got {ttl_hours}")
        self.ttl_seconds = int(ttl_hours * 3600)

        # Tests inject a fakeredis client; production builds a real one. Building the
        # client doesn't connect (the first command does), so nothing can fail here.
        if client is not None:
            self.redis_client = client
        else:
            self.redis_client = redis.Redis.from_url(
                url,
                socket_timeout=timeout_s,  # a slow Redis fails fast instead of stalling
                socket_connect_timeout=timeout_s,
                decode_responses=True,
            )

    def get(self, key: str) -> Answer | None:
        try:
            raw = self.redis_client.get(key)
        except redis.exceptions.RedisError as error:
            raise CacheUnavailableError(f"Redis unavailable: {error}") from error
        if raw is None:
            try:
                self.redis_client.incr(STATS_MISSES)
            except redis.exceptions.RedisError as error:
                raise CacheUnavailableError(f"Redis unavailable: {error}") from error
            return None

        try:
            answer = Answer.model_validate_json(raw)
            try:
                self.redis_client.incr(STATS_HITS)
            except redis.exceptions.RedisError as error:
                raise CacheUnavailableError(f"Redis unavailable: {error}") from error
            return answer
        except ValidationError:
            # Corrupt or outdated entry: drop it. No answer is returned, so it's a miss.
            try:
                self.redis_client.delete(key)
                self.redis_client.incr(STATS_MISSES)
            except redis.exceptions.RedisError as error:
                raise CacheUnavailableError(f"Redis unavailable: {error}") from error
            return None

    def set(self, key: str, answer: Answer) -> None:
        try:
            self.redis_client.set(key, answer.model_dump_json(), ex=self.ttl_seconds)
        except redis.exceptions.RedisError as error:
            raise CacheUnavailableError(f"Redis unavailable: {error}") from error

    def stats(self) -> dict[str, int]:
        try:
            hits, misses = self.redis_client.mget(STATS_HITS, STATS_MISSES)
            # decode_responses=True returns strings ("2"); a counter never set is None
            return {"hits": int(hits or 0), "misses": int(misses or 0)}
        except redis.exceptions.RedisError as error:
            raise CacheUnavailableError(f"Redis unavailable: {error}") from error

    def clear(self) -> int:
        count = 0
        try:
            for key in self.redis_client.scan_iter(match=KEY_PREFIX + "*"):
                self.redis_client.delete(key)
                count += 1

            return count
        except redis.exceptions.RedisError as error:
            raise CacheUnavailableError(f"Redis unavailable: {error}") from error

    def ping(self) -> float:
        """Round-trip time to Redis in milliseconds (used by /readyz)."""
        try:
            started = time.perf_counter()
            self.redis_client.ping()
            return (time.perf_counter() - started) * 1000.0
        except redis.exceptions.RedisError as error:
            raise CacheUnavailableError(f"Redis unavailable: {error}") from error
