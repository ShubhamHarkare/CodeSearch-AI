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
