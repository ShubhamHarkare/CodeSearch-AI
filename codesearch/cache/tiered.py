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
