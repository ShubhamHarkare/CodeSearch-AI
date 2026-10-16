"""In-process LRU cache used when Redis is unavailable.

What goes here:
    - ``LRUCache(max_items=1000, ttl_seconds)`` with the same get/set interface
      as RedisExactCache (OrderedDict + timestamps).

Tested by: tests/unit/test_memory_cache.py (eviction order, TTL expiry).
Phase: 1
"""
