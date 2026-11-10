"""Tests for codesearch/cache/memory.py: the in-process LRU cache.

It is the fallback when Redis is down, and the cache for local runs without Redis.
Same get/set interface as the Redis tier. Time is injected (``clock``), so expiry is
tested without sleeping.
"""

from __future__ import annotations

import pytest

from codesearch.cache.memory import LRUCache
from codesearch.interfaces import Cache
from codesearch.schemas import Answer


def answer(text: str = "Use useState [1].") -> Answer:
    return Answer(
        text=text,
        citations=[],
        abstained=False,
        trace_id="t",
        index_version="v",
        prompt_version="1",
    )


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


# =====================================================================
# get / set
# =====================================================================


def test_set_then_get(clock: Clock) -> None:
    cache = LRUCache(clock=clock)
    cache.set("k", answer("A"))
    assert cache.get("k") == answer("A")


def test_unknown_key_is_a_miss(clock: Clock) -> None:
    assert LRUCache(clock=clock).get("nope") is None


def test_set_overwrites(clock: Clock) -> None:
    cache = LRUCache(clock=clock)
    cache.set("k", answer("old"))
    cache.set("k", answer("new"))
    assert cache.get("k") == answer("new")
    assert len(cache) == 1


def test_satisfies_the_cache_protocol() -> None:
    assert isinstance(LRUCache(), Cache)


# =====================================================================
# Size limit: least recently USED goes first
# =====================================================================


def test_oldest_entry_is_evicted_beyond_max_items(clock: Clock) -> None:
    cache = LRUCache(max_items=2, clock=clock)
    cache.set("a", answer("A"))
    cache.set("b", answer("B"))
    cache.set("c", answer("C"))
    assert cache.get("a") is None
    assert cache.get("b") == answer("B")
    assert cache.get("c") == answer("C")
    assert len(cache) == 2


def test_a_get_makes_an_entry_recent(clock: Clock) -> None:
    """LRU, not FIFO: reading "a" protects it, so "b" is evicted instead."""
    cache = LRUCache(max_items=2, clock=clock)
    cache.set("a", answer("A"))
    cache.set("b", answer("B"))
    cache.get("a")
    cache.set("c", answer("C"))
    assert cache.get("a") == answer("A")
    assert cache.get("b") is None


def test_overwriting_an_entry_makes_it_recent(clock: Clock) -> None:
    cache = LRUCache(max_items=2, clock=clock)
    cache.set("a", answer("A"))
    cache.set("b", answer("B"))
    cache.set("a", answer("A2"))
    cache.set("c", answer("C"))
    assert cache.get("a") == answer("A2")
    assert cache.get("b") is None


def test_a_miss_does_not_change_the_order(clock: Clock) -> None:
    cache = LRUCache(max_items=2, clock=clock)
    cache.set("a", answer("A"))
    cache.set("b", answer("B"))
    cache.get("zzz")
    cache.set("c", answer("C"))
    assert cache.get("a") is None


# =====================================================================
# Expiry (TTL)
# =====================================================================


def test_entry_is_served_until_it_expires(clock: Clock) -> None:
    cache = LRUCache(ttl_seconds=60, clock=clock)
    cache.set("k", answer())
    clock.now += 59.9
    assert cache.get("k") == answer()


def test_entry_expires_after_ttl(clock: Clock) -> None:
    cache = LRUCache(ttl_seconds=60, clock=clock)
    cache.set("k", answer())
    clock.now += 60
    assert cache.get("k") is None


def test_expired_entry_is_removed_when_read(clock: Clock) -> None:
    cache = LRUCache(ttl_seconds=60, clock=clock)
    cache.set("k", answer())
    clock.now += 61
    cache.get("k")
    assert len(cache) == 0


def test_reading_does_not_extend_the_ttl(clock: Clock) -> None:
    """The TTL protects against stale answers; popularity must not keep one alive."""
    cache = LRUCache(ttl_seconds=60, clock=clock)
    cache.set("k", answer())
    clock.now += 50
    assert cache.get("k") is not None
    clock.now += 20  # 70s after set
    assert cache.get("k") is None


def test_overwriting_restarts_the_ttl(clock: Clock) -> None:
    cache = LRUCache(ttl_seconds=60, clock=clock)
    cache.set("k", answer("old"))
    clock.now += 50
    cache.set("k", answer("new"))
    clock.now += 50  # 100s after the first set, 50s after the second
    assert cache.get("k") == answer("new")


# =====================================================================
# Housekeeping
# =====================================================================


def test_clear_empties_the_cache(clock: Clock) -> None:
    cache = LRUCache(clock=clock)
    cache.set("a", answer())
    cache.set("b", answer())
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_defaults() -> None:
    cache = LRUCache()
    assert (cache.max_items, cache.ttl_seconds) == (1000, 24 * 3600)


@pytest.mark.parametrize(("max_items", "ttl"), [(0, 60), (-1, 60), (10, 0), (10, -5)])
def test_limits_must_be_positive(max_items: int, ttl: float) -> None:
    with pytest.raises(ValueError):
        LRUCache(max_items=max_items, ttl_seconds=ttl)
