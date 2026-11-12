"""Tests for codesearch/cache/tiered.py: one cache object for the pipeline.

TieredCache combines
  * the exact tier (Redis), with an in-memory LRU fallback when Redis is down, and
  * the semantic tier (paraphrases), which the pipeline checks after retrieval.

Redis is fakeredis on a FakeServer: ``server.connected = False`` takes it "down".
Time is a hand-moved clock, so the retry-after-a-pause logic is tested without sleeping.
"""

from __future__ import annotations

import fakeredis
import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.cache.exact import RedisExactCache
from codesearch.cache.keys import make_key
from codesearch.cache.memory import LRUCache
from codesearch.cache.semantic import SemanticCache
from codesearch.cache.tiered import TieredCache
from codesearch.schemas import Answer

V = {"index_version": "v1", "prompt_version": "1", "model": "gemma3:4b"}
KEY = make_key("How do I use useState?", **V)


def answer(text: str = "Use useState [1].") -> Answer:
    return Answer(
        text=text,
        citations=[],
        abstained=False,
        trace_id="t",
        index_version="v1",
        prompt_version="1",
    )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class OneVectorEmbedder:
    """Every text gets the same vector: similarity is always 1.0 (only top1 differs)."""

    model_name = "one"
    dim = 3

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self.embed_query(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make(server: fakeredis.FakeServer, clock: Clock, *, semantic: bool = True) -> TieredCache:
    redis_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    return TieredCache(
        exact=RedisExactCache(client=redis_client),
        fallback=LRUCache(clock=clock),
        semantic=SemanticCache(OneVectorEmbedder()) if semantic else None,
        retry_after_s=30.0,
        clock=clock,
    )


# =====================================================================
# Exact tier through Redis
# =====================================================================


def test_set_then_get_through_redis(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock)
    cache.set(KEY, answer())
    assert cache.get(KEY) == answer()
    assert cache.degraded is False


def test_answers_really_live_in_redis(server: fakeredis.FakeServer, clock: Clock) -> None:
    make(server, clock).set(KEY, answer())
    other_process = make(server, clock)  # a second worker sharing the same Redis
    assert other_process.get(KEY) == answer()


def test_get_miss(server: fakeredis.FakeServer, clock: Clock) -> None:
    assert make(server, clock).get(KEY) is None


# =====================================================================
# Redis down: fall back to memory, never raise
# =====================================================================


def test_redis_down_on_get_falls_back_without_raising(
    server: fakeredis.FakeServer, clock: Clock
) -> None:
    cache = make(server, clock)
    server.connected = False
    assert cache.get(KEY) is None
    assert cache.degraded is True


def test_while_degraded_set_and_get_use_memory(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock)
    server.connected = False
    cache.set(KEY, answer("from memory"))
    assert cache.get(KEY) == answer("from memory")


def test_redis_is_not_retried_on_every_request(server: fakeredis.FakeServer, clock: Clock) -> None:
    """Each failed Redis call can cost a 0.5 s timeout: wait retry_after_s before trying again.

    Proof: Redis comes back, but within the pause the cache still uses memory.
    """
    cache = make(server, clock)
    server.connected = False
    cache.get(KEY)  # fails -> degraded
    server.connected = True
    fakeredis.FakeRedis(server=server, decode_responses=True).set(
        KEY, answer("redis").model_dump_json()
    )
    clock.now += 10  # < 30 s
    assert cache.get(KEY) is None  # memory, not Redis
    assert cache.degraded is True


def test_redis_is_retried_after_the_pause_and_recovers(
    server: fakeredis.FakeServer, clock: Clock
) -> None:
    cache = make(server, clock)
    server.connected = False
    cache.get(KEY)
    server.connected = True
    cache_writer = fakeredis.FakeRedis(server=server, decode_responses=True)
    cache_writer.set(KEY, answer("redis").model_dump_json())
    clock.now += 31
    assert cache.get(KEY) == answer("redis")
    assert cache.degraded is False


def test_a_failed_retry_starts_a_new_pause(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock)
    server.connected = False
    cache.get(KEY)
    clock.now += 31
    cache.get(KEY)  # retry fails
    server.connected = True
    fakeredis.FakeRedis(server=server, decode_responses=True).set(KEY, answer().model_dump_json())
    clock.now += 10  # 10 s after the failed retry
    assert cache.get(KEY) is None
    assert cache.degraded is True


def test_redis_down_on_set_falls_back(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock)
    server.connected = False
    cache.set(KEY, answer())  # must not raise
    assert cache.degraded is True
    assert cache.get(KEY) == answer()  # served from memory


# =====================================================================
# Semantic tier
# =====================================================================


def test_similar_lookup_and_add(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock)
    cache.add_similar("How do I use useState?", answer(), top1="c1", **V)
    assert cache.lookup_similar("Using useState?", top1="c1", **V) == answer()
    assert cache.lookup_similar("Using useState?", top1="c2", **V) is None


def test_semantic_tier_can_be_switched_off(server: fakeredis.FakeServer, clock: Clock) -> None:
    cache = make(server, clock, semantic=False)
    cache.add_similar("How do I use useState?", answer(), top1="c1", **V)  # no-op
    assert cache.lookup_similar("How do I use useState?", top1="c1", **V) is None


def test_semantic_tier_still_works_while_redis_is_down(
    server: fakeredis.FakeServer, clock: Clock
) -> None:
    """The semantic tier is in-process, so a Redis outage doesn't affect it."""
    cache = make(server, clock)
    server.connected = False
    cache.add_similar("How do I use useState?", answer(), top1="c1", **V)
    assert cache.lookup_similar("How do I use useState?", top1="c1", **V) == answer()


# =====================================================================
# Counters for monitoring
# =====================================================================


def test_stats_count_hits_per_tier_and_misses(server: fakeredis.FakeServer, clock: Clock) -> None:
    """The pipeline asks get() first, then lookup_similar() after retrieval.

    An exact hit counts once; a request that misses both tiers counts one miss.
    """
    cache = make(server, clock)
    assert cache.stats() == {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    cache.set(KEY, answer())
    cache.get(KEY)  # exact hit

    cache.add_similar("q", answer(), top1="c1", **V)
    cache.get(make_key("q", **V))  # exact miss (not counted yet) ...
    cache.lookup_similar("q", top1="c1", **V)  # ... semantic hit

    cache.lookup_similar("q", top1="other", **V)  # semantic miss = the request missed

    assert cache.stats() == {"exact_hits": 1, "semantic_hits": 1, "misses": 1}


def test_a_semantic_lookup_with_the_tier_off_counts_a_miss(
    server: fakeredis.FakeServer, clock: Clock
) -> None:
    cache = make(server, clock, semantic=False)
    cache.lookup_similar("q", top1="c1", **V)
    assert cache.stats()["misses"] == 1


def test_retry_after_must_be_positive(server: fakeredis.FakeServer, clock: Clock) -> None:
    with pytest.raises(ValueError):
        TieredCache(
            exact=RedisExactCache(client=fakeredis.FakeRedis(server=server)),
            fallback=LRUCache(),
            retry_after_s=0,
        )
