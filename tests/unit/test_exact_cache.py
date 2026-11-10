"""Tests for codesearch/cache/exact.py: the Redis exact-match tier.

Uses fakeredis: an in-memory Redis that speaks the real protocol, so no server is
needed. A FakeServer with ``connected = False`` simulates Redis being down.
"""

from __future__ import annotations

import fakeredis
import pytest
import redis

from codesearch.cache.exact import STATS_HITS, STATS_MISSES, RedisExactCache
from codesearch.cache.keys import make_key
from codesearch.errors import CacheUnavailableError
from codesearch.interfaces import Cache
from codesearch.schemas import Answer, Citation

KEY = make_key("How do I use useState?", index_version="v", prompt_version="1", model="m")
OTHER = make_key("How do I use useReducer?", index_version="v", prompt_version="1", model="m")


def answer(text: str = "Use useState [1].") -> Answer:
    return Answer(
        text=text,
        citations=[
            Citation(
                number=1,
                chunk_id="c1",
                url="https://react.dev/reference/react/useState#usestate",
                title="useState",
            )
        ],
        abstained=False,
        timings={"total_ms": 1234.5},
        trace_id="abc",
        index_version="v",
        prompt_version="1",
    )


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def client(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def cache(client: fakeredis.FakeRedis) -> RedisExactCache:
    return RedisExactCache(client=client, ttl_hours=24)


# =====================================================================
# get / set
# =====================================================================


def test_set_then_get_round_trips_the_whole_answer(cache: RedisExactCache) -> None:
    cache.set(KEY, answer())
    assert cache.get(KEY) == answer()  # citations, timings, versions all survive


def test_unknown_key_is_a_miss(cache: RedisExactCache) -> None:
    assert cache.get(OTHER) is None


def test_stored_as_json_under_the_given_key(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    cache.set(KEY, answer())
    raw = client.get(KEY)
    assert raw is not None
    assert Answer.model_validate_json(raw) == answer()


def test_entries_expire_after_ttl_hours(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    cache.set(KEY, answer())
    assert 24 * 3600 - 5 <= client.ttl(KEY) <= 24 * 3600


def test_custom_ttl(client: fakeredis.FakeRedis) -> None:
    RedisExactCache(client=client, ttl_hours=2).set(KEY, answer())
    assert 2 * 3600 - 5 <= client.ttl(KEY) <= 2 * 3600


def test_set_overwrites_and_restarts_the_ttl(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    cache.set(KEY, answer("old"))
    client.expire(KEY, 10)
    cache.set(KEY, answer("new"))
    assert cache.get(KEY) == answer("new")
    assert client.ttl(KEY) > 10


def test_corrupt_entry_is_a_miss_and_is_removed(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    """Bad data (e.g. an old Answer format) must not crash a request."""
    client.set(KEY, "{not valid json")
    assert cache.get(KEY) is None
    assert client.exists(KEY) == 0


def test_satisfies_the_cache_protocol(cache: RedisExactCache) -> None:
    assert isinstance(cache, Cache)


def test_ttl_must_be_positive(client: fakeredis.FakeRedis) -> None:
    with pytest.raises(ValueError):
        RedisExactCache(client=client, ttl_hours=0)


# =====================================================================
# Hit / miss counters
# =====================================================================


def test_counters_track_hits_and_misses(cache: RedisExactCache) -> None:
    cache.set(KEY, answer())
    cache.get(KEY)
    cache.get(KEY)
    cache.get(OTHER)
    assert cache.stats() == {"hits": 2, "misses": 1}


def test_counters_start_at_zero(cache: RedisExactCache) -> None:
    assert cache.stats() == {"hits": 0, "misses": 0}


def test_counters_live_in_redis_so_every_worker_shares_them(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    cache.get(OTHER)
    assert STATS_MISSES == "codesearch:stats:misses"
    assert STATS_HITS == "codesearch:stats:hits"
    assert client.get(STATS_MISSES) == "1"


def test_a_corrupt_entry_counts_as_a_miss(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    client.set(KEY, "garbage")
    cache.get(KEY)
    assert cache.stats() == {"hits": 0, "misses": 1}


# =====================================================================
# clear
# =====================================================================


def test_clear_removes_cached_answers_and_reports_how_many(
    cache: RedisExactCache,
) -> None:
    cache.set(KEY, answer())
    cache.set(OTHER, answer())
    assert cache.clear() == 2
    assert cache.get(KEY) is None
    assert cache.get(OTHER) is None


def test_clear_leaves_other_redis_data_alone(
    cache: RedisExactCache, client: fakeredis.FakeRedis
) -> None:
    """Never FLUSHDB: the Redis may hold other apps' data, and the stats."""
    client.set("someone-else:key", "keep me")
    cache.set(KEY, answer())
    cache.get(KEY)  # bumps the hit counter
    cache.clear()
    assert client.get("someone-else:key") == "keep me"
    assert client.get(STATS_HITS) == "1"


def test_clear_on_an_empty_cache(cache: RedisExactCache) -> None:
    assert cache.clear() == 0


# =====================================================================
# Redis being down
# =====================================================================


def test_ping_returns_latency_in_ms(cache: RedisExactCache) -> None:
    latency = cache.ping()
    assert isinstance(latency, float)
    assert latency >= 0


@pytest.mark.parametrize("operation", ["get", "set", "ping", "stats", "clear"])
def test_every_operation_raises_cache_unavailable_when_redis_is_down(
    cache: RedisExactCache, server: fakeredis.FakeServer, operation: str
) -> None:
    """tiered.py catches CacheUnavailableError and falls back to the memory cache."""
    server.connected = False
    calls = {
        "get": lambda: cache.get(KEY),
        "set": lambda: cache.set(KEY, answer()),
        "ping": cache.ping,
        "stats": cache.stats,
        "clear": cache.clear,
    }
    with pytest.raises(CacheUnavailableError) as info:
        calls[operation]()
    assert isinstance(info.value.__cause__, redis.exceptions.RedisError)


def test_cache_unavailable_is_a_codesearch_error() -> None:
    from codesearch.errors import CodeSearchError

    assert issubclass(CacheUnavailableError, CodeSearchError)


# =====================================================================
# Building the real client
# =====================================================================


def test_builds_its_own_client_from_a_url_without_connecting() -> None:
    """Constructing must not touch the network; the first command connects."""
    cache = RedisExactCache(url="redis://localhost:1/0", ttl_hours=1)
    with pytest.raises(CacheUnavailableError):
        cache.get(KEY)  # nothing listens on port 1
