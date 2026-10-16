# =============================================================================
# TODO(v2): RETIRED IN v2. Replaced by codesearch/cache/:
#   keys.py (normalize + versioned key), exact.py (Redis), semantic.py
#   (near-duplicate questions), memory.py (fallback LRU), tiered.py (combines).
# =============================================================================

import hashlib
import json
import os
from datetime import timedelta
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()


class RedisCache:
    """
    Implementation of Caching using redis for chatbot responses
    Features:
    - Automatic TTL (Time to live) for cache entries
    - Query normalization for better hits
    - Comprehensive evaluation stratergies
    - Connection pooling for performance
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        ttl_hours: int = 24,
        key_prefix: str = "codesearch",
    ):
        """
        Initialize Redis cache connection

        Args:
            #? host: Redis server host
            #? port: Redis server port
            #? db: Redis database number
            #?ttl_hours: Titme to live for cache entries in hours
            #? key_prefix: Prefix for all cache_keys
        """
        try:
            redisUrl = os.getenv("REDIS_URL")
            if redisUrl:
                # TODO(v2): print() -> structured logging (codesearch/log.py).
                print("🔗 Connecting to Redis via URL...")
                self.pool = redis.ConnectionPool.from_url(
                    redisUrl, decode_responses=True, max_connections=10
                )
            else:
                host = host or os.getenv("REDIS_HOST", "localhost")
                port = port or int(os.getenv("REDIS_PORT", 6379))

                print(f"🔗 Connecting to Redis at {host}:{port}...")

                self.pool = redis.ConnectionPool(
                    host=host, port=port, db=db, decode_responses=True, max_connections=10
                )

            self.redis_client = redis.Redis(connection_pool=self.pool)
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")

        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            raise

        self.ttl = timedelta(hours=ttl_hours)
        self.key_prefix = key_prefix

        # TODO(v2): missing ':' separator gives keys like 'codesearchstats:hits'. v2 keys look like
        #     'codesearch:stats:hits'. Also rename stats_hits_miss -> stats_misses_key.
        self.stats_hits_key = f"{key_prefix}stats:hits"
        self.stats_hits_miss = f"{key_prefix}stats:misses"

    # TODO(v2): only lower() + strip(); the README says punctuation is removed too.
    #     v2 cache/keys.py normalize() must do what the README claims, with tests.
    def _normalize_query(self, query: str) -> str:
        """
        Normalizing the query for better cache hit rates
        """
        return query.lower().strip()

    def _generate_cache_key(self, query: str) -> str:
        """
        Generate cache key from query using MD5 hash

        Args:
            #? query: user's question
        Returns:
            #? Cache key
        """

        normalized = self._normalize_query(query)
        # TODO(v2): use sha256 and include index_version + prompt_version in the key, so a
        #     re-index or prompt change never serves stale answers.
        query_hash = hashlib.md5(normalized.encode()).hexdigest()

        return f"{self.key_prefix}query:{query_hash}"

    def get(self, query: str) -> dict[str, Any] | None:
        """
        Retrieve the cached response for a query
        Args:
            #?query: user's query
        Returns:
            #? Cached response or None if not found
        """
        cached_key = self._generate_cache_key(query)
        try:
            cached_data = self.redis_client.get(cached_key)

            if cached_data:
                self.redis_client.incr(self.stats_hits_key)

                return json.loads(cached_data)
            else:
                self.redis_client.incr(self.stats_hits_miss)
                return None

        except Exception as e:
            # TODO(v2): if Redis is down, the incr() below raises again inside this except block.
            print(f"Cache read error: {e}")
            self.redis_client.incr(self.stats_hits_miss)
            return None

    def set(self, query: str, response: dict[str, Any]) -> bool:
        """
        Setting the cache using TTL
        """
        cached_key = self._generate_cache_key(query)

        try:
            cached_data = json.dumps(response, ensure_ascii=False)

            self.redis_client.setex(cached_key, self.ttl, cached_data)
            return True
        except Exception as e:
            print(f"Cache write error: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Comprehensive statistics
        """
        try:
            hits = int(self.redis_client.get(self.stats_hits_key) or 0)
            misses = int(self.redis_client.get(self.stats_hits_miss) or 0)

            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0

            pattern = f"{self.key_prefix}query*"
            # TODO(v2): KEYS blocks Redis while it scans every key. Use scan_iter().
            total_keys = len(self.redis_client.keys(pattern))

            info = self.redis_client.info("memory")
            memory_mb = info["used_memory"] / (1024**2)

            return {
                "hits": hits,
                "misses": misses,
                "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "cached_queries": total_keys,
                "memory_usage_mb": round(memory_mb, 2),
            }
        except Exception as e:
            print(f"Stats error : {e}")
            return {"error": str(e)}

    def clear(self) -> dict[str, int]:
        """
        Clear all the cached queries but keep the stats
        """
        try:
            pattern = f"{self.key_prefix}query*"
            # TODO(v2): same: use scan_iter() and delete in batches.
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
            else:
                deleted = 0

            print(f"Cleared {deleted} cached queries")
            return {"deleted": deleted}

        except Exception as e:
            print(f"Clear cache error: {e}")
            return {"deleted": 0, "error": str(e)}

    # TODO(v2): flushdb() wipes the whole Redis DB, including other apps' data. Remove it.
    def clear_all(self) -> None:
        """
        Clears everything including the stats
        #! WARNING: this deleted ALL the data in the Redis Database
        """
        try:
            self.redis_client.flushdb()
            print("Cleared all the redis data")

        except Exception as e:
            print(f"Clear error: {e}")

    def reset_stats(self) -> None:
        """Reset hit/miss counters"""
        try:
            # TODO(v2): BUG: self.stats_misses_key does not exist (the attribute is stats_hits_miss),
            #     so this always raises AttributeError, which the except hides.
            self.redis_client.delete(self.stats_hits_key, self.stats_misses_key)
            print(" Reset cache statistics")
        except Exception as e:
            print(f" Reset stats error: {e}")

    def health_check(self) -> dict[str, Any]:
        """
        Check Redis connection health.

        Returns:
            Dict with connection status and latency
        """
        try:
            # TODO(v2): import inside a method; move to the top of the file.
            import time

            start = time.time()
            self.redis_client.ping()
            latency_ms = (time.time() - start) * 1000

            return {"status": "healthy", "latency_ms": round(latency_ms, 2), "connected": True}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "connected": False}

    def close(self) -> None:
        """Close Redis connection pool"""
        try:
            self.redis_client.close()
            print("Closed Redis connection")
        except Exception as e:
            print(f" Error closing Redis: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Redis Cache...\n")

    # Initialize cache
    cache = RedisCache(ttl_hours=24)

    # Test health check
    print("1. Health Check:")
    health = cache.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Latency: {health.get('latency_ms', 'N/A')}ms\n")

    # Test set/get
    print("2. Testing Cache Set/Get:")
    test_query = "How do I use useState?"
    test_response = {
        "answer": "useState is a React Hook...",
        "sources": ["react.dev/reference/react/useState"],
    }

    cache.set(test_query, test_response)
    print(f"Cached: '{test_query}'")

    # Retrieve
    cached = cache.get(test_query)
    if cached:
        print(f"Retrieved: {cached['answer'][:30]}...")

    # Test normalization (should hit cache)
    similar_query = "  HOW DO I USE USESTATE?  "
    cached2 = cache.get(similar_query)
    if cached2:
        print(f" Normalization works! Got cached result for: '{similar_query}'\n")

    # Test stats
    print("3. Cache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\nRedis cache is working correctly!")
