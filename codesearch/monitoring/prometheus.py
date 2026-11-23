"""Prometheus metrics for GET /metrics. The only module that imports prometheus_client.

Per-request numbers (counters, histograms) are updated as requests happen; the
rolling and index gauges are recomputed by refresh() each time Prometheus scrapes.
Labels only take values from small fixed sets (route templates, cache tiers):
never trace ids or question text, or every request would create a new series.
"""

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,  # Prometheus's Counter, not collections.Counter
    Gauge,
    Histogram,
    generate_latest,
)

from codesearch.ingest.manifest import Manifest
from codesearch.monitoring.drift import DriftTracker
from codesearch.monitoring.windows import RollingWindow, signals_from_answer
from codesearch.schemas import Answer, RequestSignals

CONTENT_TYPE = CONTENT_TYPE_LATEST  # what GET /metrics returns as its Content-Type
# Bucket edges in seconds: from cache hits (~10 ms) to slow LLM answers (~60 s).
SECONDS_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)


def tier_label(tier: str | None) -> str:
    """Prometheus labels can't be None: a freshly generated answer is "generated"."""
    return tier or "generated"


class Metrics:
    """All CodeSearch metrics, in their own registry (one per app, so tests don't clash)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry

        # Every HTTP request (middleware). path is the route template, never the raw URL.
        self.http_requests = Counter(
            "codesearch_http_requests", "HTTP requests", ["method", "path", "status"], registry=r
        )
        self.feedback = Counter("codesearch_feedback", "user feedback", ["vote"], registry=r)
        self.http_seconds = Histogram(
            "codesearch_http_request_seconds",
            "Time to response headers",
            ["method", "path"],
            buckets=SECONDS_BUCKETS,
            registry=r,
        )

        # Every finished answer (/v1/ask and /v1/ask/stream).
        self.answers = Counter(
            "codesearch_answers", "Answers served", ["cache_tier", "abstained"], registry=r
        )
        self.answer_seconds = Histogram(
            "codesearch_answer_seconds",
            "Time to a full answer",
            ["cache_tier"],
            buckets=SECONDS_BUCKETS,
            registry=r,
        )
        self.ttft_seconds = Histogram(
            "codesearch_ttft_seconds",
            "Time to first token (streamed)",
            buckets=SECONDS_BUCKETS,
            registry=r,
        )
        self.retrieval_seconds = Histogram(
            "codesearch_retrieval_seconds",
            "Retrieval + rerank time",
            buckets=SECONDS_BUCKETS,
            registry=r,
        )

        # Recomputed by refresh() at every scrape.
        self.window_answers = Gauge(
            "codesearch_window_answers", "Answers in the rolling window", registry=r
        )
        self.abstain_rate = Gauge(
            "codesearch_abstain_rate", "Share of recent answers that abstained", registry=r
        )
        self.cache_hit_rate = Gauge(
            "codesearch_cache_hit_rate", "Share of recent answers served from a cache", registry=r
        )
        self.groundedness_median = Gauge(
            "codesearch_groundedness_median", "Median groundedness of recent answers", registry=r
        )
        self.index_chunks = Gauge("codesearch_index_chunks", "Chunks in the live index", registry=r)
        self.index_info = Gauge(
            "codesearch_index_info",
            "The live index (always 1; the labels carry the information)",
            ["index_version", "corpus_sha"],
            registry=r,
        )
        self.reindex_runs = Gauge("codesearch_reindex_runs", "Rebuilds attempted", registry=r)
        self.reindex_last_ok = Gauge(
            "codesearch_reindex_last_success",
            "1 if the last rebuild worked, 0 if it failed",
            registry=r,
        )

        self.query_drift = Gauge(
            "codesearch_query_drift",
            "Cosine distance between recent questions and the golden set (0 = same topics)",
            registry=r,
        )

    def observe_http(self, method: str, path: str, status: int, seconds: float) -> None:
        """One HTTP request, from the middleware."""
        self.http_requests.labels(method, path, str(status)).inc()
        self.http_seconds.labels(method, path).observe(seconds)

    def observe(self, s: RequestSignals) -> None:
        """One finished answer."""
        tier = tier_label(s.cache_tier)
        self.answers.labels(tier, str(s.abstained).lower()).inc()
        self.answer_seconds.labels(tier).observe(s.latency_ms / 1000)
        if s.ttft_ms is not None:  # only streamed answers have one
            self.ttft_seconds.observe(s.ttft_ms / 1000)
        if s.retrieval_ms is not None:  # cache hits skip retrieval
            self.retrieval_seconds.observe(s.retrieval_ms / 1000)

    def observe_feedback(self, vote: int) -> None:
        self.feedback.labels("up" if vote > 0 else "down").inc()

    def refresh(
        self,
        window: RollingWindow,
        manifest: Manifest | None,
        reindex: dict[str, Any] | None,
        drift: float | None = None,
    ) -> None:
        """Set the gauges from current state; called by GET /metrics before rendering."""
        self.window_answers.set(len(window))
        for gauge, value in (
            (self.abstain_rate, window.rate(lambda s: s.abstained)),
            (self.cache_hit_rate, window.rate(lambda s: s.cache_tier is not None)),
            (self.groundedness_median, window.percentile("groundedness", 50)),
            (self.query_drift, drift),
        ):
            gauge.set(value if value is not None else float("nan"))  # NaN = "no data yet"

        self.index_info.clear()  # forget the previous index_version's series
        if manifest is not None:
            self.index_chunks.set(manifest.chunk_count)
            self.index_info.labels(manifest.index_version, manifest.corpus_sha).set(1)

        if reindex is not None:
            self.reindex_runs.set(reindex["runs"])
            if reindex["runs"]:  # "last success" means nothing before the first rebuild
                self.reindex_last_ok.set(0 if reindex["last_error"] else 1)

    def render(self) -> bytes:
        """The Prometheus text format, for GET /metrics."""
        return generate_latest(self.registry)


class Monitor:
    """One per app: the Prometheus metrics plus the rolling window.

    Routes call record(answer, question) once per finished answer; GET /metrics
    calls refresh(manifest, reindex) and metrics.render().
    """

    def __init__(
        self, window_size: int = 500, retrieval_mode: str = "", drift: DriftTracker | None = None
    ) -> None:
        self.metrics = Metrics()
        self.window = RollingWindow(window_size)
        self.retrieval_mode = retrieval_mode  # the same for every answer, so set once
        self.drift = drift  # None: drift can't be measured, the gauge shows NaN

    def record(self, answer: Answer, question: str) -> RequestSignals:
        """Turn one answer into signals, keep them in the window, count them in the metrics."""
        signals = signals_from_answer(answer, question, self.retrieval_mode)
        self.window.record(signals)  # feeds the rolling gauges (abstain rate, ...)
        self.metrics.observe(signals)  # counters and histograms, right away
        return signals  # returned so tests (and item 6's drift checks) can use it

    def refresh(self, manifest: Manifest | None, reindex: dict[str, Any] | None) -> None:
        """Compute drift, then set every gauge; called by GET /metrics."""
        drift = self.drift.distance(self.window) if self.drift is not None else None
        self.metrics.refresh(self.window, manifest, reindex, drift)
