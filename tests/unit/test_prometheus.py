"""Tests for codesearch/monitoring/prometheus.py: Metrics and Monitor."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from codesearch.ingest.manifest import ChunkParams, Manifest
from codesearch.monitoring.prometheus import Metrics, Monitor
from codesearch.monitoring.windows import RollingWindow
from codesearch.schemas import Answer, RequestSignals


def sig(**overrides: Any) -> RequestSignals:
    fields: dict[str, Any] = {
        "trace_id": "t",
        "timestamp": datetime.now(UTC),
        "query": "q",
        "retrieval_mode": "dense",
        "latency_ms": 4200.0,
        "abstained": False,
    }
    return RequestSignals(**(fields | overrides))


def value(metrics: Metrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


MANIFEST = Manifest(
    index_version="abc123def456",
    corpus_sha="a" * 40,
    embedding_model="hash",
    embedding_dim=64,
    chunk_params=ChunkParams(max_chars=1000, overlap=0),
    pipeline_version="1",
    chunk_count=4858,
    built_at=datetime.now(UTC),
)


# =====================================================================
# per-request metrics
# =====================================================================


def test_each_metrics_object_has_its_own_registry() -> None:
    """Two apps in one process (every test file) must not clash on metric names."""
    a, b = Metrics(), Metrics()
    a.observe(sig())
    assert value(a, "codesearch_answers_total", cache_tier="generated", abstained="false") == 1
    assert value(b, "codesearch_answers_total", cache_tier="generated", abstained="false") is None


def test_observe_counts_answers_by_tier_and_abstained() -> None:
    m = Metrics()
    m.observe(sig())
    m.observe(sig(cache_tier="exact", latency_ms=9.0))
    m.observe(sig(abstained=True))
    assert value(m, "codesearch_answers_total", cache_tier="generated", abstained="false") == 1
    assert value(m, "codesearch_answers_total", cache_tier="exact", abstained="false") == 1
    assert value(m, "codesearch_answers_total", cache_tier="generated", abstained="true") == 1


def test_observe_records_latencies_in_seconds() -> None:
    m = Metrics()
    m.observe(sig(latency_ms=4200.0, ttft_ms=1800.0, retrieval_ms=40.0))
    assert value(m, "codesearch_answer_seconds_sum", cache_tier="generated") == 4.2
    assert value(m, "codesearch_ttft_seconds_sum") == 1.8
    assert value(m, "codesearch_retrieval_seconds_sum") == 0.04
    # 1.8 s falls in the 2 s bucket, not the 1 s one
    assert value(m, "codesearch_ttft_seconds_bucket", le="1.0") == 0
    assert value(m, "codesearch_ttft_seconds_bucket", le="2.0") == 1


def test_a_cache_hit_adds_no_ttft_or_retrieval_sample() -> None:
    m = Metrics()
    m.observe(sig(cache_tier="exact", latency_ms=9.0))
    assert value(m, "codesearch_ttft_seconds_count") == 0
    assert value(m, "codesearch_retrieval_seconds_count") == 0


def test_observe_http_counts_by_route_template_and_status() -> None:
    m = Metrics()
    m.observe_http("GET", "/v1/explain/{trace_id}", 404, 0.002)
    m.observe_http("GET", "/v1/explain/{trace_id}", 200, 0.003)
    labels = {"method": "GET", "path": "/v1/explain/{trace_id}"}
    assert value(m, "codesearch_http_requests_total", status="404", **labels) == 1
    assert value(m, "codesearch_http_requests_total", status="200", **labels) == 1
    assert value(m, "codesearch_http_request_seconds_count", **labels) == 2


# =====================================================================
# refresh: gauges computed when Prometheus scrapes
# =====================================================================


def test_refresh_sets_rolling_rates_from_the_window() -> None:
    window = RollingWindow()
    for s in (
        sig(abstained=True),
        sig(cache_tier="exact"),
        sig(groundedness=0.2),
        sig(groundedness=0.9),
    ):
        window.record(s)
    m = Metrics()
    m.refresh(window, None, None)
    assert value(m, "codesearch_window_answers") == 4
    assert value(m, "codesearch_abstain_rate") == 0.25
    assert value(m, "codesearch_cache_hit_rate") == 0.25
    assert value(m, "codesearch_groundedness_median") == 0.2  # nearest rank of [0.2, 0.9]


def test_refresh_on_an_empty_window_reports_unknown_not_zero() -> None:
    m = Metrics()
    m.refresh(RollingWindow(), None, None)
    abstain = value(m, "codesearch_abstain_rate")
    assert abstain is not None and math.isnan(abstain)  # "no data", so alerts don't fire


def test_refresh_exposes_the_live_index() -> None:
    m = Metrics()
    m.refresh(RollingWindow(), MANIFEST, None)
    assert value(m, "codesearch_index_chunks") == 4858
    info = {"index_version": "abc123def456", "corpus_sha": "a" * 40}
    assert value(m, "codesearch_index_info", **info) == 1


def test_after_a_reindex_only_the_new_index_version_is_reported() -> None:
    m = Metrics()
    m.refresh(RollingWindow(), MANIFEST, None)
    m.refresh(RollingWindow(), MANIFEST.model_copy(update={"index_version": "new"}), None)
    old = {"index_version": "abc123def456", "corpus_sha": "a" * 40}
    assert value(m, "codesearch_index_info", **old) is None
    assert value(m, "codesearch_index_info", index_version="new", corpus_sha="a" * 40) == 1


def test_refresh_reports_the_last_reindex() -> None:
    m = Metrics()
    m.refresh(RollingWindow(), None, {"runs": 3, "last_error": None})
    assert value(m, "codesearch_reindex_runs") == 3
    assert value(m, "codesearch_reindex_last_success") == 1
    m.refresh(RollingWindow(), None, {"runs": 4, "last_error": "RuntimeError: boom"})
    assert value(m, "codesearch_reindex_last_success") == 0


def test_render_is_the_prometheus_text_format() -> None:
    m = Metrics()
    m.observe(sig())
    text = m.render().decode()
    assert "# TYPE codesearch_answers_total counter" in text
    assert 'codesearch_answers_total{abstained="false",cache_tier="generated"} 1.0' in text


# =====================================================================
# Monitor: one call per answer
# =====================================================================


def test_monitor_records_into_the_window_and_the_metrics() -> None:
    monitor = Monitor(window_size=10, retrieval_mode="dense")
    a = Answer(
        text="x",
        citations=[],
        abstained=False,
        trace_id="t-1",
        index_version="v",
        prompt_version="3",
        timings={"total_ms": 900.0},
    )
    signals = monitor.record(a, "How?")
    assert signals.query == "How?" and signals.retrieval_mode == "dense"
    assert len(monitor.window) == 1
    counted = value(
        monitor.metrics, "codesearch_answers_total", cache_tier="generated", abstained="false"
    )
    assert counted == 1
