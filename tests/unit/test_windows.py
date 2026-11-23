"""Tests for codesearch/monitoring/windows.py: the rolling window of recent answers."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

import pytest

from codesearch.monitoring.windows import RollingWindow, signals_from_answer
from codesearch.schemas import Answer, RequestSignals


def sig(**overrides: Any) -> RequestSignals:
    fields: dict[str, Any] = {
        "trace_id": "t",
        "timestamp": datetime.now(UTC),
        "query": "q",
        "retrieval_mode": "dense",
        "latency_ms": 100.0,
        "abstained": False,
    }
    return RequestSignals(**(fields | overrides))


def answer(**overrides: Any) -> Answer:
    fields: dict[str, Any] = {
        "text": "Use useState [1].",
        "citations": [],
        "abstained": False,
        "trace_id": "t-1",
        "index_version": "v",
        "prompt_version": "3",
        "timings": {"total_ms": 4200.0, "ttft_ms": 1800.0, "retrieval_ms": 40.0},
        "groundedness": 0.8,
        "confidence": 0.7,
    }
    return Answer(**(fields | overrides))


# =====================================================================
# signals_from_answer
# =====================================================================


def test_signals_copy_what_monitoring_needs_from_an_answer() -> None:
    s = signals_from_answer(answer(), "How do I add state?", "dense")
    assert (s.trace_id, s.query, s.retrieval_mode) == ("t-1", "How do I add state?", "dense")
    assert (s.latency_ms, s.ttft_ms, s.retrieval_ms) == (4200.0, 1800.0, 40.0)
    assert (s.groundedness, s.confidence, s.abstained, s.cache_tier) == (0.8, 0.7, False, None)


def test_signals_from_a_cache_hit_have_no_ttft_or_groundedness() -> None:
    hit = answer(cache_tier="exact", timings={"total_ms": 9.0}, groundedness=None, confidence=None)
    s = signals_from_answer(hit, "q", "dense")
    assert s.cache_tier == "exact"
    assert (s.ttft_ms, s.retrieval_ms, s.groundedness) == (None, None, None)


# =====================================================================
# RollingWindow
# =====================================================================


def test_the_window_keeps_only_the_newest_maxlen() -> None:
    window = RollingWindow(maxlen=3)
    for i in range(5):
        window.record(sig(trace_id=str(i)))
    assert len(window) == 3
    assert [s.trace_id for s in window.snapshot()] == ["2", "3", "4"]


@pytest.mark.parametrize("maxlen", [0, -1])
def test_maxlen_must_be_positive(maxlen: int) -> None:
    with pytest.raises(ValueError):
        RollingWindow(maxlen=maxlen)


def test_values_skip_missing_ones() -> None:
    window = RollingWindow()
    for g in (0.2, None, 0.6):
        window.record(sig(groundedness=g))
    assert window.values("groundedness") == [0.2, 0.6]
    assert window.mean("groundedness") == pytest.approx(0.4)


@pytest.mark.parametrize(("p", "expected"), [(50, 50.0), (95, 95.0), (100, 100.0), (0, 1.0)])
def test_percentile_is_nearest_rank(p: float, expected: float) -> None:
    window = RollingWindow()
    for ms in range(100, 0, -1):
        window.record(sig(latency_ms=float(ms)))
    assert window.percentile("latency_ms", p) == expected


def test_an_empty_window_has_no_statistics() -> None:
    window = RollingWindow()
    assert window.mean("latency_ms") is None
    assert window.percentile("latency_ms", 50) is None
    assert window.rate(lambda s: s.abstained) is None  # unknown, not 0%


def test_percentile_rejects_p_out_of_range() -> None:
    with pytest.raises(ValueError):
        RollingWindow().percentile("latency_ms", 101)


def test_rate_counts_the_share_that_matches() -> None:
    window = RollingWindow()
    for abstained in (True, False, False, True):
        window.record(sig(abstained=abstained))
    assert window.rate(lambda s: s.abstained) == 0.5


def test_records_from_many_threads_are_all_kept() -> None:
    window = RollingWindow(maxlen=10_000)

    def add() -> None:
        for _ in range(1000):
            window.record(sig())

    threads = [threading.Thread(target=add) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(window) == 4000
