"""Tests for eval/loadtest.py. httpx.MockTransport plays the API, so no server runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eval.loadtest import (
    BURST_QUESTION,
    HIT_QUESTION,
    Outcome,
    format_summary,
    hits_until,
    payloads_for,
    percentile,
    run,
    run_load,
    send,
    send_stream,
    summarize,
    write_report,
)

BASE = "http://test"


def client_for(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler))


def answer(tier: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={"answer": "a", "cache_tier": tier})


# =====================================================================
# percentile
# =====================================================================


@pytest.mark.parametrize(("q", "expected"), [(0, 1), (50, 50), (90, 90), (99, 99), (100, 100)])
def test_percentile_nearest_rank(q: float, expected: float) -> None:
    assert percentile([float(v) for v in range(100, 0, -1)], q) == expected


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.0], 99) == 7.0


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 50) == 0.0


@pytest.mark.parametrize("q", [-1, 101])
def test_percentile_rejects_q_out_of_range(q: float) -> None:
    with pytest.raises(ValueError):
        percentile([1.0], q)


# =====================================================================
# summarize
# =====================================================================


def test_summarize_counts_errors_apart_from_latency() -> None:
    outcomes = [
        Outcome(10.0, 200, "exact"),
        Outcome(20.0, 200, None),
        Outcome(5000.0, 500),
        Outcome(9000.0, 0),
    ]
    s = summarize(outcomes, wall_s=2.0)
    assert (s["requests"], s["ok"], s["errors"]) == (4, 2, 2)
    assert s["status_counts"] == {"0": 1, "200": 2, "500": 1}
    assert s["throughput_rps"] == 1.0  # only successful requests count
    assert s["latency_ms"]["max"] == 20.0  # failures don't pollute latency
    assert s["cache_tiers"] == {"exact": 1, "generated": 1}


def test_summarize_with_no_time_or_no_successes_is_safe() -> None:
    s = summarize([Outcome(1.0, 503)], wall_s=0.0)
    assert s["throughput_rps"] == 0.0
    assert s["latency_ms"]["p50"] == 0.0
    assert s["latency_ms"]["mean"] == 0.0


# =====================================================================
# payloads_for
# =====================================================================


def test_hit_repeats_one_question_through_the_cache() -> None:
    payloads = payloads_for("hit", 5, 2, ["unused"], "n")
    assert payloads == [{"question": HIT_QUESTION}] * 5


def test_miss_cycles_questions_and_bypasses_the_cache() -> None:
    payloads = payloads_for("miss", 5, 2, ["a", "b"], "n")
    assert [p["question"] for p in payloads] == ["a", "b", "a", "b", "a"]
    assert all(p["bypass_cache"] is True for p in payloads)


def test_burst_sends_one_new_question_concurrency_times() -> None:
    payloads = payloads_for("burst", 999, 4, [], "20261120T1200")
    assert len(payloads) == 4  # burst size is the concurrency, not --requests
    assert len({p["question"] for p in payloads}) == 1
    assert "20261120T1200" in payloads[0]["question"]  # never seen before, so not cached
    assert "bypass_cache" not in payloads[0]  # single-flight lives in the cache path
    assert payloads[0]["question"].startswith(BURST_QUESTION)
    assert BURST_QUESTION != HIT_QUESTION  # earlier hit runs must not have cached it


def test_miss_needs_questions() -> None:
    with pytest.raises(ValueError):
        payloads_for("miss", 3, 1, [], "n")


@pytest.mark.parametrize(("n", "c"), [(0, 1), (1, 0), (-1, 1)])
def test_sizes_must_be_positive(n: int, c: int) -> None:
    with pytest.raises(ValueError):
        payloads_for("hit", n, c, [], "n")


# =====================================================================
# send
# =====================================================================


async def test_send_reads_the_cache_tier() -> None:
    async with client_for(lambda r: answer("exact")) as client:
        outcome = await send(client, {"question": "q"})
    assert (outcome.status, outcome.cache_tier) == (200, "exact")
    assert outcome.latency_ms >= 0


async def test_send_records_http_errors_without_raising() -> None:
    async with client_for(lambda r: httpx.Response(503, json={"error": "x"})) as client:
        outcome = await send(client, {"question": "q"})
    assert (outcome.status, outcome.cache_tier) == (503, None)


async def test_send_records_no_response_as_status_zero() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with client_for(refuse) as client:
        outcome = await send(client, {"question": "q"})
    assert outcome.status == 0


async def test_send_survives_a_non_json_success() -> None:
    async with client_for(lambda r: httpx.Response(200, text="not json")) as client:
        outcome = await send(client, {"question": "q"})
    assert (outcome.status, outcome.cache_tier) == (200, None)


# =====================================================================
# run_load
# =====================================================================


async def test_run_load_never_exceeds_the_concurrency() -> None:
    in_flight = peak = 0

    async def slow(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return answer()

    async with client_for(slow) as client:
        outcomes, wall_s = await run_load(client, [{"question": "q"}] * 12, concurrency=3)
    assert len(outcomes) == 12
    assert peak == 3
    assert wall_s > 0


# =====================================================================
# run + report
# =====================================================================


async def test_run_warms_the_cache_for_hit_mode() -> None:
    seen: list[dict[str, Any]] = []

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"versions": {"prompt_version": "3"}})
        seen.append(json.loads(request.content))
        return answer("exact" if len(seen) > 1 else None)

    report = await run(
        "hit",
        url=BASE,
        requests=4,
        concurrency=2,
        timeout_s=5,
        questions=[],
        transport=httpx.MockTransport(api),
    )
    assert len(seen) == 5  # 1 warm-up + 4 counted
    assert report["summary"]["requests"] == 4
    assert report["summary"]["cache_tiers"] == {"exact": 4}
    assert report["versions"] == {"prompt_version": "3"}
    assert report["mode"] == "hit"


async def test_run_works_when_readyz_is_down() -> None:
    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(503, text="down")
        return answer()

    report = await run(
        "miss",
        url=BASE,
        requests=2,
        concurrency=1,
        timeout_s=5,
        questions=["a"],
        transport=httpx.MockTransport(api),
    )
    assert report["versions"] == {}
    assert report["summary"]["ok"] == 2


def test_write_report_and_summary(tmp_path: Path) -> None:
    report = {
        "mode": "hit",
        "concurrency": 2,
        "summary": summarize([Outcome(12.0, 200, "exact")], wall_s=1.0),
    }
    path = write_report(report, tmp_path / "load")
    assert path.name == "hit.json"
    assert json.loads(path.read_text())["summary"]["ok"] == 1
    text = format_summary(report)
    assert "1/1 ok" in text and "p50 12.0" in text
    assert "during the misses" not in text


# =====================================================================
# mixed mode
# =====================================================================


def test_mixed_sends_misses_like_miss_mode() -> None:
    assert payloads_for("mixed", 3, 2, ["a"], "n") == payloads_for("miss", 3, 2, ["a"], "n")


async def test_hits_until_stops_when_told() -> None:
    stop = asyncio.Event()
    sent = 0

    async def api(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        if sent >= 6:
            stop.set()
        await asyncio.sleep(0)
        return answer("exact")

    async with client_for(api) as client:
        outcomes = await hits_until(client, {"question": "q"}, workers=2, stop=stop)
    assert 6 <= len(outcomes) <= 7  # each worker finishes the request it had in flight


async def test_mixed_reports_hits_and_misses_separately() -> None:
    async def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"versions": {}})
        body = json.loads(request.content)
        if body.get("bypass_cache"):
            await asyncio.sleep(0.03)  # a slow miss
            return answer(None)
        return answer("exact")  # a fast hit

    report = await run(
        "mixed",
        url=BASE,
        requests=4,
        concurrency=2,
        timeout_s=5,
        questions=["a", "b"],
        transport=httpx.MockTransport(api),
    )
    assert report["summary"]["requests"] == 4
    assert report["summary"]["cache_tiers"] == {"generated": 4}
    hits = report["hit_during_miss"]
    assert hits["ok"] > 0
    assert set(hits["cache_tiers"]) == {"exact"}
    assert "during the misses" in format_summary(report)


# =====================================================================
# burst mode: warm-up and the "already cached" guard
# =====================================================================


def burst_api(tier: str | None, seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"versions": {}})
        seen.append(json.loads(request.content))
        return answer(tier)

    return httpx.MockTransport(api)


async def test_burst_warms_up_with_an_uncached_different_question() -> None:
    seen: list[dict[str, Any]] = []
    report = await run(
        "burst",
        url=BASE,
        requests=1,
        concurrency=3,
        timeout_s=5,
        questions=["warm-up question"],
        transport=burst_api("shared", seen),
    )
    assert seen[0] == {"question": "warm-up question", "bypass_cache": True}
    assert len(seen) == 4  # warm-up + 3 counted
    assert report["summary"]["requests"] == 3
    assert "warning" not in report


async def test_burst_flags_a_run_served_from_the_cache() -> None:
    report = await run(
        "burst",
        url=BASE,
        requests=1,
        concurrency=2,
        timeout_s=5,
        questions=["w"],
        transport=burst_api("semantic", []),
    )
    assert "restart the API" in report["warning"]
    assert "WARNING" in format_summary(report)


# =====================================================================
# stream mode: client-side time to first token
# =====================================================================


def sse_body(*frames: tuple[str, Any]) -> bytes:
    return "".join(f"event: {k}\ndata: {json.dumps(d)}\n\n" for k, d in frames).encode()


GOOD_STREAM = sse_body(
    ("retrieval", [{"number": 1}]),
    ("token", "Use "),
    ("token", "useState [1]."),
    ("done", {"text": "Use useState [1].", "cache_tier": None, "timings": {}}),
)


class SlowStream(httpx.AsyncByteStream):
    """A response body whose first token arrives only after a delay."""

    def __init__(self, before: bytes, after: bytes, delay_s: float) -> None:
        self.before, self.after, self.delay_s = before, after, delay_s

    async def __aiter__(self) -> Any:
        yield self.before
        await asyncio.sleep(self.delay_s)
        yield self.after


async def test_send_stream_records_ttft_and_total() -> None:
    before = sse_body(("retrieval", [{"number": 1}]))
    after = GOOD_STREAM[len(before) :]

    def api(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ask/stream"
        return httpx.Response(200, stream=SlowStream(before, after, delay_s=0.05))

    async with client_for(api) as client:
        outcome = await send_stream(client, {"question": "q"})
    assert outcome.status == 200 and not outcome.failed
    assert outcome.ttft_ms is not None and outcome.ttft_ms >= 50  # waited for the first token
    assert outcome.latency_ms >= outcome.ttft_ms
    assert outcome.cache_tier is None


async def test_send_stream_reads_the_cache_tier_from_done() -> None:
    body = sse_body(("token", "x"), ("done", {"cache_tier": "exact"}))
    async with client_for(lambda r: httpx.Response(200, content=body)) as client:
        outcome = await send_stream(client, {"question": "q"})
    assert outcome.cache_tier == "exact"


async def test_an_error_event_counts_as_a_failed_stream() -> None:
    body = sse_body(("token", "x"), ("error", {"error": "down", "trace_id": "t"}))
    async with client_for(lambda r: httpx.Response(200, content=body)) as client:
        outcome = await send_stream(client, {"question": "q"})
    assert outcome.failed
    assert outcome.ttft_ms is not None  # tokens did arrive before the failure


async def test_a_stream_without_done_counts_as_failed() -> None:
    body = sse_body(("token", "x"))
    async with client_for(lambda r: httpx.Response(200, content=body)) as client:
        assert (await send_stream(client, {"question": "q"})).failed


async def test_send_stream_http_error_and_no_response() -> None:
    async with client_for(lambda r: httpx.Response(503, json={"error": "x"})) as client:
        assert (await send_stream(client, {"question": "q"})).status == 503

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with client_for(refuse) as client:
        assert (await send_stream(client, {"question": "q"})).status == 0


def test_summarize_reports_ttft_and_excludes_failed_streams() -> None:
    outcomes = [
        Outcome(6000.0, 200, None, ttft_ms=800.0),
        Outcome(7000.0, 200, None, ttft_ms=900.0),
        Outcome(500.0, 200, None, ttft_ms=100.0, failed=True),
    ]
    s = summarize(outcomes, wall_s=13.0)
    assert (s["ok"], s["errors"], s["failed_streams"]) == (2, 1, 1)
    assert s["ttft_ms"] == {"p50": 800.0, "p90": 900.0, "max": 900.0}
    assert "first token p50 800.0" in format_summary(
        {"mode": "stream", "concurrency": 1, "summary": s}
    )


def test_summarize_without_streams_has_no_ttft() -> None:
    s = summarize([Outcome(10.0, 200, "exact")], wall_s=1.0)
    assert "ttft_ms" not in s and "failed_streams" not in s


def test_stream_mode_payloads_bypass_the_cache() -> None:
    payloads = payloads_for("stream", 3, 1, ["a", "b"], "n")
    assert [p["question"] for p in payloads] == ["a", "b", "a"]
    assert all(p["bypass_cache"] for p in payloads)


async def test_stream_mode_warms_up_then_measures() -> None:
    paths: list[str] = []

    def api(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"versions": {}})
        return httpx.Response(200, content=GOOD_STREAM)

    report = await run(
        "stream",
        url=BASE,
        requests=3,
        concurrency=1,
        timeout_s=5,
        questions=["a"],
        transport=httpx.MockTransport(api),
    )
    assert paths.count("/v1/ask/stream") == 4  # warm-up + 3 measured
    assert report["summary"]["ok"] == 3
    assert "ttft_ms" in report["summary"]
