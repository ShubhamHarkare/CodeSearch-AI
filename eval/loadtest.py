"""Load test for POST /v1/ask: throughput and latency on each serving path.

Three modes, one per question the numbers in the README have to answer:

    hit    One question, asked once to warm the cache, then N more times. Measures
           the API + exact-cache path: no retrieval, no LLM.
    miss   Golden in-scope questions with ``bypass_cache``, so every request runs
           retrieval, the LLM and the groundedness check. This is what a new
           question costs.
    burst  The same never-seen question sent C times at once. Without single-flight
           every copy calls the LLM; with it, one call serves all C, so the wall
           time drops to about one answer's latency.

Usage, with the API running (``make serve``):

    uv run python -m eval.loadtest hit                 # 2000 requests, 50 at a time
    uv run python -m eval.loadtest miss                # 20 requests, 4 at a time
    uv run python -m eval.loadtest burst               # 10 identical, all at once
    uv run python -m eval.loadtest hit -n 500 -c 20    # override the defaults

Each run prints a summary and writes ``docs/loadtest/<mode>.json``. That folder is
committed, so every figure the README quotes can be re-run and diffed.
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer

from eval.golden import load_golden

Mode = Literal["hit", "miss", "burst"]

GOLDEN = Path("eval/golden.jsonl")
OUT_DIR = Path("docs/loadtest")
HIT_QUESTION = "How do I update state based on the previous state?"
# (requests, concurrency) when not given on the command line
DEFAULTS: dict[str, tuple[int, int]] = {"hit": (2000, 50), "miss": (20, 4), "burst": (10, 10)}


@dataclass(frozen=True)
class Outcome:
    """One request as the client saw it."""

    latency_ms: float
    status: int  # HTTP status; 0 = no response (timeout or connection error)
    cache_tier: str | None = None  # "exact" | "semantic" | None (generated)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. 0.0 for no values."""
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(outcomes: list[Outcome], wall_s: float) -> dict[str, Any]:
    """Throughput and latency over the successful requests; errors counted apart."""
    ok = [o for o in outcomes if 200 <= o.status < 300]
    latencies = [o.latency_ms for o in ok]
    return {
        "requests": len(outcomes),
        "ok": len(ok),
        "errors": len(outcomes) - len(ok),
        "status_counts": {
            str(k): v for k, v in sorted(Counter(o.status for o in outcomes).items())
        },
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(ok) / wall_s, 2) if wall_s > 0 else 0.0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p90": round(percentile(latencies, 90), 1),
            "p99": round(percentile(latencies, 99), 1),
            "max": round(max(latencies, default=0.0), 1),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        },
        "cache_tiers": dict(Counter(o.cache_tier or "generated" for o in ok)),
    }


def payloads_for(
    mode: Mode, requests: int, concurrency: int, questions: list[str], nonce: str
) -> list[dict[str, Any]]:
    """The request bodies one run sends, in order."""
    if requests <= 0 or concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    if mode == "hit":
        return [{"question": HIT_QUESTION} for _ in range(requests)]
    if mode == "miss":
        if not questions:
            raise ValueError("miss mode needs at least one question")
        return [
            {"question": questions[i % len(questions)], "bypass_cache": True}
            for i in range(requests)
        ]
    # burst: one question nobody has asked before, sent `concurrency` times at once
    question = f"{HIT_QUESTION} (load test {nonce})"
    return [{"question": question} for _ in range(concurrency)]


async def send(client: httpx.AsyncClient, payload: dict[str, Any]) -> Outcome:
    """POST one question; never raises, so one failure can't stop the run."""
    start = time.perf_counter()
    try:
        response = await client.post("/v1/ask", json=payload)
    except httpx.HTTPError:
        return Outcome(latency_ms=(time.perf_counter() - start) * 1000, status=0)
    latency_ms = (time.perf_counter() - start) * 1000
    tier = None
    if response.is_success:
        try:
            tier = response.json().get("cache_tier")
        except ValueError:
            tier = None
    return Outcome(latency_ms=latency_ms, status=response.status_code, cache_tier=tier)


async def run_load(
    client: httpx.AsyncClient, payloads: list[dict[str, Any]], concurrency: int
) -> tuple[list[Outcome], float]:
    """Send every payload with at most ``concurrency`` in flight; return outcomes and wall time."""
    gate = asyncio.Semaphore(concurrency)

    async def one(payload: dict[str, Any]) -> Outcome:
        async with gate:
            return await send(client, payload)

    start = time.perf_counter()
    outcomes = await asyncio.gather(*(one(p) for p in payloads))
    return list(outcomes), time.perf_counter() - start


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


async def run(
    mode: Mode,
    *,
    url: str,
    requests: int,
    concurrency: int,
    timeout_s: float,
    questions: list[str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """One full load test against a running API. Returns the report."""
    nonce = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    payloads = payloads_for(mode, requests, concurrency, questions, nonce)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=url, timeout=timeout_s, limits=limits, transport=transport
    ) as client:
        versions: dict[str, Any] = {}
        try:
            ready = await client.get("/readyz")
            versions = ready.json().get("versions", {})
        except (httpx.HTTPError, ValueError):
            pass
        if mode == "hit":
            await send(client, payloads[0])  # warm the cache; not counted
        outcomes, wall_s = await run_load(client, payloads, concurrency)

    return {
        "mode": mode,
        "concurrency": concurrency,
        "url": url,
        "summary": summarize(outcomes, wall_s),
        "versions": versions,
        "git_sha": _git_sha(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_report(report: dict[str, Any], out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['mode']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def format_summary(report: dict[str, Any]) -> str:
    s = report["summary"]
    lat = s["latency_ms"]
    return (
        f"{report['mode']}: {s['ok']}/{s['requests']} ok, concurrency {report['concurrency']}\n"
        f"  throughput  {s['throughput_rps']} req/s over {s['wall_s']} s\n"
        f"  latency ms  p50 {lat['p50']}  p90 {lat['p90']}  p99 {lat['p99']}  max {lat['max']}\n"
        f"  cache tiers {s['cache_tiers']}   status {s['status_counts']}"
    )


def main(
    mode: Annotated[str, typer.Argument(help="hit | miss | burst")],
    requests: Annotated[int | None, typer.Option("--requests", "-n")] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c")] = None,
    url: Annotated[str, typer.Option()] = "http://localhost:8000",
    timeout_s: Annotated[float, typer.Option("--timeout")] = 300.0,
    out_dir: Annotated[Path, typer.Option()] = OUT_DIR,
) -> None:
    if mode not in DEFAULTS:
        raise typer.BadParameter(f"mode must be one of {sorted(DEFAULTS)}, got {mode!r}")
    default_n, default_c = DEFAULTS[mode]
    questions = [row.question for row in load_golden(GOLDEN) if row.expected == "answer"]
    report = asyncio.run(
        run(
            mode,  # type: ignore[arg-type]  # checked against DEFAULTS above
            url=url,
            requests=requests or default_n,
            concurrency=concurrency or default_c,
            timeout_s=timeout_s,
            questions=questions,
        )
    )
    path = write_report(report, out_dir)
    typer.echo(format_summary(report))
    typer.echo(f"report -> {path}")


if __name__ == "__main__":
    typer.run(main)
