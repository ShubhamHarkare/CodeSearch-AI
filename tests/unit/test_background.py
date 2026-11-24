"""Tests for codesearch/api/background.py: ReindexRunner and the periodic timer.

The build is a fake that runs in the runner's worker thread and can be held open
with a threading.Event, so each test controls exactly when a build finishes.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from codesearch.api.background import ReindexRunner, periodic
from codesearch.ingest.build import IngestReport

REPORT = IngestReport(
    corpus_sha="a" * 40,
    index_version="v2",
    pages=3,
    chunks=10,
    added=1,
    removed=1,
    unchanged=9,
    full_rebuild=False,
    duration_s=0.1,
)


class FakeBuild:
    """Counts calls and how many run at once; waits for `release` before finishing."""

    def __init__(self, *, hold: bool = False, fail: bool = False) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self._lock = threading.Lock()

    def __call__(self) -> IngestReport:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        self.release.wait(timeout=5)
        with self._lock:
            self.active -= 1
        if self.fail:
            raise RuntimeError("embedding service crashed")
        return REPORT


async def until(condition: threading.Event) -> None:
    """Wait (without blocking the loop) until a worker thread sets the event."""
    for _ in range(500):
        if condition.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


# =====================================================================
# trigger
# =====================================================================


async def test_trigger_returns_at_once_and_builds_in_the_background() -> None:
    build = FakeBuild(hold=True)
    runner = ReindexRunner(build)
    assert await runner.trigger("webhook") == "started"
    await until(build.started)
    assert runner.running  # still building, but trigger() already returned
    build.release.set()
    await runner.wait()
    assert not runner.running
    assert (runner.runs, runner.last_report, runner.last_reason) == (1, REPORT, "webhook")


async def test_the_build_never_blocks_the_event_loop() -> None:
    """build_index is slow and blocking (git, embedding): it must run in a thread."""
    build = FakeBuild(hold=True)
    runner = ReindexRunner(build)
    await runner.trigger("webhook")
    await until(build.started)
    start = time.perf_counter()
    await asyncio.sleep(0.01)  # would hang until release if the build held the loop
    assert time.perf_counter() - start < 1
    build.release.set()
    await runner.wait()


async def test_a_trigger_during_a_build_is_queued_and_runs_after_it() -> None:
    build = FakeBuild(hold=True)
    runner = ReindexRunner(build)
    await runner.trigger("webhook")
    await until(build.started)
    assert await runner.trigger("schedule") == "queued"
    build.release.set()
    await runner.wait()
    assert build.calls == 2  # the queued one ran, after the first finished
    assert build.max_active == 1  # two builds never overlap


async def test_many_triggers_during_a_build_collapse_into_one_more_run() -> None:
    build = FakeBuild(hold=True)
    runner = ReindexRunner(build)
    await runner.trigger("webhook")
    await until(build.started)
    for _ in range(5):
        assert await runner.trigger("webhook") == "queued"
    build.release.set()
    await runner.wait()
    assert build.calls == 2


async def test_a_trigger_after_a_build_finished_starts_a_new_one() -> None:
    runner = ReindexRunner(FakeBuild())
    await runner.trigger("webhook")
    await runner.wait()
    assert await runner.trigger("webhook") == "started"
    await runner.wait()
    assert runner.runs == 2


async def test_a_failed_build_is_recorded_and_the_runner_keeps_working() -> None:
    failing = FakeBuild(fail=True)
    runner = ReindexRunner(failing)
    await runner.trigger("webhook")
    await runner.wait()  # does not raise: the error is recorded instead
    assert runner.last_error is not None and "embedding service crashed" in runner.last_error
    assert runner.last_report is None

    failing.fail = False
    await runner.trigger("webhook")
    await runner.wait()
    assert runner.last_error is None
    assert runner.last_report == REPORT


# =====================================================================
# status, stop, periodic
# =====================================================================


async def test_status_is_plain_json() -> None:
    runner = ReindexRunner(FakeBuild())
    before = runner.status()
    assert before["running"] is False and before["runs"] == 0 and before["last_report"] is None
    await runner.trigger("webhook")
    await runner.wait()
    after = json.loads(json.dumps(runner.status()))  # serialisable as-is
    assert after["runs"] == 1
    assert after["last_reason"] == "webhook"
    assert after["last_report"]["added"] == 1
    assert after["finished_at"] is not None


async def test_stop_cancels_a_running_loop_without_raising() -> None:
    build = FakeBuild(hold=True)
    runner = ReindexRunner(build)
    await runner.trigger("webhook")
    await until(build.started)
    await runner.stop()
    assert not runner.running
    build.release.set()  # let the worker thread finish


async def test_stop_when_idle_is_a_no_op() -> None:
    await ReindexRunner(FakeBuild()).stop()


async def test_periodic_triggers_on_every_interval() -> None:
    runner = ReindexRunner(FakeBuild())
    timer = asyncio.create_task(periodic(runner, interval_s=0.02))
    await asyncio.sleep(0.15)
    timer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await timer
    await runner.wait()
    assert runner.runs >= 2
    assert runner.last_reason in ("schedule", "queued")
