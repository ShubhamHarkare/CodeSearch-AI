"""Tests for codesearch/cache/singleflight.py: one call per key at a time.

Each test controls timing with an asyncio.Event "gate": the leader's work blocks
on the gate, so followers can arrive while it is still in flight. Nothing sleeps.
"""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest

from codesearch.cache.singleflight import SingleFlight


class Work:
    """A fake expensive call: counts calls, waits for the gate, then returns or raises."""

    def __init__(self, result: str = "answer", error: Exception | None = None) -> None:
        self.calls = 0
        self.gate = asyncio.Event()
        self.result = result
        self.error = error

    async def __call__(self) -> str:
        self.calls += 1
        await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result


async def settle() -> None:
    """Let every started task run up to its first await."""
    for _ in range(5):
        await asyncio.sleep(0)


# =====================================================================
# The basic contract
# =====================================================================


async def test_a_single_call_runs_the_work_and_is_not_shared() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    work.gate.set()
    assert await flight.do("k", work) == ("answer", False)
    assert work.calls == 1


async def test_concurrent_calls_with_the_same_key_run_the_work_once() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    tasks = [asyncio.create_task(flight.do("k", work)) for _ in range(10)]
    await settle()
    work.gate.set()
    results = await asyncio.gather(*tasks)

    assert work.calls == 1
    assert [r for r, _ in results] == ["answer"] * 10
    assert sorted(shared for _, shared in results) == [False] + [True] * 9


async def test_different_keys_do_not_wait_for_each_other() -> None:
    flight: SingleFlight[str] = SingleFlight()
    a, b = Work("A"), Work("B")
    task_a = asyncio.create_task(flight.do("a", a))
    task_b = asyncio.create_task(flight.do("b", b))
    await settle()
    b.gate.set()
    assert await task_b == ("B", False)  # finishes while "a" is still in flight
    a.gate.set()
    assert await task_a == ("A", False)
    assert (a.calls, b.calls) == (1, 1)


async def test_it_is_not_a_cache_a_later_call_runs_the_work_again() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    work.gate.set()
    await flight.do("k", work)
    assert await flight.do("k", work) == ("answer", False)
    assert work.calls == 2


async def test_len_counts_keys_in_flight() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    assert len(flight) == 0
    tasks = [asyncio.create_task(flight.do("k", work)) for _ in range(3)]
    await settle()
    assert len(flight) == 1  # three callers, one key
    work.gate.set()
    await asyncio.gather(*tasks)
    assert len(flight) == 0


# =====================================================================
# Failures
# =====================================================================


async def test_the_leaders_error_reaches_every_follower() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work(error=ValueError("ollama down"))
    tasks = [asyncio.create_task(flight.do("k", work)) for _ in range(4)]
    await settle()
    work.gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert work.calls == 1
    assert all(isinstance(r, ValueError) and str(r) == "ollama down" for r in results)
    assert len(flight) == 0


async def test_after_an_error_the_next_call_starts_fresh() -> None:
    flight: SingleFlight[str] = SingleFlight()
    failing = Work(error=RuntimeError("boom"))
    failing.gate.set()
    with pytest.raises(RuntimeError):
        await flight.do("k", failing)

    ok = Work("recovered")
    ok.gate.set()
    assert await flight.do("k", ok) == ("recovered", False)


async def test_a_failure_with_no_followers_logs_no_asyncio_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exception set on a Future nobody awaits makes asyncio log
    'Future exception was never retrieved' when the Future is collected."""
    flight: SingleFlight[str] = SingleFlight()
    work = Work(error=ValueError("x"))
    work.gate.set()
    with caplog.at_level(logging.ERROR, logger="asyncio"), pytest.raises(ValueError):
        await flight.do("k", work)
    gc.collect()
    await settle()
    assert "never retrieved" not in caplog.text


# =====================================================================
# Cancellation (a user closes the tab mid-answer)
# =====================================================================


async def test_cancelling_a_follower_does_not_cancel_the_leader() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    leader = asyncio.create_task(flight.do("k", work))
    await settle()
    quitter = asyncio.create_task(flight.do("k", work))
    stayer = asyncio.create_task(flight.do("k", work))
    await settle()

    quitter.cancel()
    await settle()
    work.gate.set()

    assert await leader == ("answer", False)
    assert await stayer == ("answer", True)
    with pytest.raises(asyncio.CancelledError):
        await quitter
    assert work.calls == 1


async def test_if_the_leader_is_cancelled_followers_do_not_hang() -> None:
    flight: SingleFlight[str] = SingleFlight()
    work = Work()
    leader = asyncio.create_task(flight.do("k", work))
    await settle()
    follower = asyncio.create_task(flight.do("k", work))
    await settle()

    leader.cancel()
    # The follower must finish (with CancelledError), not wait forever.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(follower, timeout=1)
    assert len(flight) == 0
