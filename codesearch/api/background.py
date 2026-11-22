"""Re-indexing that runs beside requests, in the same process (no queues, no workers).

ReindexRunner runs one rebuild at a time in a worker thread; triggers that arrive
mid-build collapse into one more run. periodic() triggers it on a timer, and the
webhook (POST /v1/admin/reindex) triggers it on demand.
"""

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal

from codesearch.config import Settings
from codesearch.factory import build_embedder
from codesearch.index.handle import IndexHandle
from codesearch.ingest.build import IngestReport, build_index
from codesearch.interfaces import Embedder
from codesearch.log import get_logger

log = get_logger(__name__)

Build = Callable[[], IngestReport]  # one full rebuild; blocking, so it runs in a thread


class ReindexRunner:
    """Runs rebuilds one at a time; extra triggers during a build become one more run."""

    def __init__(self, build: Build) -> None:
        self._build = build  # the function that does one rebuild
        self._task: asyncio.Task[None] | None = None  # the background build loop, if started
        self._again = False  # another build was asked for mid-build
        # what status() reports:
        self.runs = 0
        self.last_reason: str | None = None
        self.last_report: IngestReport | None = None
        self.last_error: str | None = None
        self.finished_at: datetime | None = None

    @property
    def running(self) -> bool:
        """True while a build loop is in progress."""
        return self._task is not None and not self._task.done()

    async def trigger(self, reason: str) -> Literal["started", "queued"]:
        """Start a build now, or queue one more if a build is already running. Returns at once.

        No lock needed: asyncio only switches tasks at an await, and there is none
        between checking `running` and starting the task.
        """
        if self.running:
            self._again = True  # the running loop will see this and build once more
            return "queued"
        self._task = asyncio.create_task(self._run(reason))
        return "started"

    async def _run(self, reason: str) -> None:
        """The build loop: build in a thread, record the result, repeat if a trigger came in."""
        while True:
            self._again = False  # from here on, any new trigger means "build again after this"
            try:
                # build_index blocks (git, embedding): a worker thread keeps the API responsive.
                report = await asyncio.to_thread(self._build)
            except Exception as e:
                # Recorded, not raised: a failed build must not kill the runner,
                # and the live index is untouched (build_index only switches on success).
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("reindex.failed", reason=reason)
            else:
                self.last_report = report
                self.last_error = None
                log.info(
                    "reindex.done",
                    reason=reason,
                    added=report.added,
                    removed=report.removed,
                    index_version=report.index_version,
                )
            self.runs += 1
            self.last_reason = reason
            self.finished_at = datetime.now(UTC)

            if not self._again:
                return  # nobody asked during the build
            reason = "queued"  # someone did: build once more

    async def wait(self) -> None:
        """Return once no build is running (used by tests and the gate demo)."""
        while self._task is not None and not self._task.done():
            await asyncio.shield(self._task)  # waiting must never cancel the build

    async def stop(self) -> None:
        """Cancel the build loop on shutdown, without raising.

        A build already in its worker thread finishes in the background (threads
        can't be cancelled); its result is just not used.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def status(self) -> dict[str, Any]:
        """Plain JSON for GET /v1/admin/reindex."""
        return {
            "running": self.running,
            "runs": self.runs,
            "last_reason": self.last_reason,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "last_error": self.last_error,
            "last_report": asdict(self.last_report) if self.last_report else None,
        }


async def periodic(runner: ReindexRunner, interval_s: float) -> None:
    """Trigger a rebuild every interval_s seconds, until cancelled."""
    while True:
        await asyncio.sleep(interval_s)
        await runner.trigger("schedule")


def make_build(settings: Settings, handle: IndexHandle, embedder: Embedder | None = None) -> Build:
    """The real build: build_index on the latest upstream main if follow_upstream is on.

    The embedder is created on the first build, not at startup (loading a model is slow).
    """
    if settings.admin.follow_upstream:
        live = settings.with_overrides(index={"react_dev_sha": None})  # None: latest main
    else:
        live = settings

    def build() -> IngestReport:
        nonlocal embedder  # the outer variable: create it once, reuse it on every build
        if embedder is None:
            embedder = build_embedder(settings)
        return build_index(live, embedder, handle=handle)  # swaps into the live handle

    return build
