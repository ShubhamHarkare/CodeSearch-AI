"""Operations endpoints.

Endpoints:
    - ``GET /healthz``: process is up (always 200 if the app runs).
    - ``GET /readyz``: index loaded + Redis reachable (or "degraded" with the
      in-memory fallback); 503 when the index isn't ready.
    - ``GET /v1/index``: manifest (index_version, corpus SHA, chunk count,
      last update, changelog tail), prompt version, image SHA.
    - ``POST /v1/admin/reindex``: HMAC-verified (deps.verify_webhook);
      schedules build_index in the background and returns 202 immediately.
    - ``GET /v1/admin/gaps``: content-gap clusters.
    - ``GET /metrics``: Prometheus exposition.

Tested by: tests/integration/test_api.py.
Phase: 3-4
"""

import asyncio
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Response

from codesearch.api.deps import (
    HandleDep,
    MonitorDep,
    PipelineDep,
    ReindexerDep,
    SettingsDep,
    verify_webhook,
)
from codesearch.api.models import HealthStatus, IndexInfo
from codesearch.cache.exact import RedisExactCache
from codesearch.cache.tiered import TieredCache
from codesearch.errors import CacheUnavailableError
from codesearch.monitoring.gaps import content_gaps
from codesearch.monitoring.prometheus import CONTENT_TYPE

router = APIRouter(tags=["admin"])
CHANGELOG_TAIL = 5


def cache_status(cache: TieredCache | None) -> Literal["ok", "degraded", "disabled"]:
    # None -> "disabled"
    if cache is None:
        return "disabled"

    if isinstance(cache.exact, RedisExactCache):
        try:
            cache.exact.ping()
        except CacheUnavailableError:
            return "degraded"
    return "ok"


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    # always {"status": "ok"}: if this code runs, the process is alive
    return {"status": "ok"}


@router.get("/readyz", response_model=HealthStatus)
async def readyz(response: Response, pipeline: PipelineDep, settings: SettingsDep) -> HealthStatus:
    # handle = pipeline.handle; cache = cache_status(pipeline.cache)
    handle = pipeline.handle
    cache = cache_status(pipeline.cache)
    # versions = {"prompt_version": pipeline.answer_prompt.version}
    versions = {"prompt_version": pipeline.answer_prompt.version}
    # if handle.ready: add versions["index_version"] = handle.version
    #                  status = "degraded" if cache == "degraded" else "ok"
    # else:            status = "unavailable"; response.status_code = 503
    # components = {"index": "ok" if ready else "missing", "cache": cache}
    # return HealthStatus(..., image_sha=settings.image_sha)
    if handle.ready:
        versions["index_version"] = handle.version
        status: Literal["ok", "degraded", "unavailable"] = (
            "degraded" if cache == "degraded" else "ok"
        )
    else:
        status = "unavailable"
        response.status_code = 503

    index_state = "ok" if handle.ready else "missing"

    components = {"index": index_state, "cache": cache}

    return HealthStatus(
        status=status, components=components, versions=versions, image_sha=settings.image_sha
    )


@router.get("/v1/index", response_model=IndexInfo)
async def index_info(handle: HandleDep, pipeline: PipelineDep, settings: SettingsDep) -> IndexInfo:
    m = handle.snapshot().manifest
    return IndexInfo(
        index_version=m.index_version,
        corpus_sha=m.corpus_sha,  # react.dev commit the docs came from
        embedding_model=m.embedding_model,
        chunk_count=m.chunk_count,
        built_at=m.built_at,
        changelog=m.changelog[-CHANGELOG_TAIL:],  # last 5 builds only
        prompt_version=pipeline.answer_prompt.version,
        image_sha=settings.image_sha,
    )


@router.post("/v1/admin/reindex", status_code=202, dependencies=[Depends(verify_webhook)])
async def reindex(runner: ReindexerDep) -> dict[str, str]:
    """Start a rebuild, or queue one if a rebuild is running. Returns before it finishes.

    202 Accepted: the work has started, not finished; GET this path to follow it.
    verify_webhook runs first: a bad signature is a 401 and starts nothing.
    """
    return {"status": await runner.trigger("webhook")}


@router.get("/v1/admin/reindex")
async def reindex_status(runner: ReindexerDep) -> dict[str, Any]:
    """Is a rebuild running, and how did the last one go? Read-only, so not signed."""
    return runner.status()


@router.get("/metrics")
async def metrics(monitor: MonitorDep, handle: HandleDep, runner: ReindexerDep) -> Response:
    manifest = handle.snapshot().manifest if handle.ready else None
    monitor.refresh(manifest, runner.status())
    return Response(monitor.metrics.render(), media_type=CONTENT_TYPE)


@router.get("/v1/admin/gaps", dependencies=[Depends(verify_webhook)])
async def gaps(
    monitor: MonitorDep, pipeline: PipelineDep, k: int = Query(5, ge=1, le=10)
) -> list[dict[str, Any]]:
    """Topics users asked about that the app couldn't answer, biggest first.

    Signed like the reindex webhook: the clusters contain raw user questions.
    """
    if pipeline.embedder is None:  # lexical-only mode: nothing to cluster with
        return []
    questions = [s.query for s in monitor.window.snapshot() if s.abstained]
    # embedding + k-means are CPU-bound: run them off the event loop so /ask keeps serving
    clusters = await asyncio.to_thread(content_gaps, questions, pipeline.embedder, k=k)
    return [asdict(c) for c in clusters]
