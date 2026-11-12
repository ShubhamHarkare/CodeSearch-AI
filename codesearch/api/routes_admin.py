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

from typing import Literal

from fastapi import APIRouter, Response

from codesearch.api.deps import HandleDep, PipelineDep, SettingsDep
from codesearch.api.models import HealthStatus, IndexInfo
from codesearch.cache.exact import RedisExactCache
from codesearch.cache.tiered import TieredCache
from codesearch.errors import CacheUnavailableError

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
