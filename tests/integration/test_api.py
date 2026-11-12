"""HTTP API tests: httpx.AsyncClient against create_app(settings, pipeline=...).

A real index (mini corpus, hash embedder) is built once; the LLM is a FakeLLM. The
client runs the app's lifespan, exactly as uvicorn would.

Phase 1 covers: POST /v1/ask, GET /healthz, GET /readyz, GET /v1/index, the
X-Trace-Id middleware and the error handlers. Stream, explain, feedback and the
reindex webhook are added in later phases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi import FastAPI

from codesearch.api.main import create_app
from codesearch.cache.exact import RedisExactCache
from codesearch.cache.memory import LRUCache
from codesearch.cache.tiered import TieredCache
from codesearch.config import Settings
from codesearch.errors import IndexNotReadyError, LLMUnavailableError
from codesearch.generation.llm import FakeLLM
from codesearch.generation.prompts import load_prompt
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.pipeline import AnswerPipeline
from codesearch.retrieval.abstain import ABSTAIN_MESSAGE
from codesearch.retrieval.hybrid import HybridRetriever
from codesearch.retrieval.rerank import NoopReranker
from tests.conftest import FakeReactDev

QUESTION = "How do I update state based on the previous state?"
ALWAYS_ANSWER = -2.0  # hash-embedder cosines are in [-1, 1]
ALWAYS_ABSTAIN = 2.0
GENERIC_500 = "An internal server error occurred."


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, IndexSnapshot]:
    root = tmp_path_factory.mktemp("api")
    repo = FakeReactDev(root / "upstream")
    settings = Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={
            "corpus_dir": root / "data",
            "index_dir": root / "index",
            "embedding_model": "hash",
            "react_dev_repo": repo.url,
            "react_dev_sha": repo.head,
        },
        retrieval={"mode": "dense", "dense_k": 10, "rerank_top_n": 5},
        llm={"provider": "fake"},
        cache={"enabled": False},
        image_sha="abc1234",
    )
    build_index(settings, HashEmbedder())
    snapshot = IndexSnapshot.open(root / "index")
    assert snapshot is not None
    return settings, snapshot


class DownLLM(FakeLLM):
    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise LLMUnavailableError("ollama is not running on localhost:11434")


class BrokenLLM(FakeLLM):
    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("secret internal detail")


def make_pipeline(
    built: tuple[Settings, IndexSnapshot],
    *,
    llm: Any = None,
    cache: Any = None,
    handle: IndexHandle | None = None,
    threshold: float = ALWAYS_ANSWER,
) -> tuple[Settings, AnswerPipeline]:
    settings, snapshot = built
    settings = settings.with_overrides(retrieval={"abstain_threshold": threshold})
    handle = handle if handle is not None else IndexHandle(snapshot)
    pipeline = AnswerPipeline(
        handle=handle,
        retriever=HybridRetriever(handle, HashEmbedder(), settings.retrieval),
        reranker=NoopReranker(),
        llm=llm if llm is not None else FakeLLM(),
        answer_prompt=load_prompt("answer"),
        condense_prompt=load_prompt("condense"),
        settings=settings,
        cache=cache,
    )
    return settings, pipeline


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the lifespan (startup/shutdown) and talk to the app in-process.

    raise_app_exceptions=False: an unhandled error comes back as the 500 response
    a real client would see, instead of being re-raised inside the test.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


Client = Callable[..., Any]


@pytest.fixture
def client(built: tuple[Settings, IndexSnapshot]) -> Client:
    """client(**make_pipeline kwargs) -> async context manager yielding an AsyncClient."""

    def _client(**kwargs: Any) -> Any:
        settings, pipeline = make_pipeline(built, **kwargs)
        return serve(create_app(settings, pipeline=pipeline))

    return _client


# =====================================================================
# GET /healthz
# =====================================================================


async def test_healthz_is_200_when_the_process_is_up(client: Client) -> None:
    async with client() as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_healthz_is_200_even_without_an_index(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200


# =====================================================================
# GET /readyz
# =====================================================================


async def test_readyz_is_200_once_the_index_is_loaded(client: Client, built: Any) -> None:
    _, snapshot = built
    async with client() as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"] == {"index": "ok", "cache": "disabled"}
    assert body["versions"]["index_version"] == snapshot.manifest.index_version
    assert body["versions"]["prompt_version"] == load_prompt("answer").version
    assert body["image_sha"] == "abc1234"


async def test_readyz_is_503_before_the_index_loads(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["components"]["index"] == "missing"
    assert "index_version" not in body["versions"]


async def test_readyz_with_a_healthy_cache(client: Client) -> None:
    redis_up = RedisExactCache(client=fakeredis.FakeRedis(decode_responses=True))
    async with client(cache=TieredCache(exact=redis_up, fallback=LRUCache())) as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    assert r.json()["components"]["cache"] == "ok"


async def test_readyz_is_still_200_when_redis_is_down(client: Client) -> None:
    """Redis down means the in-memory fallback is serving: degraded, not unready."""
    server = fakeredis.FakeServer()
    server.connected = False
    redis_down = RedisExactCache(client=fakeredis.FakeRedis(server=server, decode_responses=True))
    async with client(cache=TieredCache(exact=redis_down, fallback=LRUCache())) as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["components"]["cache"] == "degraded"


# =====================================================================
# POST /v1/ask
# =====================================================================


async def test_ask_returns_an_answer_with_citations(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/ask", json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == FakeLLM.DEFAULT_ANSWER
    assert body["abstained"] is False
    assert [cite["number"] for cite in body["citations"]] == [1, 2]
    assert set(body["citations"][0]) == {"number", "chunk_id", "url", "title"}
    assert body["cache_tier"] is None
    assert body["confidence"] is None and body["groundedness"] is None  # Phase 2
    assert "total_ms" in body["timings"]
    assert body["index_version"] and body["prompt_version"]


async def test_ask_trace_id_is_in_the_header_and_the_body(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/ask", json={"question": QUESTION})
    assert r.headers["X-Trace-Id"] == r.json()["trace_id"]
    assert len(r.json()["trace_id"]) >= 8


async def test_a_client_supplied_trace_id_is_reused(client: Client) -> None:
    async with client() as c:
        r = await c.post(
            "/v1/ask", json={"question": QUESTION}, headers={"X-Trace-Id": "ui-req-42"}
        )
    assert r.headers["X-Trace-Id"] == "ui-req-42"
    assert r.json()["trace_id"] == "ui-req-42"


@pytest.mark.parametrize("bad", ["x" * 200, "has spaces", "semi;colon", ""])
async def test_an_unsafe_trace_id_is_replaced(client: Client, bad: str) -> None:
    """Only [A-Za-z0-9_-]{1,64} is accepted; anything else could pollute the logs."""
    async with client() as c:
        r = await c.get("/healthz", headers={"X-Trace-Id": bad})
    assert r.headers["X-Trace-Id"] != bad
    assert len(r.headers["X-Trace-Id"]) >= 8


async def test_every_response_has_a_trace_id(client: Client) -> None:
    async with client() as c:
        for r in [await c.get("/healthz"), await c.get("/readyz"), await c.get("/nope")]:
            assert r.headers.get("X-Trace-Id")


async def test_ask_with_history_condenses_first(client: Client) -> None:
    llm = FakeLLM(responses=["How do I update an object in state?", "Copy it first [1]."])
    history = [
        {"role": "user", "content": "How do I use useState?"},
        {"role": "assistant", "content": "Call useState at the top level [1]."},
    ]
    async with client(llm=llm) as c:
        r = await c.post("/v1/ask", json={"question": "what about objects?", "history": history})
    assert r.status_code == 200
    assert r.json()["answer"] == "Copy it first [1]."
    assert len(llm.prompts) == 2  # condense + answer


async def test_ask_abstains_on_weak_retrieval(client: Client) -> None:
    llm = FakeLLM()
    async with client(llm=llm, threshold=ALWAYS_ABSTAIN) as c:
        r = await c.post("/v1/ask", json={"question": "What's the capital of France?"})
    assert r.status_code == 200  # abstaining is a normal answer, not an error
    body = r.json()
    assert body["abstained"] is True
    assert body["answer"] == ABSTAIN_MESSAGE
    assert body["citations"] == []
    assert llm.prompts == []


async def test_ask_uses_the_cache_and_bypass_cache_skips_it(client: Client) -> None:
    llm = FakeLLM()
    async with client(llm=llm, cache=TieredCache(exact=LRUCache(), fallback=LRUCache())) as c:
        first = (await c.post("/v1/ask", json={"question": QUESTION})).json()
        second = (await c.post("/v1/ask", json={"question": QUESTION})).json()
        third = (await c.post("/v1/ask", json={"question": QUESTION, "bypass_cache": True})).json()
    assert first["cache_tier"] is None
    assert second["cache_tier"] == "exact"
    assert third["cache_tier"] is None
    assert len(llm.prompts) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},  # whitespace only
        {"question": "x" * 1001},
        {"question": QUESTION, "history": [{"role": "user", "content": "hi"}] * 11},
        {"question": QUESTION, "history": [{"role": "system", "content": "hi"}]},
        {"question": QUESTION, "unknown_field": 1},
        {"question": 42},
    ],
)
async def test_ask_rejects_invalid_requests(client: Client, payload: dict[str, Any]) -> None:
    llm = FakeLLM()
    async with client(llm=llm) as c:
        r = await c.post("/v1/ask", json=payload)
    assert r.status_code == 422
    assert llm.prompts == []


async def test_ask_accepts_the_limits_exactly(client: Client) -> None:
    history = [{"role": "user", "content": "hi"}] * 10
    async with client() as c:
        r = await c.post("/v1/ask", json={"question": "x" * 1000, "history": history})
    assert r.status_code == 200


# =====================================================================
# Errors: CodeSearchError -> its status + public message; anything else -> 500
# =====================================================================


async def test_ask_before_the_index_loads_is_503(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.post("/v1/ask", json={"question": QUESTION})
    assert r.status_code == 503
    assert r.json() == {
        "error": IndexNotReadyError.public_message,
        "trace_id": r.headers["X-Trace-Id"],
    }


async def test_llm_outage_is_503_without_leaking_details(client: Client) -> None:
    async with client(llm=DownLLM()) as c:
        r = await c.post("/v1/ask", json={"question": QUESTION})
    assert r.status_code == 503
    assert r.json()["error"] == LLMUnavailableError.public_message
    assert "localhost" not in r.text


async def test_an_unexpected_error_is_a_generic_500(client: Client) -> None:
    async with client(llm=BrokenLLM()) as c:
        r = await c.post("/v1/ask", json={"question": QUESTION, "bypass_cache": True})
    assert r.status_code == 500
    assert r.json()["error"] == GENERIC_500
    assert "secret" not in r.text
    assert r.json()["trace_id"] == r.headers["X-Trace-Id"]


async def test_the_app_keeps_serving_after_a_500(client: Client) -> None:
    async with client(llm=BrokenLLM()) as c:
        await c.post("/v1/ask", json={"question": QUESTION})
        r = await c.get("/healthz")
    assert r.status_code == 200


# =====================================================================
# GET /v1/index
# =====================================================================


async def test_index_info(client: Client, built: Any) -> None:
    _, snapshot = built
    async with client() as c:
        r = await c.get("/v1/index")
    assert r.status_code == 200
    body = r.json()
    manifest = snapshot.manifest
    assert body["index_version"] == manifest.index_version
    assert body["corpus_sha"] == manifest.corpus_sha
    assert body["embedding_model"] == "hash"
    assert body["chunk_count"] == manifest.chunk_count > 0
    assert body["built_at"]
    assert body["prompt_version"] == load_prompt("answer").version
    assert body["image_sha"] == "abc1234"
    assert isinstance(body["changelog"], list)
    assert len(body["changelog"]) <= 5  # only the tail


async def test_index_info_is_503_without_an_index(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.get("/v1/index")
    assert r.status_code == 503
    assert r.json()["error"] == IndexNotReadyError.public_message


# =====================================================================
# create_app() without an injected pipeline (what `make serve` runs)
# =====================================================================


async def test_create_app_builds_the_pipeline_from_settings(built: Any) -> None:
    settings, _ = built
    app = create_app(settings.with_overrides(retrieval={"abstain_threshold": ALWAYS_ANSWER}))
    async with serve(app) as c:
        ready = await c.get("/readyz")
        r = await c.post("/v1/ask", json={"question": QUESTION})
    assert ready.status_code == 200
    assert r.status_code == 200
    assert r.json()["answer"] == FakeLLM.DEFAULT_ANSWER


async def test_create_app_with_no_index_built_yet(built: Any, tmp_path: Any) -> None:
    settings, _ = built
    app = create_app(settings.with_overrides(index={"index_dir": tmp_path / "empty"}))
    async with serve(app) as c:
        assert (await c.get("/healthz")).status_code == 200
        assert (await c.get("/readyz")).status_code == 503


def test_openapi_lists_the_phase_1_routes(built: Any) -> None:
    settings, _ = built
    paths = create_app(settings).openapi()["paths"]
    assert {"/v1/ask", "/healthz", "/readyz", "/v1/index"} <= set(paths)
