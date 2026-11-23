"""HTTP API tests: httpx.AsyncClient against create_app(settings, pipeline=...).

A real index (mini corpus, hash embedder) is built once; the LLM is a FakeLLM. The
client runs the app's lifespan, exactly as uvicorn would.

Phase 1 covers: POST /v1/ask, GET /healthz, GET /readyz, GET /v1/index, the
X-Trace-Id middleware and the error handlers. Stream, explain, feedback and the
reindex webhook are added in later phases.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi import FastAPI

from codesearch.api.background import ReindexRunner
from codesearch.api.main import create_app
from codesearch.cache.exact import RedisExactCache
from codesearch.cache.memory import LRUCache
from codesearch.cache.tiered import TieredCache
from codesearch.config import Settings
from codesearch.errors import (
    ExplainUnavailableError,
    IndexNotReadyError,
    InvalidWebhookError,
    LLMUnavailableError,
    TraceNotFoundError,
)
from codesearch.explain.trace import TraceStore
from codesearch.generation.llm import FakeLLM
from codesearch.generation.prompts import load_prompt
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import IngestReport, build_index
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
        explain={"trace_db": root / "traces.db", "groundedness": False},  # no NLI download
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
    trace_store: Any = None,
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
        trace_store=trace_store if trace_store is not None else TraceStore(),
        embedder=HashEmbedder(),
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
    assert {
        "/v1/ask",
        "/healthz",
        "/readyz",
        "/v1/index",
        "/v1/explain/{trace_id}",
        "/v1/explain/{trace_id}/deep",
    } <= set(paths)


# =====================================================================
# GET /v1/explain/{trace_id}
# =====================================================================


async def test_explain_returns_the_trace_of_an_answer(client: Client) -> None:
    async with client() as c:
        answer = (await c.post("/v1/ask", json={"question": QUESTION})).json()
        r = await c.get(f"/v1/explain/{answer['trace_id']}")
    assert r.status_code == 200
    trace = r.json()
    assert trace["trace_id"] == answer["trace_id"]
    assert trace["question"] == QUESTION
    assert trace["candidates"] and trace["used_chunk_ids"]
    first = trace["candidates"][0]
    assert {"chunk", "dense_score", "bm25_score", "rrf_score", "rerank_score"} <= set(first)
    assert first["chunk"]["url"].startswith("https://react.dev")
    assert trace["timings"] == answer["timings"]
    assert set(trace["matched_terms"]) == set(trace["used_chunk_ids"])


async def test_explain_shows_why_an_answer_abstained(client: Client) -> None:
    async with client(threshold=ALWAYS_ABSTAIN) as c:
        answer = (await c.post("/v1/ask", json={"question": "capital of France?"})).json()
        r = await c.get(f"/v1/explain/{answer['trace_id']}")
    assert r.status_code == 200
    assert r.json()["abstain_reason"].startswith("below_threshold")


async def test_explain_unknown_trace_is_404(client: Client) -> None:
    async with client() as c:
        r = await c.get("/v1/explain/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"] == TraceNotFoundError.public_message


async def test_explain_uses_the_client_supplied_trace_id(client: Client) -> None:
    async with client() as c:
        await c.post("/v1/ask", json={"question": QUESTION}, headers={"X-Trace-Id": "ui-7"})
        r = await c.get("/v1/explain/ui-7")
    assert r.status_code == 200


async def test_explain_without_a_trace_store_is_404(built: Any) -> None:
    settings, pipeline = make_pipeline(built)
    pipeline.trace_store = None
    async with serve(create_app(settings, pipeline=pipeline)) as c:
        r = await c.get("/v1/explain/anything")
    assert r.status_code == 404


# =====================================================================
# POST /v1/explain/{trace_id}/deep
# =====================================================================


async def test_deep_explain_returns_a_score_per_source(client: Client) -> None:
    async with client() as c:
        answer = (await c.post("/v1/ask", json={"question": QUESTION})).json()
        trace = (await c.get(f"/v1/explain/{answer['trace_id']}")).json()
        r = await c.post(f"/v1/explain/{answer['trace_id']}/deep")
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == answer["trace_id"]
    assert list(body["scores"]) == trace["used_chunk_ids"]


async def test_deep_explain_of_an_unknown_trace_is_404(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/explain/nope/deep")
    assert r.status_code == 404


async def test_deep_explain_of_an_abstained_answer_is_409(client: Client) -> None:
    async with client(threshold=ALWAYS_ABSTAIN) as c:
        answer = (await c.post("/v1/ask", json={"question": "capital of France?"})).json()
        r = await c.post(f"/v1/explain/{answer['trace_id']}/deep")
    assert r.status_code == 409
    assert r.json()["error"] == ExplainUnavailableError.public_message


# =====================================================================
# POST /v1/ask/stream (server-sent events)
# =====================================================================


def sse_frames(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into (event, data) pairs."""
    out: list[tuple[str, Any]] = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_stream_sends_retrieval_tokens_then_done(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/ask/stream", json={"question": QUESTION})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    frames = sse_frames(r.text)
    kinds = [kind for kind, _ in frames]
    assert kinds[0] == "retrieval" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}
    assert "".join(data for kind, data in frames if kind == "token") == FakeLLM.DEFAULT_ANSWER


async def test_stream_done_carries_the_answer_and_the_requests_trace_id(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/ask/stream", json={"question": QUESTION})
    done = sse_frames(r.text)[-1][1]
    assert done["trace_id"] == r.headers["x-trace-id"]
    assert [c["number"] for c in done["citations"]] == [1, 2]
    assert "ttft_ms" in done["timings"]


async def test_stream_rejects_an_empty_question_before_streaming(client: Client) -> None:
    async with client() as c:
        r = await c.post("/v1/ask/stream", json={"question": ""})
    assert r.status_code == 422  # validation runs before any event is sent


async def test_stream_turns_an_llm_outage_into_an_error_event(client: Client) -> None:
    async with client(llm=DownLLM()) as c:
        r = await c.post("/v1/ask/stream", json={"question": QUESTION})
    assert r.status_code == 200  # the status was sent before the failure
    kind, data = sse_frames(r.text)[-1]
    assert kind == "error"
    assert data == {
        "error": LLMUnavailableError.public_message,
        "trace_id": r.headers["x-trace-id"],
    }


async def test_stream_hides_unexpected_errors(client: Client) -> None:
    async with client(llm=BrokenLLM()) as c:
        r = await c.post("/v1/ask/stream", json={"question": QUESTION})
    kind, data = sse_frames(r.text)[-1]
    assert kind == "error"
    assert data["trace_id"] == r.headers["x-trace-id"]
    assert "secret internal detail" not in r.text


async def test_stream_without_an_index_reports_it_as_an_error_event(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.post("/v1/ask/stream", json={"question": QUESTION})
    assert sse_frames(r.text) == [
        ("error", {"error": IndexNotReadyError.public_message, "trace_id": r.headers["x-trace-id"]})
    ]


# =====================================================================
# POST / GET /v1/admin/reindex (signed webhook)
# =====================================================================

SECRET = "s3cret"
REPORT = IngestReport(
    corpus_sha="a" * 40,
    index_version="v2",
    pages=3,
    chunks=10,
    added=1,
    removed=0,
    unchanged=9,
    full_rebuild=False,
    duration_s=0.1,
)


def signed(body: bytes, secret: str = SECRET) -> dict[str, str]:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"}


class HeldBuild:
    """A fake build that waits until released, so a second trigger finds it running."""

    def __init__(self) -> None:
        self.calls = 0
        self.release = threading.Event()

    def __call__(self) -> IngestReport:
        self.calls += 1
        self.release.wait(timeout=5)
        return REPORT


def admin_client(built: Any, secret: str = SECRET) -> tuple[Any, ReindexRunner, HeldBuild]:
    settings, pipeline = make_pipeline(built)
    settings = settings.with_overrides(admin={"webhook_secret": secret})
    build = HeldBuild()
    runner = ReindexRunner(build)
    return serve(create_app(settings, pipeline=pipeline, reindexer=runner)), runner, build


async def test_a_signed_webhook_starts_a_reindex_and_returns_202(built: Any) -> None:
    client, runner, build = admin_client(built)
    body = b'{"ref": "refs/heads/main"}'
    async with client as c:
        first = await c.post("/v1/admin/reindex", content=body, headers=signed(body))
        second = await c.post("/v1/admin/reindex", content=body, headers=signed(body))
        build.release.set()
        await runner.wait()
    assert (first.status_code, first.json()) == (202, {"status": "started"})
    assert (second.status_code, second.json()) == (202, {"status": "queued"})
    assert build.calls == 2


async def test_a_bad_signature_is_401_and_starts_nothing(built: Any) -> None:
    client, runner, build = admin_client(built)
    body = b"{}"
    async with client as c:
        r = await c.post("/v1/admin/reindex", content=body, headers=signed(body, secret="guess"))
    assert r.status_code == 401
    assert r.json()["error"] == InvalidWebhookError.public_message
    assert build.calls == 0 and runner.runs == 0


async def test_a_missing_signature_is_401(built: Any) -> None:
    client, _, build = admin_client(built)
    async with client as c:
        r = await c.post("/v1/admin/reindex", json={})
    assert r.status_code == 401
    assert build.calls == 0


async def test_the_webhook_is_refused_when_no_secret_is_configured(built: Any) -> None:
    client, _, build = admin_client(built, secret="")
    body = b"{}"
    async with client as c:
        r = await c.post("/v1/admin/reindex", content=body, headers=signed(body, secret=""))
    assert r.status_code == 401
    assert "not enabled" in r.json()["error"]
    assert build.calls == 0


async def test_reindex_status_reports_the_last_run(built: Any) -> None:
    client, runner, build = admin_client(built)
    build.release.set()
    body = b"{}"
    async with client as c:
        idle = (await c.get("/v1/admin/reindex")).json()
        await c.post("/v1/admin/reindex", content=body, headers=signed(body))
        await runner.wait()
        done = (await c.get("/v1/admin/reindex")).json()
    assert idle["running"] is False and idle["runs"] == 0
    assert done["runs"] == 1 and done["last_reason"] == "webhook"
    assert done["last_report"]["index_version"] == "v2"


# =====================================================================
# GET /metrics (Prometheus)
# =====================================================================


def metric(text: str, line_start: str) -> float:
    """The value of the first exposition line starting with `line_start`."""
    for line in text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"no line starting with {line_start!r}")


async def test_metrics_is_prometheus_text(client: Client) -> None:
    async with client() as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE codesearch_answers_total counter" in r.text
    assert "codesearch_index_chunks" in r.text


async def test_metrics_count_answers_and_their_latency(client: Client) -> None:
    async with client() as c:
        await c.post("/v1/ask", json={"question": QUESTION})
        await c.post("/v1/ask/stream", json={"question": QUESTION})
        text = (await c.get("/metrics")).text
    assert metric(text, 'codesearch_answers_total{abstained="false",cache_tier="generated"}') == 2
    assert metric(text, "codesearch_ttft_seconds_count") == 1  # only the streamed one has a TTFT
    assert metric(text, "codesearch_window_answers") == 2


async def test_metrics_label_routes_by_template_not_by_value(client: Client) -> None:
    """One series per route, not one per trace id: label cardinality stays bounded."""
    async with client() as c:
        await c.get("/v1/explain/abc")
        await c.get("/v1/explain/def")
        text = (await c.get("/metrics")).text
    templated = (
        'codesearch_http_requests_total{method="GET",path="/v1/explain/{trace_id}",status="404"}'
    )
    assert metric(text, templated) == 2
    assert "/v1/explain/abc" not in text


async def test_metrics_report_the_live_index(client: Client, built: Any) -> None:
    _, snapshot = built
    async with client() as c:
        text = (await c.get("/metrics")).text
    assert metric(text, "codesearch_index_chunks") == snapshot.manifest.chunk_count
    assert f'index_version="{snapshot.manifest.index_version}"' in text


async def test_metrics_work_before_the_index_is_loaded(client: Client) -> None:
    async with client(handle=IndexHandle()) as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert "codesearch_index_info{" not in r.text
