"""End-to-end tests for codesearch/pipeline.py over the mini corpus.

A real index (22 pages, hash embedder) is built once; the LLM is a FakeLLM, so no
model is needed. Phase 1 covers answer(): condense -> retrieve -> rerank/cut ->
abstain -> prompt -> LLM -> citations. The cache (repeat question) and stream()
cases are added in later phases.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import fakeredis
import pytest

from codesearch.cache.exact import RedisExactCache
from codesearch.cache.memory import LRUCache
from codesearch.cache.semantic import SemanticCache
from codesearch.cache.tiered import TieredCache
from codesearch.config import Settings
from codesearch.errors import IndexNotReadyError, LLMUnavailableError
from codesearch.factory import build_cache, build_pipeline
from codesearch.generation.llm import FakeLLM
from codesearch.generation.prompts import load_prompt
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.pipeline import AnswerPipeline
from codesearch.retrieval.abstain import ABSTAIN_MESSAGE
from codesearch.retrieval.hybrid import HybridRetriever
from codesearch.retrieval.rerank import NoopReranker
from codesearch.schemas import Answer, ChatMessage, ScoredChunk
from tests.conftest import FakeReactDev

QUESTION = "How do I update state based on the previous state?"
NOT_COVERED = "This isn't covered in the React docs I have."

# The hash embedder's cosine scores are in [-1, 1]: a threshold below that range
# always answers, one above it always abstains.
ALWAYS_ANSWER = -2.0
ALWAYS_ABSTAIN = 2.0


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, IndexSnapshot]:
    root = tmp_path_factory.mktemp("pipeline")
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
    )
    build_index(settings, HashEmbedder())
    snapshot = IndexSnapshot.open(root / "index")
    assert snapshot is not None
    return settings, snapshot


class SpyReranker(NoopReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        self.calls.append((query, len(chunks), top_n))
        return super().rerank(query, chunks, top_n)


class DownLLM(FakeLLM):
    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise LLMUnavailableError("ollama is not running")


def make(
    built: tuple[Settings, IndexSnapshot],
    *,
    llm: Any = None,
    reranker: Any = None,
    threshold: float = ALWAYS_ANSWER,
    handle: IndexHandle | None = None,
    cache: Any = None,
    answer_prompt: Any = None,
    **retrieval: Any,
) -> tuple[AnswerPipeline, Any]:
    settings, snapshot = built
    settings = settings.with_overrides(retrieval={"abstain_threshold": threshold, **retrieval})
    handle = handle if handle is not None else IndexHandle(snapshot)
    llm = llm if llm is not None else FakeLLM()
    pipeline = AnswerPipeline(
        handle=handle,
        retriever=HybridRetriever(handle, HashEmbedder(), settings.retrieval),
        reranker=reranker if reranker is not None else NoopReranker(),
        llm=llm,
        answer_prompt=answer_prompt if answer_prompt is not None else load_prompt("answer"),
        condense_prompt=load_prompt("condense"),
        settings=settings,
        cache=cache,
    )
    return pipeline, llm


# =====================================================================
# Answering
# =====================================================================


async def test_answers_with_citations_to_react_dev(built: Any) -> None:
    pipeline, _ = make(built)
    answer = await pipeline.answer(QUESTION)
    assert isinstance(answer, Answer)
    assert answer.abstained is False
    assert answer.text == FakeLLM.DEFAULT_ANSWER  # cites [1] and [2]
    assert [c.number for c in answer.citations] == [1, 2]
    assert all(c.url.startswith("https://react.dev/") for c in answer.citations)


async def test_answer_metadata(built: Any) -> None:
    _, snapshot = built
    pipeline, _ = make(built)
    answer = await pipeline.answer(QUESTION)
    assert answer.index_version == snapshot.manifest.index_version
    assert answer.prompt_version == load_prompt("answer").version
    assert len(answer.trace_id) >= 16
    assert answer.cache_tier is None
    # filled in Phase 2 (groundedness scorer, calibration)
    assert answer.confidence is None
    assert answer.groundedness is None


async def test_every_call_gets_a_new_trace_id(built: Any) -> None:
    pipeline, _ = make(built)
    first = await pipeline.answer(QUESTION)
    second = await pipeline.answer(QUESTION)
    assert first.trace_id != second.trace_id


async def test_timings_are_recorded(built: Any) -> None:
    pipeline, _ = make(built)
    timings = (await pipeline.answer(QUESTION)).timings
    for key in ("retrieval_ms", "generation_ms", "total_ms"):
        assert timings[key] >= 0
    assert timings["total_ms"] >= timings["retrieval_ms"]
    assert "condense_ms" not in timings  # no history, no condense step


async def test_llm_gets_the_rendered_answer_prompt_and_its_system_rules(built: Any) -> None:
    pipeline, llm = make(built)
    await pipeline.answer(QUESTION)
    (prompt,) = llm.prompts
    assert f"Question: {QUESTION}" in prompt
    assert "[1] https://react.dev/" in prompt
    assert llm.systems == [load_prompt("answer").system]


async def test_context_is_cut_to_rerank_top_n(built: Any) -> None:
    pipeline, llm = make(built, rerank_top_n=3)
    await pipeline.answer(QUESTION)
    assert "[3] https://react.dev/" in llm.prompts[0]
    assert "[4] https://react.dev/" not in llm.prompts[0]


async def test_reranker_is_given_the_candidates_and_top_n(built: Any) -> None:
    spy = SpyReranker()
    pipeline, _ = make(built, reranker=spy, dense_k=10, rerank_top_n=4)
    await pipeline.answer(QUESTION)
    assert spy.calls == [(QUESTION, 10, 4)]


async def test_invalid_citations_are_stripped(built: Any) -> None:
    pipeline, _ = make(built, llm=FakeLLM(responses=["Use an updater [1]. Also [9]."]))
    answer = await pipeline.answer(QUESTION)
    assert answer.text == "Use an updater [1]. Also."
    assert [c.number for c in answer.citations] == [1]


async def test_citations_point_at_the_chunks_that_were_sent(built: Any) -> None:
    pipeline, llm = make(built, llm=FakeLLM(responses=["Second source [2]."]))
    answer = await pipeline.answer(QUESTION)
    (citation,) = answer.citations
    second_entry = llm.prompts[0].split("[2] ")[1].split("\n")[0]  # the URL line of [2]
    assert citation.url == second_entry


# =====================================================================
# Abstaining
# =====================================================================


async def test_low_score_abstains_without_calling_the_llm(built: Any) -> None:
    pipeline, llm = make(built, threshold=ALWAYS_ABSTAIN)
    answer = await pipeline.answer("How do I write a computed property in Vue?")
    assert answer.abstained is True
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == []
    assert llm.prompts == []  # no LLM call: fast and free
    assert "generation_ms" not in answer.timings


async def test_model_saying_not_covered_counts_as_abstaining(built: Any) -> None:
    pipeline, _ = make(built, llm=FakeLLM(responses=[NOT_COVERED]))
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is True
    assert answer.citations == []
    assert answer.text == NOT_COVERED


# =====================================================================
# Follow-up questions (history -> condense)
# =====================================================================

HISTORY = [
    ChatMessage(role="user", content="How do I add state to a component?"),
    ChatMessage(role="assistant", content="Call useState at the top level [1]."),
]
REWRITE = "How do I update an object stored in useState?"


async def test_history_condenses_the_question_before_retrieval(built: Any) -> None:
    llm = FakeLLM(responses=[REWRITE, "Copy the object first [1]."])
    pipeline, _ = make(built, llm=llm)
    answer = await pipeline.answer("what about objects?", history=HISTORY)
    assert len(llm.prompts) == 2
    assert "Follow-up question: what about objects?" in llm.prompts[0]  # condense prompt
    assert f"Question: {REWRITE}" in llm.prompts[1]  # answer prompt uses the rewrite
    assert "condense_ms" in answer.timings


async def test_reranker_gets_the_condensed_question(built: Any) -> None:
    spy = SpyReranker()
    llm = FakeLLM(responses=[REWRITE, "ok [1]."])
    pipeline, _ = make(built, llm=llm, reranker=spy)
    await pipeline.answer("what about objects?", history=HISTORY)
    assert spy.calls[0][0] == REWRITE


async def test_empty_history_is_the_same_as_none(built: Any) -> None:
    pipeline, llm = make(built)
    await pipeline.answer(QUESTION, history=[])
    assert len(llm.prompts) == 1


# =====================================================================
# Errors
# =====================================================================


@pytest.mark.parametrize("question", ["", "   "])
async def test_blank_question_is_rejected(built: Any, question: str) -> None:
    pipeline, llm = make(built)
    with pytest.raises(ValueError):
        await pipeline.answer(question)
    assert llm.prompts == []


async def test_no_index_raises_index_not_ready(built: Any) -> None:
    pipeline, _ = make(built, handle=IndexHandle())
    with pytest.raises(IndexNotReadyError):
        await pipeline.answer(QUESTION)


async def test_llm_outage_propagates(built: Any) -> None:
    """The API turns this into a 503; the pipeline must not invent an answer."""
    pipeline, _ = make(built, llm=DownLLM())
    with pytest.raises(LLMUnavailableError):
        await pipeline.answer(QUESTION)


# =====================================================================
# factory.build_pipeline
# =====================================================================


async def test_build_pipeline_wires_everything_from_settings(built: Any) -> None:
    settings, _ = built
    settings = settings.with_overrides(retrieval={"abstain_threshold": ALWAYS_ANSWER})
    pipeline = build_pipeline(settings)
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is False
    assert answer.citations


# =====================================================================
# Cache (exact before retrieval, semantic after retrieval)
# =====================================================================


def memory_cache() -> TieredCache:
    """Both exact tiers in memory (no Redis needed) plus a real semantic tier."""
    return TieredCache(
        exact=LRUCache(), fallback=LRUCache(), semantic=SemanticCache(HashEmbedder())
    )


async def test_a_repeated_question_is_an_exact_hit_without_calling_the_llm(built: Any) -> None:
    cache = memory_cache()
    pipeline, llm = make(built, cache=cache)
    first = await pipeline.answer(QUESTION)
    second = await pipeline.answer(QUESTION)
    assert len(llm.prompts) == 1
    assert second.cache_tier == "exact"
    assert second.text == first.text
    assert second.citations == first.citations
    assert first.cache_tier is None


async def test_a_cache_hit_gets_its_own_trace_id_and_timings(built: Any) -> None:
    pipeline, _ = make(built, cache=memory_cache())
    first = await pipeline.answer(QUESTION)
    second = await pipeline.answer(QUESTION)
    assert second.trace_id != first.trace_id
    assert "total_ms" in second.timings
    assert "generation_ms" not in second.timings  # nothing was generated this time


async def test_spelling_variants_share_the_exact_key(built: Any) -> None:
    pipeline, llm = make(built, cache=memory_cache())
    await pipeline.answer(QUESTION)
    again = await pipeline.answer("  how do i update state based on the previous state  ")
    assert again.cache_tier == "exact"
    assert len(llm.prompts) == 1


async def test_a_paraphrase_is_a_semantic_hit(built: Any) -> None:
    """The hash embedder is a bag of words: the same words reordered embed identically,
    so retrieval finds the same top chunk and the cosine is 1.0."""
    pipeline, llm = make(built, cache=memory_cache())
    await pipeline.answer(QUESTION)
    paraphrase = "Based on the previous state how do I update state?"
    hit = await pipeline.answer(paraphrase)
    assert hit.cache_tier == "semantic"
    assert len(llm.prompts) == 1


async def test_abstained_answers_are_not_cached(built: Any) -> None:
    cache = memory_cache()
    pipeline, _ = make(built, cache=cache, threshold=ALWAYS_ABSTAIN)
    await pipeline.answer(QUESTION)
    second = await pipeline.answer(QUESTION)
    assert second.cache_tier is None
    assert cache.stats()["exact_hits"] == 0


async def test_a_model_refusal_is_not_cached(built: Any) -> None:
    llm = FakeLLM(responses=[NOT_COVERED, "Use an updater [1]."])
    pipeline, _ = make(built, llm=llm, cache=memory_cache())
    first = await pipeline.answer(QUESTION)
    second = await pipeline.answer(QUESTION)
    assert first.abstained is True
    assert second.abstained is False  # the LLM was asked again
    assert len(llm.prompts) == 2


async def test_bypass_cache_skips_the_lookup_but_refreshes_the_entry(built: Any) -> None:
    llm = FakeLLM(responses=["Old [1].", "New [1]."])
    pipeline, _ = make(built, llm=llm, cache=memory_cache())
    await pipeline.answer(QUESTION)
    fresh = await pipeline.answer(QUESTION, bypass_cache=True)
    assert fresh.cache_tier is None
    assert fresh.text == "New [1]."
    third = await pipeline.answer(QUESTION)
    assert third.cache_tier == "exact"
    assert third.text == "New [1]."  # the refreshed answer was stored


async def test_a_new_prompt_version_never_serves_old_answers(built: Any) -> None:
    cache = memory_cache()
    old_pipeline, _ = make(built, cache=cache)
    await old_pipeline.answer(QUESTION)
    new_prompt = dataclasses.replace(load_prompt("answer"), version="999")
    new_pipeline, new_llm = make(built, cache=cache, answer_prompt=new_prompt)
    answer = await new_pipeline.answer(QUESTION)
    assert answer.cache_tier is None
    assert len(new_llm.prompts) == 1


async def test_follow_ups_are_cached_under_the_condensed_question(built: Any) -> None:
    """ "what about objects?" means different things in different chats, so the key is
    built from the standalone rewrite, never the raw follow-up."""
    llm = FakeLLM(responses=[REWRITE, "Copy the object first [1]."])
    pipeline, _ = make(built, llm=llm, cache=memory_cache())
    await pipeline.answer("what about objects?", history=HISTORY)
    direct = await pipeline.answer(REWRITE)
    assert direct.cache_tier == "exact"
    assert len(llm.prompts) == 2  # condense + one answer, nothing for the direct ask


async def test_redis_down_still_answers(built: Any) -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    cache = TieredCache(
        exact=RedisExactCache(client=fakeredis.FakeRedis(server=server, decode_responses=True)),
        fallback=LRUCache(),
        semantic=SemanticCache(HashEmbedder()),
    )
    pipeline, _ = make(built, cache=cache)
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is False
    assert cache.degraded is True
    again = await pipeline.answer(QUESTION)
    assert again.cache_tier == "exact"  # served from the memory fallback


async def test_no_cache_means_every_question_reaches_the_llm(built: Any) -> None:
    pipeline, llm = make(built)
    await pipeline.answer(QUESTION)
    await pipeline.answer(QUESTION)
    assert len(llm.prompts) == 2


# =====================================================================
# factory.build_cache
# =====================================================================


def test_build_cache_from_settings(built: Any) -> None:
    settings, _ = built
    cache = build_cache(settings, HashEmbedder())
    assert isinstance(cache, TieredCache)
    assert isinstance(cache.exact, RedisExactCache)
    assert isinstance(cache.fallback, LRUCache)
    assert isinstance(cache.semantic, SemanticCache)
    assert cache.semantic.threshold == settings.cache.semantic_threshold


def test_build_cache_can_be_disabled(built: Any) -> None:
    settings, _ = built
    assert build_cache(settings.with_overrides(cache={"enabled": False}), HashEmbedder()) is None


def test_build_cache_without_the_semantic_tier(built: Any) -> None:
    settings, _ = built
    cache = build_cache(settings.with_overrides(cache={"semantic_enabled": False}), HashEmbedder())
    assert cache is not None
    assert cache.semantic is None


def test_build_pipeline_includes_the_cache(built: Any) -> None:
    settings, _ = built
    assert isinstance(build_pipeline(settings).cache, TieredCache)


async def test_a_caller_supplied_trace_id_is_used(built: Any) -> None:
    """The API passes the request's X-Trace-Id so logs, header and body all match."""
    pipeline, _ = make(built)
    assert (await pipeline.answer(QUESTION, trace_id="req-1")).trace_id == "req-1"


async def test_a_cache_hit_takes_the_callers_trace_id(built: Any) -> None:
    pipeline, _ = make(built, cache=memory_cache())
    await pipeline.answer(QUESTION)
    assert (await pipeline.answer(QUESTION, trace_id="req-2")).trace_id == "req-2"
