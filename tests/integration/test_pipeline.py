"""End-to-end tests for codesearch/pipeline.py over the mini corpus.

A real index (22 pages, hash embedder) is built once; the LLM is a FakeLLM, so no
model is needed. Phase 1 covers answer(): condense -> retrieve -> rerank/cut ->
abstain -> prompt -> LLM -> citations. The cache (repeat question) and stream()
cases are added in later phases.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from typing import Any

import fakeredis
import pytest

from codesearch.cache.exact import RedisExactCache
from codesearch.cache.memory import LRUCache
from codesearch.cache.semantic import SemanticCache
from codesearch.cache.tiered import TieredCache
from codesearch.config import Settings
from codesearch.errors import IndexNotReadyError, LLMUnavailableError, TraceNotFoundError
from codesearch.explain.calibration import Calibrator
from codesearch.explain.groundedness import NLIScorer
from codesearch.explain.trace import TraceStore
from codesearch.factory import build_cache, build_pipeline
from codesearch.generation.llm import FakeLLM
from codesearch.generation.prompts import load_prompt
from codesearch.index.bm25 import tokenize
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
        # never write traces into the repo; never download the NLI model in tests
        explain={"trace_db": root / "traces.db", "groundedness": False},
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
    trace_store: Any = None,
    grounder: Any = None,
    calibrator: Any = None,
    explain: dict[str, Any] | None = None,
    **retrieval: Any,
) -> tuple[AnswerPipeline, Any]:
    settings, snapshot = built
    settings = settings.with_overrides(
        retrieval={"abstain_threshold": threshold, **retrieval}, explain=explain or {}
    )
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
        trace_store=trace_store,
        grounder=grounder,
        calibrator=calibrator,
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


# =====================================================================
# Traces: every answer (fresh, abstained, refused, cached) gets one
# =====================================================================


async def test_an_answer_saves_a_trace_under_its_trace_id(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    assert trace.question == QUESTION
    assert trace.standalone_question == QUESTION
    assert trace.abstain_reason is None
    assert trace.cache_tier is None


async def test_the_trace_lists_candidates_and_the_chunks_sent_to_the_llm(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, dense_k=10, rerank_top_n=3)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    assert len(trace.candidates) == 10  # everything retrieval returned
    assert len(trace.used_chunk_ids) == 3  # what the reranker kept for the prompt
    candidate_ids = [c.chunk.id for c in trace.candidates]
    assert set(trace.used_chunk_ids) <= set(candidate_ids)
    assert all(c.dense_score is not None for c in trace.candidates)
    cited = {c.chunk_id for c in answer.citations}
    assert cited <= set(trace.used_chunk_ids)  # every citation points at a used chunk


async def test_the_trace_records_versions_and_the_same_timings_as_the_answer(built: Any) -> None:
    settings, snapshot = built
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    assert trace.versions == {
        "index_version": snapshot.manifest.index_version,
        "prompt_version": load_prompt("answer").version,
        "model": settings.llm.model,
    }
    assert trace.timings == answer.timings
    assert {"retrieval_ms", "generation_ms", "total_ms"} <= set(trace.timings)


async def test_a_follow_up_trace_has_both_questions(built: Any) -> None:
    store = TraceStore()
    llm = FakeLLM(responses=[REWRITE, "Copy the object first [1]."])
    pipeline, _ = make(built, llm=llm, trace_store=store)
    answer = await pipeline.answer("what about objects?", history=HISTORY)
    trace = store.get(answer.trace_id)
    assert trace.question == "what about objects?"
    assert trace.standalone_question == REWRITE
    assert "condense_ms" in trace.timings


async def test_an_abstained_answer_has_a_trace_saying_why(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, threshold=ALWAYS_ABSTAIN)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    assert trace.abstain_reason is not None
    assert trace.abstain_reason.startswith("below_threshold")
    assert "dense" in trace.abstain_reason  # which score was judged
    assert trace.candidates  # you can still see what retrieval found
    assert trace.used_chunk_ids == []  # nothing went to the LLM


async def test_a_model_refusal_is_recorded_as_the_abstain_reason(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, llm=FakeLLM(responses=[NOT_COVERED]), trace_store=store)
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is True
    assert store.get(answer.trace_id).abstain_reason == "model_refusal"


async def test_an_exact_cache_hit_has_its_own_trace(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, cache=memory_cache(), trace_store=store)
    first = await pipeline.answer(QUESTION)
    hit = await pipeline.answer(QUESTION)
    trace = store.get(hit.trace_id)
    assert trace.cache_tier == "exact"
    assert trace.candidates == []  # retrieval never ran
    assert trace.used_chunk_ids == [c.chunk_id for c in first.citations]  # from the cached answer
    assert len(store) == 2


async def test_a_semantic_cache_hit_trace_keeps_the_retrieval(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, cache=memory_cache(), trace_store=store)
    await pipeline.answer(QUESTION)
    hit = await pipeline.answer("Based on the previous state how do I update state?")
    trace = store.get(hit.trace_id)
    assert trace.cache_tier == "semantic"
    assert trace.candidates  # retrieval ran before the semantic lookup


async def test_a_failed_request_saves_no_trace(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, llm=DownLLM(), trace_store=store)
    with pytest.raises(LLMUnavailableError):
        await pipeline.answer(QUESTION, trace_id="failed")
    with pytest.raises(TraceNotFoundError):
        store.get("failed")


class BrokenStore(TraceStore):
    def save(self, trace: Any) -> None:
        raise sqlite3.OperationalError("disk I/O error")


async def test_a_broken_trace_store_never_breaks_the_answer(built: Any) -> None:
    """Explanations are secondary: losing a trace must not lose the answer."""
    pipeline, _ = make(built, trace_store=BrokenStore())
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is False


async def test_without_a_store_answers_still_work(built: Any) -> None:
    pipeline, _ = make(built)
    assert (await pipeline.answer(QUESTION)).abstained is False


def test_build_pipeline_includes_a_trace_store(built: Any, tmp_path: Any) -> None:
    settings, _ = built
    settings = settings.with_overrides(explain={"trace_db": tmp_path / "t" / "traces.db"})
    pipeline = build_pipeline(settings)
    assert isinstance(pipeline.trace_store, TraceStore)
    assert (tmp_path / "t" / "traces.db").exists()


# =====================================================================
# Matched terms: which words of the question each used source contains
# =====================================================================


async def test_every_used_chunk_has_its_matched_terms(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store)
    trace = store.get((await pipeline.answer(QUESTION)).trace_id)
    assert set(trace.matched_terms) == set(trace.used_chunk_ids)


async def test_matched_terms_come_from_the_question_and_have_positive_weights(
    built: Any,
) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store)
    trace = store.get((await pipeline.answer(QUESTION)).trace_id)
    question_words = set(tokenize(QUESTION))
    all_terms = [t for terms in trace.matched_terms.values() for t in terms]
    assert all_terms, "at least one used source should share a word with the question"
    assert set(all_terms) <= question_words
    assert all(w > 0 for terms in trace.matched_terms.values() for w in terms.values())


async def test_matched_terms_use_the_condensed_question(built: Any) -> None:
    """The follow-up "what about objects?" was searched as the rewrite, so the
    matched words must come from the rewrite."""
    store = TraceStore()
    llm = FakeLLM(responses=[REWRITE, "Copy the object first [1]."])
    pipeline, _ = make(built, llm=llm, trace_store=store)
    trace = store.get((await pipeline.answer("what about objects?", history=HISTORY)).trace_id)
    all_terms = {t for terms in trace.matched_terms.values() for t in terms}
    assert all_terms <= set(tokenize(REWRITE))


async def test_matched_terms_add_up_to_the_bm25_score(built: Any) -> None:
    """In hybrid mode, a chunk that BM25 retrieved has a bm25_score; its matched-term
    contributions must add up to exactly that score."""
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, mode="hybrid", bm25_k=10, dense_k=10)
    trace = store.get((await pipeline.answer(QUESTION)).trace_id)
    by_id = {c.chunk.id: c for c in trace.candidates}
    checked = 0
    for chunk_id, terms in trace.matched_terms.items():
        bm25_score = by_id[chunk_id].bm25_score
        if bm25_score is not None:
            assert sum(terms.values()) == pytest.approx(bm25_score)
            checked += 1
    assert checked, "at least one used chunk should have come from BM25"


async def test_abstains_and_cache_hits_have_no_matched_terms(built: Any) -> None:
    store = TraceStore()
    abstaining, _ = make(built, trace_store=store, threshold=ALWAYS_ABSTAIN)
    assert store.get((await abstaining.answer(QUESTION)).trace_id).matched_terms == {}

    cached, _ = make(built, trace_store=store, cache=memory_cache())
    await cached.answer(QUESTION)
    hit = await cached.answer(QUESTION)
    assert store.get(hit.trace_id).matched_terms == {}


# =====================================================================
# Groundedness: each cited sentence checked against its chunk
# =====================================================================


class FakeGrounder:
    """Entailment from a rule on (premise, hypothesis); records every batch."""

    def __init__(self, rule: Any = 0.9) -> None:
        self.rule = rule
        self.calls: list[list[tuple[str, str]]] = []

    def entailment(self, pairs: Any) -> list[float]:
        self.calls.append(list(pairs))
        if callable(self.rule):
            return [self.rule(p, h) for p, h in pairs]
        return [float(self.rule)] * len(pairs)


class ExplodingGrounder(FakeGrounder):
    def entailment(self, pairs: Any) -> list[float]:
        raise RuntimeError("NLI model failed to load")


# FakeLLM's default answer: two sentences citing [1] and [2].
DEFAULT_SENTENCES = FakeLLM.DEFAULT_ANSWER.count(".")


async def test_a_grounded_answer_gets_a_score_and_sentence_labels(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, grounder=FakeGrounder(0.9))
    answer = await pipeline.answer(QUESTION)
    assert answer.groundedness == 1.0
    trace = store.get(answer.trace_id)
    assert len(trace.sentences) == DEFAULT_SENTENCES
    assert all(s.label == "supported" for s in trace.sentences)
    assert "groundedness_ms" in answer.timings


async def test_an_ungrounded_answer_scores_zero(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, grounder=FakeGrounder(0.1))
    answer = await pipeline.answer(QUESTION)
    assert answer.groundedness == 0.0
    assert all(s.label == "unsupported" for s in store.get(answer.trace_id).sentences)


async def test_each_sentence_is_checked_against_the_chunk_it_cites(built: Any) -> None:
    grounder = FakeGrounder(0.9)
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, grounder=grounder)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    texts = {c.chunk.id: c.chunk.text for c in trace.candidates}
    cited = {c.number: c.chunk_id for c in answer.citations}
    premises = [premise for premise, _ in grounder.calls[0]]
    assert premises == [texts[cited[1]], texts[cited[2]]]


async def test_thresholds_come_from_settings(built: Any) -> None:
    store = TraceStore()
    loose, _ = make(
        built,
        trace_store=store,
        grounder=FakeGrounder(0.5),
        explain={"supported_threshold": 0.45, "partial_threshold": 0.2},
    )
    answer = await loose.answer(QUESTION)
    assert answer.groundedness == 1.0
    assert {s.label for s in store.get(answer.trace_id).sentences} == {"supported"}

    strict, _ = make(built, trace_store=store, grounder=FakeGrounder(0.5))
    answer = await strict.answer(QUESTION)
    assert {s.label for s in store.get(answer.trace_id).sentences} == {"partial"}


async def test_abstained_answers_are_not_graded(built: Any) -> None:
    grounder = FakeGrounder(0.9)
    pipeline, _ = make(built, grounder=grounder, threshold=ALWAYS_ABSTAIN)
    answer = await pipeline.answer(QUESTION)
    assert answer.groundedness is None
    assert grounder.calls == []


async def test_a_model_refusal_is_not_graded(built: Any) -> None:
    grounder = FakeGrounder(0.9)
    pipeline, _ = make(built, llm=FakeLLM(responses=[NOT_COVERED]), grounder=grounder)
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is True
    assert answer.groundedness is None
    assert grounder.calls == []


async def test_a_failing_grounder_never_breaks_the_answer(built: Any) -> None:
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, grounder=ExplodingGrounder())
    answer = await pipeline.answer(QUESTION)
    assert answer.abstained is False
    assert answer.groundedness is None
    assert store.get(answer.trace_id).sentences == []


async def test_without_a_grounder_groundedness_is_none(built: Any) -> None:
    pipeline, _ = make(built)
    assert (await pipeline.answer(QUESTION)).groundedness is None


async def test_a_cache_hit_keeps_the_original_groundedness(built: Any) -> None:
    grounder = FakeGrounder(0.9)
    pipeline, _ = make(built, cache=memory_cache(), grounder=grounder)
    await pipeline.answer(QUESTION)
    hit = await pipeline.answer(QUESTION)
    assert hit.cache_tier == "exact"
    assert hit.groundedness == 1.0
    assert len(grounder.calls) == 1  # graded once, when it was generated


def test_build_pipeline_adds_the_nli_grounder_when_enabled(built: Any) -> None:
    settings, _ = built
    pipeline = build_pipeline(settings.with_overrides(explain={"groundedness": True}))
    assert isinstance(pipeline.grounder, NLIScorer)
    assert pipeline.grounder.model_name == settings.explain.nli_model


def test_build_pipeline_without_groundedness(built: Any) -> None:
    settings, _ = built
    assert build_pipeline(settings).grounder is None  # the fixture turns it off


async def test_the_all_sources_diagnostic_is_off_by_default(built: Any) -> None:
    """In production each sentence is checked only against what it cites (cheap)."""
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, grounder=FakeGrounder(0.9))
    answer = await pipeline.answer(QUESTION)
    assert all(s.best_score is None for s in store.get(answer.trace_id).sentences)


async def test_the_all_sources_diagnostic_can_be_switched_on(built: Any) -> None:
    store = TraceStore()
    grounder = FakeGrounder(0.9)
    pipeline, _ = make(
        built, trace_store=store, grounder=grounder, explain={"check_all_sources": True}
    )
    answer = await pipeline.answer(QUESTION)
    sentences = store.get(answer.trace_id).sentences
    assert all(s.best_score == 0.9 for s in sentences)
    n_chunks = len(store.get(answer.trace_id).used_chunk_ids)
    assert len(grounder.calls[0]) == len(sentences) * n_chunks


# =====================================================================
# Confidence: a calibrated probability that the answer is good
# =====================================================================


class SpyCalibrator(Calibrator):
    """A real Calibrator that also records the feature vectors it was given."""

    def __init__(self) -> None:
        super().__init__(coef=[0.0, 0.0, 2.0], intercept=-1.0)  # driven by groundedness
        object.__setattr__(self, "seen", [])

    def confidence(self, x: Any) -> float:
        self.seen.append(list(x))  # type: ignore[attr-defined]
        return super().confidence(x)


async def test_an_answer_gets_a_calibrated_confidence(built: Any) -> None:
    calibrator = SpyCalibrator()
    pipeline, _ = make(built, grounder=FakeGrounder(0.9), calibrator=calibrator)
    answer = await pipeline.answer(QUESTION)
    assert answer.confidence is not None
    assert answer.confidence == pytest.approx(calibrator.confidence(calibrator.seen[0]))
    top, mean3, grounded = calibrator.seen[0]
    assert grounded == answer.groundedness == 1.0
    assert top >= mean3 > 0  # dense scores of the chunks sent to the LLM


async def test_confidence_uses_the_chunks_sent_to_the_llm(built: Any) -> None:
    calibrator = SpyCalibrator()
    store = TraceStore()
    pipeline, _ = make(built, trace_store=store, calibrator=calibrator, rerank_top_n=3)
    answer = await pipeline.answer(QUESTION)
    trace = store.get(answer.trace_id)
    used = [c for c in trace.candidates if c.chunk.id in trace.used_chunk_ids]
    expected_top = max(c.dense_score for c in used if c.dense_score is not None)
    assert calibrator.seen[0][0] == pytest.approx(expected_top)


async def test_no_grounder_means_groundedness_feature_is_zero(built: Any) -> None:
    calibrator = SpyCalibrator()
    pipeline, _ = make(built, calibrator=calibrator)
    answer = await pipeline.answer(QUESTION)
    assert calibrator.seen[0][2] == 0.0
    assert answer.confidence is not None


async def test_abstained_answers_have_no_confidence(built: Any) -> None:
    calibrator = SpyCalibrator()
    pipeline, _ = make(built, calibrator=calibrator, threshold=ALWAYS_ABSTAIN)
    assert (await pipeline.answer(QUESTION)).confidence is None
    refusing, _ = make(built, llm=FakeLLM(responses=[NOT_COVERED]), calibrator=calibrator)
    assert (await refusing.answer(QUESTION)).confidence is None
    assert calibrator.seen == []


async def test_without_a_calibrator_confidence_is_none(built: Any) -> None:
    pipeline, _ = make(built, grounder=FakeGrounder(0.9))
    assert (await pipeline.answer(QUESTION)).confidence is None


async def test_a_cache_hit_keeps_the_original_confidence(built: Any) -> None:
    calibrator = SpyCalibrator()
    pipeline, _ = make(
        built, cache=memory_cache(), grounder=FakeGrounder(0.9), calibrator=calibrator
    )
    first = await pipeline.answer(QUESTION)
    hit = await pipeline.answer(QUESTION)
    assert hit.cache_tier == "exact"
    assert hit.confidence == first.confidence
    assert len(calibrator.seen) == 1


def test_build_pipeline_loads_a_saved_calibrator(built: Any, tmp_path: Any) -> None:
    settings, _ = built
    path = tmp_path / "calibrator.json"
    Calibrator(coef=[1.0, 1.0, 1.0], intercept=0.0).save(path)
    pipeline = build_pipeline(settings.with_overrides(explain={"calibrator_path": path}))
    assert pipeline.calibrator == Calibrator(coef=[1.0, 1.0, 1.0], intercept=0.0)


def test_build_pipeline_without_a_calibrator_file(built: Any, tmp_path: Any) -> None:
    """Not fitted yet: answers simply have no confidence (never a crash)."""
    settings, _ = built
    pipeline = build_pipeline(
        settings.with_overrides(explain={"calibrator_path": tmp_path / "none.json"})
    )
    assert pipeline.calibrator is None
