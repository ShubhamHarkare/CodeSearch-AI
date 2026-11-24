"""The query path, step by step: the heart of v2.

What goes here:
    - ``AnswerPipeline``, constructed with every collaborator injected
      (retriever, reranker, llm, prompts, cache, groundedness scorer,
      calibrator, trace store, settings). It never constructs them itself.
    - ``async answer(question, history=None, bypass_cache=False) -> Answer``
    - ``async stream(question, history=None) -> AsyncIterator[StreamEvent]``

What it must do, in order:
    1. new trace_id; condense the question if there is history
    2. cache lookup (exact, then semantic); on hit, return / replay as a stream
    3. hybrid retrieval -> rerank -> abstain decision (no LLM call if abstaining)
    4. render the versioned prompt with numbered chunks; generate or stream
    5. validate citations; score sentence groundedness; compute confidence
    6. save the ExplanationTrace; store in cache; record RequestSignals
    7. return timings for every step (retrieval_ms, ttft_ms, total_ms)

Must not:
    - import httpx/Ollama, Chroma, Redis or Streamlit directly (only via interfaces),
      or know anything about HTTP.

Tested by: tests/integration/test_pipeline.py (FakeLLM + HashEmbedder,
mini corpus): answers with citations, abstains on off-topic questions,
cache hit on repeat, stream events arrive in the documented order.
Phase: 1 (answer), 2 (explain hooks), 3 (stream)
"""

import sqlite3
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codesearch.cache.keys import make_key
from codesearch.cache.singleflight import SingleFlight
from codesearch.config import Settings
from codesearch.errors import ExplainUnavailableError, TraceNotFoundError
from codesearch.explain.attribution import leave_one_out
from codesearch.explain.calibration import Calibrator, features
from codesearch.explain.groundedness import assess
from codesearch.explain.trace import TraceBuilder, TraceStore
from codesearch.generation.citations import sources, strip_invalid, to_citations
from codesearch.generation.prompts import PromptTemplate, format_context
from codesearch.index.handle import IndexHandle
from codesearch.interfaces import LLM, Embedder, EntailmentScorer, Reranker, Retriever
from codesearch.log import get_logger, new_trace_id
from codesearch.retrieval.abstain import ABSTAIN_MESSAGE, decide
from codesearch.retrieval.condense import condense
from codesearch.schemas import Answer, CacheTier, ChatMessage, Citation, ScoredChunk, StreamEvent

if TYPE_CHECKING:
    from codesearch.cache.tiered import TieredCache

log = get_logger(__name__)


@dataclass(frozen=True)
class Prepared:
    """Everything generation needs once retrieval decided to answer."""

    chunks: list[ScoredChunk]  # sent to the LLM, in prompt order
    similar: dict[str, Any]  # semantic-cache fields, reused when storing the answer
    prompt: str  # the rendered answer prompt


class AnswerPipeline:
    def __init__(
        self,
        *,
        handle: IndexHandle,
        retriever: Retriever,
        reranker: Reranker,
        llm: LLM,
        answer_prompt: PromptTemplate,
        condense_prompt: PromptTemplate,
        settings: Settings,
        cache: "TieredCache | None" = None,
        trace_store: TraceStore | None = None,
        grounder: EntailmentScorer | None = None,
        calibrator: Calibrator | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.handle = handle
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.answer_prompt = answer_prompt
        self.condense_prompt = condense_prompt
        self.settings = settings
        self.cache = cache
        self.trace_store = trace_store
        self.grounder = grounder
        self.calibrator = calibrator
        self.embedder = embedder
        # Identical questions in flight share one computation (see _compute).
        self.flights: SingleFlight[Answer] = SingleFlight()

    async def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
        *,
        bypass_cache: bool = False,
        trace_id: str | None = None,
    ) -> Answer:
        trace, pipeline_start, search_q, key = await self._start(question, history, trace_id)
        cache = None if bypass_cache else self.cache
        if cache is not None and (hit := cache.get(key)) is not None:
            trace.set_used_ids([c.chunk_id for c in hit.citations])
            return self._replay(hit, "exact", trace, pipeline_start)

        # bypass_cache means "do the real work": no cache and no sharing.
        if bypass_cache:
            return await self._compute(search_q, key, cache, trace, pipeline_start)

        # Single-flight: the first request for this key computes the answer; identical
        # requests that arrive meanwhile wait for it instead of calling the LLM again.
        answer, shared = await self.flights.do(
            key, lambda: self._compute(search_q, key, cache, trace, pipeline_start)
        )
        if shared:
            trace.set_used_ids([c.chunk_id for c in answer.citations])
            return self._replay(answer, "shared", trace, pipeline_start)
        return answer

    async def stream(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
        *,
        bypass_cache: bool = False,
        trace_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """Like answer(), but yields events as they happen: retrieval, token..., done.

        Streaming requests don't use single-flight: sharing a stream would mean
        broadcasting tokens to every waiting request. If the consumer stops early,
        the LLM stream is closed and nothing is cached (_finalize never runs).
        """
        trace, pipeline_start, search_q, key = await self._start(question, history, trace_id)
        cache = None if bypass_cache else self.cache

        # Exact cache hit: no LLM call. Send the whole cached text as one token, then done.
        if cache is not None and (hit := cache.get(key)) is not None:
            trace.set_used_ids([c.chunk_id for c in hit.citations])
            answer = self._replay(hit, "exact", trace, pipeline_start)
            yield StreamEvent(type="token", data=answer.text)
            yield StreamEvent(type="done", data=answer.model_dump(mode="json"))
            return

        # Abstain or semantic hit: also finished without the LLM.
        prepared = self._prepare(search_q, key, cache, trace, pipeline_start)
        if isinstance(prepared, Answer):
            yield StreamEvent(type="token", data=prepared.text)
            yield StreamEvent(type="done", data=prepared.model_dump(mode="json"))
            return

        # The numbered sources go out first, so the user sees them while the answer is written.
        yield StreamEvent(
            type="retrieval",
            data=[source.model_dump(mode="json") for source in sources(prepared.chunks)],
        )

        # Stream the answer. aclosing() closes the LLM stream (and Ollama's connection)
        # if this generator is closed early, e.g. because the user left.
        parts: list[str] = []
        with trace.time("generation"):
            async with aclosing(
                self.llm.stream(prepared.prompt, system=self.answer_prompt.system)
            ) as tokens:
                async for piece in tokens:
                    if not parts:  # first piece: what the user waited for
                        elapsed = time.perf_counter() - pipeline_start
                        trace.timings["ttft_ms"] = elapsed * 1000.0
                    parts.append(piece)
                    yield StreamEvent(type="token", data=piece)

        # Only reached when the stream finished: clean citations, score, cache, save the trace.
        answer = self._finalize("".join(parts), prepared, key, search_q, trace, pipeline_start)
        yield StreamEvent(type="done", data=answer.model_dump(mode="json"))

    async def _compute(
        self,
        search_q: str,
        key: str,
        cache: "TieredCache | None",
        trace: TraceBuilder,
        pipeline_start: float,
    ) -> Answer:
        """Retrieve, decide, generate and store one answer (everything after the exact cache)."""
        prepared = self._prepare(search_q, key, cache, trace, pipeline_start)
        if isinstance(prepared, Answer):  # abstain or semantic hit: nothing to generate
            return prepared
        with trace.time("generation"):
            raw = await self.llm.generate(prepared.prompt, system=self.answer_prompt.system)
        return self._finalize(raw, prepared, key, search_q, trace, pipeline_start)

    def _finalize(
        self,
        raw: str,
        prepared: Prepared,
        key: str,
        search_q: str,
        trace: TraceBuilder,
        pipeline_start: float,
    ) -> Answer:
        """Clean citations, check refusal, score groundedness and confidence, store, return."""
        chunks = prepared.chunks
        # Drop [n] markers that point past the context, then map the rest to sources.
        text, _removed = strip_invalid(raw, len(chunks))
        citations = to_citations(text, chunks)

        # The prompt tells the model to say this phrase when the context doesn't cover it.
        abstained = False
        if "isn't covered in the React docs" in text and not citations:
            abstained = True
            citations = []
            trace.set_abstain("model_refusal")

        groundedness: float | None = None
        if self.grounder is not None and not abstained:
            explain = self.settings.explain
            try:
                with trace.time("groundedness"):
                    sentences, groundedness = assess(
                        text,
                        citations,
                        chunks,
                        self.grounder,
                        supported_at=explain.supported_threshold,
                        partial_at=explain.partial_threshold,
                        all_sources=explain.check_all_sources,
                    )
                trace.set_sentences(sentences)
            except Exception as e:
                log.warning("groundedness.failed", trace_id=trace.trace_id, error=str(e))

        # Confidence: a calibrated probability that this answer is good. Only for real
        # answers (an abstain or a refusal has nothing to be confident about).
        confidence: float | None = None
        if self.calibrator is not None and not abstained:
            confidence = self.calibrator.confidence(features(chunks, groundedness))

        answer = self._finish(
            trace,
            pipeline_start,
            text=text,
            citations=citations,
            abstained=abstained,
            groundedness=groundedness,
            confidence=confidence,
        )
        # Only real answers are cached; a refusal is recomputed next time.
        if self.cache is not None and not abstained:
            self.cache.set(key, answer)
            self.cache.add_similar(search_q, answer, **prepared.similar)
        return answer

    def _finish(
        self,
        trace: TraceBuilder,
        start: float,
        *,
        text: str,
        citations: list[Citation],
        abstained: bool,
        groundedness: float | None = None,
        confidence: float | None = None,
    ) -> Answer:
        """A fresh answer (or an abstain): build it, save its trace, return it."""
        trace.timings["total_ms"] = (time.perf_counter() - start) * 1000.0
        answer = Answer(
            text=text,
            citations=citations,
            abstained=abstained,
            timings=dict(trace.timings),
            trace_id=trace.trace_id,
            index_version=self.handle.version,
            prompt_version=self.answer_prompt.version,
            groundedness=groundedness,
            confidence=confidence,
        )
        trace.set_answer(text)  # before saving: deep explain compares against it
        self._save(trace)
        return answer

    def _replay(
        self,
        hit: Answer,
        tier: CacheTier,
        trace: TraceBuilder,
        start: float,
    ) -> Answer:
        trace.set_cache_tier(tier)
        trace.timings["total_ms"] = (time.perf_counter() - start) * 1000.0
        self._save(trace)
        return hit.model_copy(
            update={
                "cache_tier": tier,
                "trace_id": trace.trace_id,
                "timings": dict(trace.timings),
            }
        )

    def _save(self, trace: TraceBuilder) -> None:
        """Store the trace. A storage failure is logged, never raised: explanations are
        secondary and must not cost the user their answer."""
        if self.trace_store is None:
            return
        try:
            self.trace_store.save(trace.build())
        except sqlite3.Error as e:
            log.warning("trace.save_failed", trace_id=trace.trace_id, error=str(e))

    async def deep_explain(self, trace_id: str) -> dict[str, float]:
        """Leave-one-out attribution for a stored answer: one LLM call per source."""
        if self.trace_store is None:
            raise TraceNotFoundError(f"no trace store, so no trace {trace_id!r}")
        trace = self.trace_store.get(trace_id)  # raises TraceNotFoundError if unknown
        if self.embedder is None:
            raise ExplainUnavailableError("deep explain needs an embedder")

        # The sources sent to the LLM, in prompt order (so [1] stays [1]).
        by_id = {c.chunk.id: c for c in trace.candidates}
        used = [by_id[cid] for cid in trace.used_chunk_ids if cid in by_id]
        if not used or not trace.answer or trace.abstain_reason:
            # a cache hit has no candidates; an abstain or refusal has nothing to attribute
            raise ExplainUnavailableError(f"trace {trace_id!r} has no sources to attribute")

        return await leave_one_out(
            trace.standalone_question,  # the question the answer was written for
            used,
            trace.answer,
            self.llm,
            self.embedder,
            self.answer_prompt,
        )

    async def _start(
        self, question: str, history: Sequence[ChatMessage] | None, trace_id: str | None
    ) -> tuple[TraceBuilder, float, str, str]:
        """Validate, start the trace, condense a follow-up, build the cache key."""
        if not question:
            raise ValueError(f"Expected question got: {question}")

        pipeline_start = time.perf_counter()
        trace = TraceBuilder(trace_id or new_trace_id(), question)
        trace.set_versions(
            index_version=self.handle.version,
            prompt_version=self.answer_prompt.version,
            model=self.settings.llm.model,
        )

        # A follow-up question is rewritten into a standalone one before retrieval.
        if history:
            with trace.time("condense"):
                search_q = await condense(
                    question=question,
                    history=history,
                    llm=self.llm,
                    prompt=self.condense_prompt,
                    max_turns=self.settings.retrieval.history_turns,
                )
            trace.set_standalone(search_q)

        else:
            search_q = question

        # Exact cache, keyed on the standalone question + everything that changes the answer.
        key = make_key(
            search_q, self.handle.version, self.answer_prompt.version, self.settings.llm.model
        )
        return trace, pipeline_start, search_q, key

    def _prepare(
        self,
        search_q: str,
        key: str,
        cache: "TieredCache | None",
        trace: TraceBuilder,
        pipeline_start: float,
    ) -> "Answer | Prepared":
        """Retrieve and decide. Returns a finished Answer (abstain or semantic hit) or Prepared."""
        with trace.time("retrieval"):
            raw_chunks = self.retriever.retrieve(search_q)
            trace.add_candidates(raw_chunks)
            chunks = self.reranker.rerank(
                search_q, raw_chunks, self.settings.retrieval.rerank_top_n
            )

        threshold = self.settings.retrieval.abstain_threshold
        decision = decide(chunks, threshold)

        if decision.abstain:
            if decision.reason == "below_threshold":
                trace.set_abstain(
                    f"below_threshold ({decision.score_kind} "
                    f"{decision.top_score:.3f} < {threshold})"
                )
            else:
                trace.set_abstain(decision.reason or "abstained")  # "no_candidates"
            return self._finish(
                trace, pipeline_start, text=ABSTAIN_MESSAGE, citations=[], abstained=True
            )

        similar = {
            "top1": chunks[0].chunk.id,
            "index_version": self.handle.version,
            "prompt_version": self.answer_prompt.version,
            "model": self.settings.llm.model,
        }
        if cache is not None and (hit := cache.lookup_similar(search_q, **similar)) is not None:
            cache.set(key, hit)
            trace.set_used_ids([c.chunk_id for c in hit.citations])
            return self._replay(hit, "semantic", trace, pipeline_start)

        # Cache miss: record what the prompt will see, then generate.
        trace.set_used(chunks)
        bm25 = self.handle.snapshot().bm25
        trace.set_matched_terms(
            {c.chunk.id: bm25.term_contributions(search_q, c.chunk.id) for c in chunks}
        )
        prompt = self.answer_prompt.render(context=format_context(chunks), question=search_q)
        return Prepared(chunks=chunks, similar=similar, prompt=prompt)
