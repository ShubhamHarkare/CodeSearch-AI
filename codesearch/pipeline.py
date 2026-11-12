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

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from codesearch.cache.keys import make_key
from codesearch.config import Settings
from codesearch.generation.citations import strip_invalid, to_citations
from codesearch.generation.prompts import PromptTemplate, format_context
from codesearch.index.handle import IndexHandle
from codesearch.interfaces import LLM, Reranker, Retriever
from codesearch.log import new_trace_id
from codesearch.retrieval.abstain import ABSTAIN_MESSAGE, decide
from codesearch.retrieval.condense import condense
from codesearch.schemas import Answer, ChatMessage

if TYPE_CHECKING:
    from codesearch.cache.tiered import TieredCache


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
    ) -> None:
        self.handle = handle
        self.retriver = retriever
        self.reranker = reranker
        self.llm = llm
        self.answer_prompt = answer_prompt
        self.condense_prompt = condense_prompt
        self.settings = settings
        self.cache = cache

    async def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
        *,
        bypass_cache: bool = False,
        trace_id: str | None = None,
    ) -> Answer:
        if not question:
            raise ValueError(f"Expected question got: {question}")

        pipeline_start = time.perf_counter()
        trace_id = trace_id or new_trace_id()
        timings: dict[str, float] = {}

        # HACK: The below code is for condensing the history and the new question
        # HACK: So that we can get the history and the record time for that
        if history:
            t0 = time.perf_counter()
            search_q = await condense(
                question=question,
                history=history,
                llm=self.llm,
                prompt=self.condense_prompt,
                max_turns=self.settings.retrieval.history_turns,
            )
            timings["condense_ms"] = (time.perf_counter() - t0) * 1000.0
        else:
            search_q = question

        # Exact cache, keyed on the standalone question + everything that changes the answer.
        key = make_key(
            search_q, self.handle.version, self.answer_prompt.version, self.settings.llm.model
        )
        cache = None if bypass_cache else self.cache
        if cache is not None and (hit := cache.get(key)) is not None:
            return self._replay(hit, "exact", trace_id, timings, pipeline_start)

        # HACK: This is for retrieval
        t0 = time.perf_counter()
        raw_chunks = self.retriver.retrieve(search_q)

        chunks = self.reranker.rerank(
            search_q,
            raw_chunks,
            self.settings.retrieval.rerank_top_n,  # how many chunks go to the LLM
        )
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

        # HACK: The below is for getting the abstain and the decisions on the code
        decision = decide(chunks, self.settings.retrieval.abstain_threshold)

        if decision.abstain:
            timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
            return Answer(
                text=ABSTAIN_MESSAGE,
                citations=[],
                abstained=True,
                timings=timings,
                trace_id=trace_id,
                index_version=self.handle.version,
                prompt_version=self.answer_prompt.version,
            )

        similar = {
            "top1": chunks[0].chunk.id,
            "index_version": self.handle.version,
            "prompt_version": self.answer_prompt.version,
            "model": self.settings.llm.model,
        }
        if cache is not None and (hit := cache.lookup_similar(search_q, **similar)) is not None:
            cache.set(key, hit)
            return self._replay(hit, "semantic", trace_id, timings, pipeline_start)

        # HACK if we do not abstain and get the answer then this is the code for that
        t0 = time.perf_counter()
        prompt = self.answer_prompt.render(context=format_context(chunks), question=search_q)
        raw = await self.llm.generate(prompt, system=self.answer_prompt.system)
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 9: Process citations
        text, _removed = strip_invalid(raw, len(chunks))
        citations = to_citations(text, chunks)

        # Step 10: Model refusal check
        abstained = False
        if "isn't covered in the React docs" in text and not citations:
            abstained = True
            citations = []

        # Step 11: Return completed Answer with total timing
        timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
        answer = Answer(
            text=text,
            citations=citations,
            abstained=abstained,
            timings=timings,
            trace_id=trace_id,
            index_version=self.handle.version,
            prompt_version=self.answer_prompt.version,
        )
        if self.cache is not None and not abstained:
            self.cache.set(key, answer)
            self.cache.add_similar(search_q, answer, **similar)
        return answer

    @staticmethod
    def _replay(
        hit: Answer,
        tier: Literal["exact", "semantic"],
        trace_id: str,
        timings: dict[str, float],
        start: float,
    ) -> Answer:
        timings["total_ms"] = (time.perf_counter() - start) * 1000.0
        return hit.model_copy(update={"cache_tier": tier, "trace_id": trace_id, "timings": timings})
