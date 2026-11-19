"""On-demand leave-one-out attribution: how much did each source drive the answer?

What goes here:
    - ``async leave_one_out(question, chunks, answer, llm, embedder, prompt)
      -> dict[str, float]``:
        * for each chunk i, regenerate the answer WITHOUT chunk i
        * score_i = 1 - cosine(embed(original_answer), embed(answer_without_i))
        * normalise so scores sum to 1; return {chunk_id: score}
    - Run regenerations concurrently (asyncio.gather) with a cap.

Cost: k extra LLM calls, so it only runs from
POST /v1/explain/{trace_id}/deep ("Deep explain" button), never by default.
Connects to your Attribution-Guided Masking paper: same question, asked at
the context level instead of inside the model.

Tested by: tests/unit/test_attribution.py with a FakeLLM whose answer depends
on which chunks are present.
Phase: 2
"""

import asyncio
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from codesearch.generation.prompts import PromptTemplate, format_context
from codesearch.interfaces import LLM, Embedder
from codesearch.schemas import ScoredChunk


def _unit(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


async def leave_one_out(
    question: str,
    chunks: Sequence[ScoredChunk],
    answer: str,
    llm: LLM,
    embedder: Embedder,
    prompt: PromptTemplate,
    *,
    max_concurrency: int = 2,
) -> dict[str, float]:
    """{chunk_id: share of the answer it drove}, in chunk order; shares sum to 1."""
    chunks = list(chunks)
    if not chunks:
        return {}
    if len(chunks) == 1:
        return {chunks[0].chunk.id: 1.0}  # the only source drove everything: no LLM call

    semaphore = asyncio.Semaphore(max_concurrency)

    async def answer_without(i: int) -> str:
        rest = chunks[:i] + chunks[i + 1 :]  # every source except number i
        rendered = prompt.render(context=format_context(rest), question=question)
        async with semaphore:  # wait for a free slot
            return await llm.generate(rendered, system=prompt.system)

    alternatives = await asyncio.gather(*(answer_without(i) for i in range(len(chunks))))

    original = _unit(embedder.embed_query(answer))
    changes = [
        max(0.0, 1.0 - float(original @ _unit(embedder.embed_query(alt)))) for alt in alternatives
    ]

    original = _unit(embedder.embed_query(answer))
    changes = [
        max(0.0, 1.0 - float(original @ _unit(embedder.embed_query(alt)))) for alt in alternatives
    ]

    total = sum(changes)
    if total <= 1e-12:  # nothing changed when any source was removed
        return {c.chunk.id: 0.0 for c in chunks}
    return {c.chunk.id: change / total for c, change in zip(chunks, changes, strict=True)}
