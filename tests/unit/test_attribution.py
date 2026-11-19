"""Tests for codesearch/explain/attribution.py: which source actually drove the answer?

Leave-one-out: regenerate the answer once per source with that source removed, and
measure how much the answer changes (1 - cosine of the answer embeddings). A source the
answer depended on changes it a lot; an ignored one changes nothing.

The LLM is faked so its answer depends on which sources are in the prompt, and the
embedder is the bag-of-words HashEmbedder: no model needed.
"""

from __future__ import annotations

import asyncio

import pytest

from codesearch.errors import LLMUnavailableError
from codesearch.explain.attribution import leave_one_out
from codesearch.generation.prompts import load_prompt
from codesearch.index.embedder import HashEmbedder
from codesearch.schemas import Chunk, ScoredChunk

# What each source contributes to the answer when it's in the prompt.
CONTRIBUTES = {
    "ALPHA": "pass an updater function to the setter so updates queue in order",  # the core
    "BETA": "effects run after render",  # a side point
    "GAMMA": "",  # ignored: removing it changes nothing
}


def chunk(name: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=name.lower(),
            url=f"https://react.dev/{name.lower()}",
            anchor="",
            title=name,
            header_path=(name,),
            text=f"{name} source text",
            source_path="x.md",
            start_line=1,
        ),
        dense_score=0.9,
    )


CHUNKS = [chunk("ALPHA"), chunk("BETA"), chunk("GAMMA")]


class ContextLLM:
    """Answers with the contributions of whichever sources appear in the prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        parts = [text for name, text in CONTRIBUTES.items() if f"{name} source text" in prompt]
        return " ".join(p for p in parts if p) or "not covered"

    async def stream(self, prompt: str, system: str | None = None):  # type: ignore[no-untyped-def]
        yield await self.generate(prompt, system)


PROMPT = load_prompt("answer")
FULL_ANSWER = f"{CONTRIBUTES['ALPHA']} {CONTRIBUTES['BETA']}"


async def run(
    llm: ContextLLM | None = None, chunks: list[ScoredChunk] = CHUNKS, **kw: int
) -> dict[str, float]:
    return await leave_one_out(
        "How do I update state?",
        chunks,
        FULL_ANSWER,
        llm or ContextLLM(),
        HashEmbedder(),
        PROMPT,
        **kw,
    )


async def test_scores_cover_every_source_in_order_and_sum_to_one() -> None:
    scores = await run()
    assert list(scores) == ["alpha", "beta", "gamma"]
    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(s >= 0 for s in scores.values())


async def test_the_source_the_answer_depends_on_scores_highest() -> None:
    scores = await run()
    assert scores["alpha"] > scores["beta"] > scores["gamma"]


async def test_an_ignored_source_scores_zero() -> None:
    assert (await run())["gamma"] == pytest.approx(0.0, abs=1e-6)


async def test_one_llm_call_per_source_each_without_that_source() -> None:
    llm = ContextLLM()
    await run(llm)
    assert len(llm.prompts) == 3
    for c in CHUNKS:
        missing_from = [p for p in llm.prompts if c.chunk.text not in p]
        assert len(missing_from) == 1  # left out of exactly one regeneration


async def test_the_remaining_sources_are_renumbered_from_one() -> None:
    """Without ALPHA, BETA is the first excerpt: [1], not [2] (no gaps for the model)."""
    llm = ContextLLM()
    await run(llm)
    without_alpha = next(p for p in llm.prompts if "ALPHA source text" not in p)
    assert "[1] https://react.dev/beta" in without_alpha
    assert "[3]" not in without_alpha


async def test_the_regenerations_use_the_answer_prompt_and_its_system_message() -> None:
    llm = ContextLLM()
    await run(llm)
    assert all("How do I update state?" in p for p in llm.prompts)
    assert llm.systems == [PROMPT.system] * 3


async def test_a_single_source_is_fully_responsible_without_any_llm_call() -> None:
    llm = ContextLLM()
    assert await run(llm, chunks=[CHUNKS[0]]) == {"alpha": 1.0}
    assert llm.prompts == []


async def test_no_sources_no_scores() -> None:
    llm = ContextLLM()
    assert await run(llm, chunks=[]) == {}
    assert llm.prompts == []


async def test_when_nothing_changes_every_score_is_zero() -> None:
    """No source mattered: report zeros, never divide by zero."""

    class StubbornLLM(ContextLLM):
        async def generate(self, prompt: str, system: str | None = None) -> str:
            self.prompts.append(prompt)
            return FULL_ANSWER

    scores = await run(StubbornLLM())
    assert scores == {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}


async def test_regenerations_run_concurrently_up_to_the_cap() -> None:
    class SlowLLM(ContextLLM):
        def __init__(self) -> None:
            super().__init__()
            self.running = 0
            self.peak = 0

        async def generate(self, prompt: str, system: str | None = None) -> str:
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.01)
            self.running -= 1
            return await super().generate(prompt, system)

    chunks = [*CHUNKS, chunk("DELTA")]
    llm = SlowLLM()
    await run(llm, chunks=chunks, max_concurrency=2)
    assert llm.peak == 2  # concurrent, but never more than the cap


async def test_an_llm_failure_propagates() -> None:
    class DownLLM(ContextLLM):
        async def generate(self, prompt: str, system: str | None = None) -> str:
            raise LLMUnavailableError("ollama is not running")

    with pytest.raises(LLMUnavailableError):
        await run(DownLLM())
