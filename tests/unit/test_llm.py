"""Tests for FakeLLM in codesearch/generation/llm.py."""

import pytest

from codesearch.generation.llm import FakeLLM
from codesearch.interfaces import LLM

# =====================================================================
# FakeLLM Tests
# =====================================================================


@pytest.mark.asyncio
async def test_fake_llm_default_response_and_prompt_recording():
    fake = FakeLLM()
    prompt = "Explain React state"

    response = await fake.generate(prompt)

    assert "[1]" in response and "[2]" in response
    assert fake.prompts == [prompt]


@pytest.mark.asyncio
async def test_fake_llm_canned_responses_and_fallback():
    canned = ["Response 1", "Response 2"]
    fake = FakeLLM(responses=canned)

    res1 = await fake.generate("Prompt 1")
    res2 = await fake.generate("Prompt 2")
    res3 = await fake.generate("Prompt 3")  # Exhausted, fallback to default

    assert res1 == "Response 1"
    assert res2 == "Response 2"
    assert res3 == fake.DEFAULT_ANSWER
    assert fake.prompts == ["Prompt 1", "Prompt 2", "Prompt 3"]


@pytest.mark.asyncio
async def test_fake_llm_streaming_yields_full_text():
    fake = FakeLLM(responses=["Hello world test"])
    tokens = []

    async for token in fake.stream("Test prompt"):
        tokens.append(token)

    full_text = "".join(tokens)
    assert full_text == "Hello world test"
    assert fake.prompts == ["Test prompt"]


@pytest.mark.asyncio
async def test_fake_llm_edge_cases():
    fake = FakeLLM(responses=[""])

    # 1. Empty canned response
    res = await fake.generate("test empty")
    assert res == ""

    # 2. Empty prompt
    res_empty_prompt = await fake.generate("")
    assert res_empty_prompt == fake.DEFAULT_ANSWER
    assert fake.prompts[-1] == ""


def test_fake_llm_satisfies_llm_protocol():
    assert isinstance(FakeLLM(), LLM)


async def test_fake_llm_instances_do_not_share_state():
    first, second = FakeLLM(), FakeLLM()
    await first.generate("only in first")
    assert second.prompts == []
