"""Tests for codesearch/retrieval/condense.py: turning a follow-up into a standalone question.

"what about objects?" can't be searched on its own; after a question about useState it
means "How do I update an object stored in useState?". condense() asks the LLM to
rewrite it, using the condense prompt and the recent chat history. Everything runs
with FakeLLM: no model needed.
"""

from __future__ import annotations

import pytest

from codesearch.errors import LLMTimeoutError, LLMUnavailableError
from codesearch.generation.llm import FakeLLM
from codesearch.generation.prompts import load_prompt
from codesearch.retrieval.condense import MAX_MESSAGE_CHARS, condense, format_history
from codesearch.schemas import ChatMessage

PROMPT = load_prompt("condense")


def user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def bot(text: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)


HISTORY = [
    user("How do I add state to a component?"),
    bot("Call useState at the top level of your component [1]."),
]


class FailingLLM(FakeLLM):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def generate(self, prompt: str, system: str | None = None) -> str:
        await super().generate(prompt, system)  # still records the call
        raise self.error


# =====================================================================
# ChatMessage
# =====================================================================


def test_chat_message_roles() -> None:
    assert user("hi").role == "user"
    assert bot("hello").role == "assistant"
    with pytest.raises(ValueError):
        ChatMessage(role="system", content="no")  # type: ignore[arg-type]


# =====================================================================
# format_history
# =====================================================================


def test_format_history_labels_each_message() -> None:
    assert format_history(HISTORY) == (
        "User: How do I add state to a component?\n"
        "Assistant: Call useState at the top level of your component [1]."
    )


def test_format_history_empty() -> None:
    assert format_history([]) == ""


def test_format_history_shortens_long_messages() -> None:
    """Long answers would crowd the prompt; the gist is enough to resolve a follow-up."""
    text = format_history([bot("x" * (MAX_MESSAGE_CHARS + 500))])
    assert text.startswith("Assistant: " + "x" * MAX_MESSAGE_CHARS)
    assert text.endswith("…")
    assert len(text) == len("Assistant: ") + MAX_MESSAGE_CHARS + 1


def test_format_history_keeps_newlines_out_of_the_transcript() -> None:
    text = format_history([bot("line one\nline two")])
    assert text == "Assistant: line one line two"


# =====================================================================
# condense
# =====================================================================


async def test_no_history_returns_the_question_without_calling_the_llm() -> None:
    llm = FakeLLM()
    assert await condense("How do I use useState?", [], llm, PROMPT) == "How do I use useState?"
    assert llm.prompts == []


async def test_follow_up_is_rewritten_by_the_llm() -> None:
    llm = FakeLLM(responses=["How do I update an object stored in useState?"])
    result = await condense("what about objects?", HISTORY, llm, PROMPT)
    assert result == "How do I update an object stored in useState?"


async def test_sends_the_rendered_condense_prompt_with_its_system_rules() -> None:
    llm = FakeLLM(responses=["rewritten?"])
    await condense("what about objects?", HISTORY, llm, PROMPT)
    (prompt,) = llm.prompts
    assert "User: How do I add state to a component?" in prompt
    assert "Follow-up question: what about objects?" in prompt
    assert llm.systems == [PROMPT.system]


async def test_only_the_last_turns_are_sent() -> None:
    """A turn is a user message plus the reply: max_turns=1 keeps the last two messages."""
    history = [user("old question"), bot("old answer"), user("recent q"), bot("recent a")]
    llm = FakeLLM(responses=["rewritten?"])
    await condense("and then?", history, llm, PROMPT, max_turns=1)
    assert "old question" not in llm.prompts[0]
    assert "recent q" in llm.prompts[0] and "recent a" in llm.prompts[0]


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  How do I update objects?  ", "How do I update objects?"),
        ('"How do I update objects?"', "How do I update objects?"),
        ("'How do I update objects?'", "How do I update objects?"),
        ("Standalone question: How do I update objects?", "How do I update objects?"),
        ("\n\nHow do I update objects?\nThis rewrites the follow-up.", "How do I update objects?"),
    ],
    ids=["spaces", "double-quotes", "single-quotes", "label", "extra-lines"],
)
async def test_cleans_up_the_model_output(raw: str, clean: str) -> None:
    """Small local models add quotes, labels or explanations despite the prompt."""
    llm = FakeLLM(responses=[raw])
    assert await condense("what about objects?", HISTORY, llm, PROMPT) == clean


@pytest.mark.parametrize("raw", ["", "   ", '""', "x" * 600])
async def test_unusable_output_falls_back_to_the_original(raw: str) -> None:
    llm = FakeLLM(responses=[raw])
    assert await condense("what about objects?", HISTORY, llm, PROMPT) == "what about objects?"


@pytest.mark.parametrize(
    "error",
    [LLMUnavailableError("ollama down"), LLMTimeoutError("too slow")],
    ids=["unavailable", "timeout"],
)
async def test_llm_failure_falls_back_to_the_original(error: Exception) -> None:
    """Retrieval with the raw follow-up beats failing the whole request (log a warning)."""
    llm = FailingLLM(error)
    result = await condense("what about objects?", HISTORY, llm, PROMPT)
    assert result == "what about objects?"
    assert len(llm.prompts) == 1


async def test_other_errors_are_not_swallowed() -> None:
    """Only LLM outages fall back; a bug (e.g. a broken prompt) must surface."""
    with pytest.raises(RuntimeError):
        await condense("q?", HISTORY, FailingLLM(RuntimeError("bug")), PROMPT)


async def test_max_turns_must_be_positive() -> None:
    with pytest.raises(ValueError):
        await condense("q?", HISTORY, FakeLLM(), PROMPT, max_turns=0)
