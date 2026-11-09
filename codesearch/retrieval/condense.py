"""Rewrites a follow-up question into a standalone one before retrieval.

What goes here:
    - ``async condense(question, history, llm, prompt, max_turns=3) -> str``:
        * empty history -> return the question unchanged (no LLM call)
        * otherwise render prompts/condense.yaml with the last ``max_turns``
          turns and return the model's single-line rewrite
        * on LLM failure, fall back to the original question and log a warning

Example: history about useState, question "what about in a class component?"
-> "How do I manage state in a React class component instead of useState?"

Tested by: tests/unit/test_condense.py with FakeLLM (no history -> no call;
history -> rewritten; LLM error -> original question).
Phase: 1
"""

from collections.abc import Sequence

from codesearch.errors import LLMTimeoutError, LLMUnavailableError
from codesearch.generation.prompts import PromptTemplate
from codesearch.interfaces import LLM
from codesearch.log import get_logger
from codesearch.schemas import ChatMessage

log = get_logger(__name__)

MAX_MESSAGE_CHARS = 500  # each history message is cut to this
MAX_QUESTION_CHARS = 500  # a longer "rewrite" is treated as rambling


def format_history(messages: Sequence[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        label = "User" if message.role == "user" else "Assistant"
        text = message.content.replace("\n", " ")
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "…"  # one ellipsis character

        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def clean_question(raw: str) -> str:
    """Tidy the model's rewrite; returns "" when nothing usable is left."""
    # 1. Take the first non-empty line and strip whitespace
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""

    cleaned = lines[0]

    # 2. Remove leading "Standalone question:" (case-insensitive) and strip again
    prefix = "standalone question:"
    if cleaned.lower().startswith(prefix):
        cleaned = cleaned[len(prefix) :].strip()

    # 3. Strip surrounding quotes and whitespace again
    return cleaned.strip("\"'").strip()


async def condense(
    question: str,
    history: Sequence[ChatMessage],
    llm: LLM,
    prompt: PromptTemplate,
    max_turns: int = 3,
) -> str:
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    if not history:
        return question

    # Get the last max_turns exchanges (user + assistant pairs)
    recent = list(history)[-2 * max_turns :]

    rendered = prompt.render(history=format_history(recent), question=question)

    try:
        raw = await llm.generate(rendered, system=prompt.system)
    except (LLMUnavailableError, LLMTimeoutError) as e:
        log.warning("condense.failed", error=str(e))
        return question

    # Clean the model's output; fall back to the user's own question if it's unusable
    cleaned = clean_question(raw)
    if not cleaned or len(cleaned) > MAX_QUESTION_CHARS:
        return question
    return cleaned
