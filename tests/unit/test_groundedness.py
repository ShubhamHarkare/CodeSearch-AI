"""Tests for codesearch/explain/groundedness.py: is each answer sentence supported
by the chunk it cites?

The NLI model is replaced by a fake scorer, so no model download is needed. One
test (needs_model) runs the real cross-encoder when it's installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from codesearch.explain.groundedness import (
    DEFAULT_NLI_MODEL,
    NLIScorer,
    assess,
    is_connective,
    split_sentences,
)
from codesearch.schemas import Chunk, Citation, ScoredChunk

# =====================================================================
# split_sentences
# =====================================================================


def test_splits_on_sentence_ends_and_keeps_citations_attached() -> None:
    text = "Use state to remember values [1]. It returns a pair [2]."
    assert split_sentences(text) == ["Use state to remember values [1].", "It returns a pair [2]."]


def test_a_citation_after_the_period_stays_with_its_sentence() -> None:
    assert split_sentences("React batches updates. [3] Then it re-renders.") == [
        "React batches updates. [3]",
        "Then it re-renders.",
    ]


def test_question_and_exclamation_marks_end_sentences() -> None:
    assert split_sentences("Why? Because it is pure [1]! Done.") == [
        "Why?",
        "Because it is pure [1]!",
        "Done.",
    ]


def test_fenced_code_blocks_are_skipped() -> None:
    text = "Do this [1]:\n```js\nconst x = a.b;\nsetCount(c => c + 1);\n```\nThen render [2]."
    assert split_sentences(text) == ["Do this [1]:", "Then render [2]."]


def test_inline_code_is_never_split() -> None:
    text = "Call `obj.setState()` to update it [1]. Then read `ref.current` [2]."
    assert split_sentences(text) == [
        "Call `obj.setState()` to update it [1].",
        "Then read `ref.current` [2].",
    ]


def test_each_bullet_is_its_own_sentence_without_the_bullet_marker() -> None:
    text = (
        "Key points:\n\n*   Call it at the top level [5].\n- Avoid loops [2]\n1. Keep it pure [1]."
    )
    assert split_sentences(text) == [
        "Key points:",
        "Call it at the top level [5].",
        "Avoid loops [2]",
        "Keep it pure [1].",
    ]


def test_bold_markers_do_not_split_or_disappear() -> None:
    assert split_sentences("Pass an **updater function** instead [1].") == [
        "Pass an **updater function** instead [1]."
    ]


def test_decimals_do_not_end_a_sentence() -> None:
    assert split_sentences("React 18.2 added batching [1]. Yes.") == [
        "React 18.2 added batching [1].",
        "Yes.",
    ]


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "```\ncode only\n```"])
def test_nothing_to_split(text: str) -> None:
    assert split_sentences(text) == []


# =====================================================================
# is_connective
# =====================================================================


@pytest.mark.parametrize("sentence", ["Key points:", "Here's how:", "For example:"])
def test_lead_ins_ending_in_a_colon_are_connective(sentence: str) -> None:
    assert is_connective(sentence)


@pytest.mark.parametrize(
    "sentence",
    ["React re-renders on every state change.", "Use an updater [1]:", "Here's how it works."],
)
def test_claims_are_not_connective(sentence: str) -> None:
    assert not is_connective(sentence)


# =====================================================================
# assess
# =====================================================================


def chunk(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            url=f"https://react.dev/{chunk_id}",
            anchor="",
            title=chunk_id,
            header_path=(chunk_id,),
            text=text,
            source_path=f"{chunk_id}.md",
            start_line=1,
        ),
        dense_score=0.9,
    )


def cite(number: int, chunk_id: str) -> Citation:
    return Citation(
        number=number, chunk_id=chunk_id, url=f"https://react.dev/{chunk_id}", title=chunk_id
    )


CHUNKS = [
    chunk("c1", "useState returns a pair: the current state and a setter function."),
    chunk("c2", "Calling the setter with an updater function queues the update."),
]
CITATIONS = [cite(1, "c1"), cite(2, "c2")]


class FakeScorer:
    """Scores (premise, hypothesis) pairs with a rule; records every call."""

    def __init__(self, rule: Callable[[str, str], float] | float = 0.9) -> None:
        self.rule = rule
        self.calls: list[list[tuple[str, str]]] = []

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        if callable(self.rule):
            return [self.rule(p, h) for p, h in pairs]
        return [float(self.rule)] * len(pairs)


def test_every_cited_sentence_is_scored_against_its_chunk() -> None:
    scorer = FakeScorer(0.9)
    sentences, groundedness = assess(
        "It returns a pair [1]. Use an updater [2].", CITATIONS, CHUNKS, scorer
    )
    assert [s.label for s in sentences] == ["supported", "supported"]
    assert [s.cited_chunk_ids for s in sentences] == [["c1"], ["c2"]]
    assert groundedness == 1.0
    premises = [p for p, _ in scorer.calls[0]]
    assert premises == [CHUNKS[0].chunk.text, CHUNKS[1].chunk.text]


def test_the_hypothesis_has_the_citation_markers_removed() -> None:
    scorer = FakeScorer(0.9)
    assess("It returns a pair [1].", CITATIONS, CHUNKS, scorer)
    assert scorer.calls[0][0][1] == "It returns a pair."


def test_all_pairs_go_to_the_model_in_one_batch() -> None:
    scorer = FakeScorer(0.9)
    assess("A [1]. B [2]. C [1, 2].", CITATIONS, CHUNKS, scorer)
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 4  # A->c1, B->c2, C->c1, C->c2


def test_labels_follow_the_thresholds() -> None:
    scores = {"High.": 0.85, "Middle.": 0.5, "Low.": 0.1}
    scorer = FakeScorer(lambda _premise, hypothesis: scores[hypothesis])
    sentences, groundedness = assess("High [1]. Middle [1]. Low [1].", CITATIONS, CHUNKS, scorer)
    assert [s.label for s in sentences] == ["supported", "partial", "unsupported"]
    assert [s.score for s in sentences] == [0.85, 0.5, 0.1]
    assert groundedness == pytest.approx(1 / 3)  # only "supported" counts


def test_scores_exactly_at_a_threshold_reach_that_label() -> None:
    scorer = FakeScorer(lambda _p, h: {"A.": 0.7, "B.": 0.4}[h])
    sentences, _ = assess("A [1]. B [1].", CITATIONS, CHUNKS, scorer)
    assert [s.label for s in sentences] == ["supported", "partial"]


def test_custom_thresholds() -> None:
    sentences, _ = assess(
        "A [1].", CITATIONS, CHUNKS, FakeScorer(0.5), supported_at=0.45, partial_at=0.2
    )
    assert sentences[0].label == "supported"


def test_a_sentence_citing_two_chunks_takes_the_best_score() -> None:
    scorer = FakeScorer(lambda premise, _h: 0.9 if "updater" in premise else 0.2)
    sentences, _ = assess("Queue it [1, 2].", CITATIONS, CHUNKS, scorer)
    assert sentences[0].cited_chunk_ids == ["c1", "c2"]
    assert sentences[0].score == 0.9
    assert sentences[0].label == "supported"


def test_an_uncited_claim_is_unsupported_without_asking_the_model() -> None:
    scorer = FakeScorer(0.9)
    sentences, groundedness = assess(
        "React is the best library. It returns a pair [1].", CITATIONS, CHUNKS, scorer
    )
    assert sentences[0].label == "unsupported"
    assert sentences[0].score == 0.0
    assert sentences[0].cited_chunk_ids == []
    assert len(scorer.calls[0]) == 1  # only the cited sentence was scored
    assert groundedness == 0.5


def test_connective_lead_ins_are_left_out() -> None:
    sentences, groundedness = assess(
        "Key points:\n- It returns a pair [1].", CITATIONS, CHUNKS, FakeScorer(0.9)
    )
    assert [s.sentence for s in sentences] == ["It returns a pair [1]."]
    assert groundedness == 1.0


def test_a_citation_to_an_unknown_number_counts_as_uncited() -> None:
    """strip_invalid normally removes these; assess must still not crash."""
    sentences, _ = assess("Something [7].", CITATIONS, CHUNKS, FakeScorer(0.9))
    assert sentences[0].label == "unsupported"
    assert sentences[0].cited_chunk_ids == []


def test_nothing_to_judge_gives_none() -> None:
    scorer = FakeScorer(0.9)
    assert assess("", CITATIONS, CHUNKS, scorer) == ([], None)
    assert assess("Here's how:", CITATIONS, CHUNKS, scorer) == ([], None)
    assert scorer.calls == []  # no model call for nothing


def test_a_scorer_returning_the_wrong_number_of_scores_is_an_error() -> None:
    class Broken(FakeScorer):
        def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [0.9]

    with pytest.raises(ValueError):
        assess("A [1]. B [2].", CITATIONS, CHUNKS, Broken())


# =====================================================================
# NLIScorer (the real model, with predict() faked)
# =====================================================================


def logits(*rows: tuple[float, float, float]) -> np.ndarray[Any, Any]:
    return np.array(rows, dtype=np.float32)


def test_nli_scorer_returns_the_softmax_entailment_probability() -> None:
    # nli-deberta-v3 label order: contradiction, entailment, neutral
    scorer = NLIScorer(predict=lambda pairs: logits((0.0, 10.0, 0.0), (10.0, 0.0, 0.0)))
    scores = scorer.entailment([("p", "h1"), ("p", "h2")])
    assert scores[0] == pytest.approx(1.0, abs=1e-3)
    assert scores[1] == pytest.approx(0.0, abs=1e-3)
    assert all(isinstance(s, float) for s in scores)


def test_nli_scorer_equal_logits_give_one_third() -> None:
    scorer = NLIScorer(predict=lambda pairs: logits((1.0, 1.0, 1.0)))
    assert scorer.entailment([("p", "h")]) == [pytest.approx(1 / 3)]


def test_nli_scorer_passes_premise_then_hypothesis() -> None:
    seen: list[list[tuple[str, str]]] = []

    def predict(pairs: list[tuple[str, str]]) -> np.ndarray[Any, Any]:
        seen.append(pairs)
        return logits(*[(0.0, 0.0, 0.0)] * len(pairs))

    NLIScorer(predict=predict).entailment([("the chunk", "the sentence")])
    assert seen == [[("the chunk", "the sentence")]]


def test_nli_scorer_with_no_pairs_never_loads_the_model() -> None:
    def explode(pairs: list[tuple[str, str]]) -> np.ndarray[Any, Any]:
        raise AssertionError("should not be called")

    assert NLIScorer(predict=explode).entailment([]) == []


def test_nli_scorer_does_not_load_anything_when_constructed() -> None:
    scorer = NLIScorer()  # no download here: the model loads on first use
    assert scorer.model_name == DEFAULT_NLI_MODEL


@pytest.mark.needs_model
def test_real_nli_model_tells_support_from_contradiction() -> None:
    pytest.importorskip("sentence_transformers")
    scorer = NLIScorer()
    premise = "useState returns an array with exactly two items: the current state and a setter."
    supported, contradicted = scorer.entailment(
        [
            (premise, "useState returns the current state and a setter."),
            (premise, "useState returns three values."),
        ]
    )
    assert supported > 0.7
    assert contradicted < 0.3
