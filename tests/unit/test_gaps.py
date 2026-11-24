"""Tests for codesearch/monitoring/gaps.py: clustering abstained questions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.monitoring.gaps import GapCluster, content_gaps


class TopicEmbedder:
    """Puts each question near its topic's axis, so the right clusters are obvious."""

    model_name = "topic"
    dim = 3
    TOPICS: ClassVar[dict[str, int]] = {"server": 0, "css": 1, "redux": 2}

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word, axis in self.TOPICS.items():
                if word in text.lower():
                    out[i, axis] = 1.0
            out[i] += 0.01 * (i % 3)  # a little noise, so points aren't identical
        return out

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed([text])[0]


SERVER = [f"How do server actions work, case {i}?" for i in range(5)]
CSS = [f"How do I center a div with css, try {i}?" for i in range(3)]


def test_two_topics_become_two_clusters_biggest_first() -> None:
    gaps = content_gaps(SERVER + CSS, TopicEmbedder(), k=2)
    assert [g.size for g in gaps] == [5, 3]
    assert all("server" in q for q in gaps[0].examples)
    assert all("css" in q for q in gaps[1].examples)


def test_examples_are_capped_and_unique() -> None:
    questions = SERVER + SERVER[:2]  # two questions asked twice
    [gap] = content_gaps(questions, TopicEmbedder(), k=1, examples=3)
    assert gap.size == 7  # repeats count: they show how often it's asked
    assert len(gap.examples) == 3
    assert len(set(gap.examples)) == 3


def test_no_questions_no_gaps() -> None:
    assert content_gaps([], TopicEmbedder()) == []


def test_k_never_exceeds_the_distinct_questions() -> None:
    """Six copies of one question are one cluster, even with k=5."""
    gaps = content_gaps(["What is RSC?"] * 6, TopicEmbedder(), k=5)
    assert gaps == [GapCluster(size=6, examples=["What is RSC?"])]


def test_every_question_lands_in_exactly_one_cluster() -> None:
    questions = SERVER + CSS + ["Should I use redux or context?"] * 2
    gaps = content_gaps(questions, TopicEmbedder(), k=3)
    assert sum(g.size for g in gaps) == len(questions)


@pytest.mark.parametrize(("k", "examples"), [(0, 5), (5, 0)])
def test_k_and_examples_must_be_positive(k: int, examples: int) -> None:
    with pytest.raises(ValueError):
        content_gaps(SERVER, TopicEmbedder(), k=k, examples=examples)
