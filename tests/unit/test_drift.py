"""Tests for codesearch/monitoring/drift.py: query drift against the golden set."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.index.embedder import HashEmbedder
from codesearch.monitoring.drift import (
    DriftTracker,
    centroid,
    cosine_distance,
    reference_questions,
)
from codesearch.monitoring.windows import RollingWindow
from codesearch.schemas import RequestSignals

REACT = [
    "How do I update state based on the previous state?",
    "When should I use useEffect?",
    "How do I pass props to a component?",
    "How do I render a list with keys?",
    "What does useMemo do?",
]
OFF_TOPIC = [
    "How do I bake sourdough bread at home?",
    "Best hiking trails near Ann Arbor?",
    "How do I change a car tyre safely?",
    "What wine goes with salmon?",
    "How do I train for a marathon?",
]


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.embedded.extend(texts)
        return super().embed(texts)


def window_of(questions: list[str], start: int = 0) -> RollingWindow:
    window = RollingWindow()
    for i, q in enumerate(questions, start=start):
        window.record(
            RequestSignals(
                trace_id=f"t{i}",
                timestamp=datetime.now(UTC),
                query=q,
                retrieval_mode="dense",
                latency_ms=100.0,
                abstained=False,
            )
        )
    return window


# =====================================================================
# cosine_distance and centroid
# =====================================================================


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [([1, 0], [1, 0], 0.0), ([1, 0], [0, 1], 1.0), ([1, 0], [-1, 0], 2.0), ([1, 0], [3, 0], 0.0)],
)
def test_cosine_distance(a: list[float], b: list[float], expected: float) -> None:
    assert cosine_distance(np.array(a), np.array(b)) == pytest.approx(expected)


def test_cosine_distance_of_a_zero_vector_is_undefined() -> None:
    with pytest.raises(ValueError):
        cosine_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_centroid_weights_every_question_equally() -> None:
    """A long question must not count more than a short one: rows are normalised first."""
    c = centroid(np.array([[1.0, 0.0], [0.0, 10.0]]))
    assert c == pytest.approx([0.5, 0.5])


def test_centroid_needs_vectors() -> None:
    with pytest.raises(ValueError):
        centroid(np.zeros((0, 4)))


# =====================================================================
# reference_questions
# =====================================================================


def test_reference_is_the_in_scope_golden_questions(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    rows = [
        {"question": "How do I use state?", "expected": "answer"},
        {"question": "What is the capital of France?", "expected": "abstain"},
        {"question": "When does an effect run?", "expected": "answer"},
    ]
    golden.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert reference_questions(golden) == ["How do I use state?", "When does an effect run?"]


def test_a_missing_golden_file_gives_no_reference(tmp_path: Path) -> None:
    assert reference_questions(tmp_path / "nope.jsonl") == []


def test_the_real_golden_set_has_55_reference_questions() -> None:
    assert len(reference_questions(Path("eval/golden.jsonl"))) == 55


# =====================================================================
# DriftTracker
# =====================================================================


def test_traffic_like_the_reference_has_almost_no_drift() -> None:
    tracker = DriftTracker(HashEmbedder(), REACT, min_questions=5)
    distance = tracker.distance(window_of(REACT))
    assert distance is not None and distance == pytest.approx(0.0, abs=1e-6)


def test_off_topic_traffic_drifts_further_than_react_traffic() -> None:
    tracker = DriftTracker(HashEmbedder(), REACT, min_questions=5)
    # distinct trace ids: the tracker caches embeddings by trace id (unique per request)
    similar = tracker.distance(window_of([*REACT[:3], "How do I use useState?", "What is JSX?"]))
    shifted = tracker.distance(window_of(OFF_TOPIC, start=100))
    assert similar is not None and shifted is not None
    assert shifted > similar


def test_too_few_questions_give_no_reading() -> None:
    """A centroid of 3 questions is noise, not a trend: report "no data"."""
    tracker = DriftTracker(HashEmbedder(), REACT, min_questions=20)
    assert tracker.distance(window_of(REACT)) is None


def test_each_question_is_embedded_only_once() -> None:
    embedder = CountingEmbedder()
    tracker = DriftTracker(embedder, REACT, min_questions=5)
    embedder.embedded.clear()  # ignore the reference
    window = window_of(OFF_TOPIC)
    tracker.distance(window)
    assert len(embedder.embedded) == 5
    tracker.distance(window)  # a second scrape, nothing new
    assert len(embedder.embedded) == 5
    window.record(window_of(["One more question?"], start=99).snapshot()[0])
    tracker.distance(window)
    assert embedder.embedded[-1] == "One more question?"
    assert len(embedder.embedded) == 6


def test_the_cache_forgets_questions_that_left_the_window() -> None:
    tracker = DriftTracker(HashEmbedder(), REACT, min_questions=1)
    window = RollingWindow(maxlen=3)
    for s in window_of(REACT).snapshot():
        window.record(s)
        tracker.distance(window)
    assert tracker.cached() == 3  # bounded by the window, not by all traffic ever


def test_a_tracker_needs_reference_questions() -> None:
    with pytest.raises(ValueError):
        DriftTracker(HashEmbedder(), [])
