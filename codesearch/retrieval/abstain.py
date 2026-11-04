"""Decides when to answer "this isn't covered in the React docs".

What goes here:
    - ``AbstainDecision`` (dataclass): abstain (bool), reason, top_score.
    - ``decide(chunks: list[ScoredChunk], threshold: float) -> AbstainDecision``:
      abstain when there are no chunks or the best rerank score is below threshold.
    - ``ABSTAIN_MESSAGE``: the canned reply, pointing to react.dev search.

How to set the threshold: run eval on the out-of-scope questions in
eval/golden.jsonl and pick the value that abstains on >= 90% of them while
keeping in-scope answers; record it in config.

Must not:
    - call the LLM. Abstaining must be fast and free.

Tested by: tests/unit/test_abstain.py (empty list, below, at, above threshold).
Phase: 1
"""

from dataclasses import dataclass
from typing import Literal

from codesearch.schemas import ScoredChunk

ScoreKind = Literal["rerank", "dense"]

ABSTAIN_MESSAGE = (
    "I couldn't find this in the React docs, so I won't guess. "
    "Try searching https://react.dev directly."
)


@dataclass(frozen=True)
class AbstainDecision:
    """Whether to refuse, why, and the score the decision was based on."""

    abstain: bool
    reason: str | None  # "no_candidates" | "below_threshold" | None when answering
    top_score: float | None
    score_kind: ScoreKind | None


def top_signal(chunks: list[ScoredChunk]) -> tuple[ScoreKind, float] | None:
    """The best usable score: rerank if the chunks were reranked, else dense, else None.

    BM25 and RRF scores have no fixed scale, so a threshold on them means nothing.
    """
    # Every reranked chunk has a rerank_score; in hybrid mode BM25-only chunks have
    # no dense_score. "is not None" keeps a real score of 0.0.
    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if rerank_scores:
        return "rerank", float(max(rerank_scores))

    dense_scores = [c.dense_score for c in chunks if c.dense_score is not None]
    if dense_scores:
        return "dense", float(max(dense_scores))

    return None


def decide(chunks: list[ScoredChunk], threshold: float) -> AbstainDecision:
    """Abstain when nothing was retrieved or the best score is below ``threshold``."""
    # Checked before top_signal: an empty result is a normal refusal, not an error.
    if not chunks:
        return AbstainDecision(
            abstain=True, reason="no_candidates", top_score=None, score_kind=None
        )

    signal = top_signal(chunks)
    if signal is None:
        raise ValueError("no rerank or dense score to judge (BM25-only results?)")
    score_kind, top_score = signal

    # A score exactly at the threshold answers.
    if top_score < threshold:
        return AbstainDecision(
            abstain=True, reason="below_threshold", top_score=top_score, score_kind=score_kind
        )
    return AbstainDecision(abstain=False, reason=None, top_score=top_score, score_kind=score_kind)
