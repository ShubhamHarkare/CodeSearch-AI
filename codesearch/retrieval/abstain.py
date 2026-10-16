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
