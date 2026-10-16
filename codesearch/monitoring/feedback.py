"""Stores thumbs up / down on answers and turns them into eval data.

What goes here:
    - ``FeedbackStore(db_path)`` (SQLite): ``add(trace_id, vote, comment)``,
      ``summary(since) -> dict``, ``thumbs_down(since) -> list``.
    - ``export_candidates(path)``: writes thumbs-down questions + their traces
      as candidate rows for eval/golden.jsonl (you review them by hand).

Tested by: tests/unit/test_feedback.py.
Phase: 3
"""
