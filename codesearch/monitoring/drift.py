"""Detects when what users ask drifts away from what the system was tested on.

What goes here:
    - ``reference_centroid(golden_question_vectors) -> np.ndarray``, computed
      once at startup (or saved alongside the index).
    - ``drift_distance(window_vectors, reference) -> float``: cosine distance
      between the window's query centroid and the reference.
    - ``low_score_share(window, threshold) -> float``: share of recent queries
      whose best rerank score is under the abstain threshold.
    - Threshold for alerts: replay the golden set to get a baseline distance,
      then set the alert at baseline + margin.

Tested by: tests/unit/test_drift.py (same distribution ~0; shifted > baseline).
Phase: 3
"""
