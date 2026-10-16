"""Turns raw scores into a confidence you can trust, and checks it.

What goes here:
    - ``features(answer, chunks) -> list[float]``: top rerank score, mean of
      top-3 rerank scores, groundedness.
    - ``Calibrator.fit(X, y)``: logistic regression where y = 1 if the answer
      was correct on the golden set (retrieval hit + grounded).
    - ``Calibrator.confidence(x) -> float`` and ``level -> "high"|"medium"|"low"``.
    - ``save(path)`` / ``load(path)`` as JSON (coefficients only, no pickle).
    - ``reliability_bins(probs, labels, n_bins=10)`` and expected calibration
      error, for the README reliability diagram.

Tested by: tests/unit/test_calibration.py (perfectly separable toy data;
save/load round-trip; ECE of a perfect calibrator is ~0).
Phase: 2
"""
