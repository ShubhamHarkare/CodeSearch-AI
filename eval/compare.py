"""The CI quality gate: compares a run against the committed baseline.

What goes here:
    - ``compare(result, baseline, tolerance_points=2.0) -> Diff`` listing each
      metric's old value, new value and change.
    - ``to_markdown(diff) -> str`` for the PR comment.
    - ``main()``: exit code 1 if Recall@5 or MRR drops more than the tolerance,
      0 otherwise. ``--update-baseline`` writes baseline.json (only on purpose).

Used by: .github/workflows/eval-gate.yml
Tested by: tests/unit/test_compare.py (within tolerance passes; drop fails).
Phase: 1 (local), 4 (CI)
"""
