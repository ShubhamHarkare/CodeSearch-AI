"""Smoke test for eval/plot_reliability.py: a report in, a PNG out."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.plot_reliability import plot

REPORT = {
    "n": 4,
    "positive_rate": 0.5,
    "folds": 2,
    "ece": 0.1,
    "bins": [
        {"lower": 0.0, "upper": 0.5, "count": 2, "confidence": 0.2, "accuracy": 0.0},
        {"lower": 0.5, "upper": 1.0, "count": 2, "confidence": 0.8, "accuracy": 1.0},
    ],
}


def test_plot_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = plot(REPORT, tmp_path / "docs" / "reliability.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
