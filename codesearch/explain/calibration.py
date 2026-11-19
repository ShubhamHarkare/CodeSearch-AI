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

import heapq
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from codesearch.schemas import ScoredChunk

FEATURE_NAMES = ("top_score", "mean_top3", "groundedness")


def features(chunks: Sequence[ScoredChunk], groundedness: float | None) -> list[float]:
    """[top_score, mean_top3, groundedness] for the chunks sent to the LLM."""
    g = 0.0 if groundedness is None else groundedness
    if not chunks:
        return [0.0, 0.0, g]
    use_rerank = any(c.rerank_score is not None for c in chunks)  # same rule as top_signal
    scores = [c.rerank_score if use_rerank else c.dense_score for c in chunks]
    best = heapq.nlargest(3, (s for s in scores if s is not None))  # top 3, largest first
    if not best:
        return [0.0, 0.0, g]
    return [best[0], sum(best) / len(best), g]


@dataclass(frozen=True)
class Calibrator:
    coef: list[float]
    intercept: float
    high_at: float = 0.75
    low_at: float = 0.4
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(
        cls, X: Sequence[Sequence[float]], y: Sequence[int], *, c: float = 10.0
    ) -> "Calibrator":
        """Learn the weights with sklearn, then keep only the numbers (no sklearn object)."""
        from sklearn.linear_model import LogisticRegression  # only fitting needs sklearn

        if len(X) != len(y):
            raise ValueError(f"got {len(X)} feature rows but {len(y)} labels")
        if len(set(y)) < 2:
            raise ValueError("need both good and bad answers to fit")
        logreg = LogisticRegression(C=c).fit(X, y)
        return cls(
            coef=[float(w) for w in logreg.coef_[0]],  # one weight per feature
            intercept=float(logreg.intercept_[0]),
        )

    def confidence(self, x: Sequence[float]) -> float:
        """Probability that the answer is good: sigmoid(intercept + coef · x)."""
        if len(x) != len(self.coef):
            raise ValueError(f"expected {len(self.coef)} features, got {len(x)}")

        # Step 1: the linear score (exactly what sklearn computes inside predict_proba)
        z = self.intercept + sum(w * v for w, v in zip(self.coef, x, strict=True))

        # Step 2: sigmoid, written so math.exp never overflows on very negative z
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def level(self, confidence: float) -> Literal["high", "medium", "low"]:
        """The badge shown in the UI."""
        if confidence >= self.high_at:
            return "high"
        if confidence >= self.low_at:
            return "medium"
        return "low"

    def save(self, path: Path) -> None:
        """Write the coefficients as JSON: readable, and safe to load (no pickle)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)  # a tuple isn't JSON; a list is
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        """Read a saved calibrator; refuse one fitted on a different feature list."""
        data = json.loads(path.read_text())
        names = tuple(data.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(
                f"{path} was fitted on features {names}, but the code now uses "
                f"{FEATURE_NAMES}: refit it with eval/fit_calibrator.py"
            )
        return cls(
            coef=[float(w) for w in data["coef"]],
            intercept=float(data["intercept"]),
            high_at=float(data.get("high_at", 0.75)),
            low_at=float(data.get("low_at", 0.4)),
        )


@dataclass(frozen=True)
class Bin:
    """One bar of the reliability diagram."""

    lower: float
    upper: float
    count: int
    confidence: float  # mean predicted probability in the bin
    accuracy: float  # share of good answers in the bin


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> list[Bin]:
    """Group predictions into equal-width confidence bins; skip empty ones.

    A calibrated model has confidence == accuracy in every bin.
    """
    if len(probs) != len(labels):
        raise ValueError(f"got {len(probs)} probabilities but {len(labels)} labels")
    groups: dict[int, list[tuple[float, int]]] = {}
    for p, label in zip(probs, labels, strict=True):
        index = min(int(p * n_bins), n_bins - 1)  # p == 1.0 belongs to the last bin
        groups.setdefault(index, []).append((p, label))
    return [
        Bin(
            lower=i / n_bins,
            upper=(i + 1) / n_bins,
            count=len(pairs),
            confidence=sum(p for p, _ in pairs) / len(pairs),
            accuracy=sum(label for _, label in pairs) / len(pairs),
        )
        for i, pairs in sorted(groups.items())
    ]


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """Average gap between confidence and accuracy, weighted by how full each bin is."""
    if not probs:
        raise ValueError("no predictions to check")
    n = len(probs)
    return sum(
        b.count / n * abs(b.accuracy - b.confidence)
        for b in reliability_bins(probs, labels, n_bins)
    )
