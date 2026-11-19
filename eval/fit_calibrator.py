"""Fits the confidence calibrator on answer-eval results, and checks it.

    uv run python -m eval.answer_eval eval/configs/dense.yaml        # all questions first
    uv run python -m eval.fit_calibrator results/answers_dense.json

Learns, from the golden-set answers, how the features (top retrieval score, mean of the
top 3, groundedness) predict a GOOD answer, saves the weights to models/calibrator.json
(plain JSON), and reports how well calibrated it is: a reliability table and the
expected calibration error (ECE), both measured out-of-fold so the model is never
checked on the questions it was fitted on.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from codesearch.explain.calibration import (
    Calibrator,
    expected_calibration_error,
    reliability_bins,
)

DEFAULT_OUT = Path("models/calibrator.json")
C = 10.0  # same regularisation as Calibrator.fit


def is_good(record: dict[str, Any]) -> bool:
    """The label: it cites a right page AND at least half its sentences are supported."""
    grounded = record.get("groundedness")
    return record.get("cited_relevant") is True and grounded is not None and grounded >= 0.5


def dataset(records: list[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
    """X, y from answered in-scope questions that have features (others show no confidence)."""
    usable = [
        r
        for r in records
        if r["expected"] == "answer" and not r["abstained"] and r.get("features") is not None
    ]
    return [list(r["features"]) for r in usable], [int(is_good(r)) for r in usable]


def run(answers: Path, out: Path, *, n_bins: int = 5, folds: int = 5) -> dict[str, Any]:
    """Fit on every usable record, save, and report out-of-fold calibration."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    X, y = dataset(json.loads(answers.read_text())["rows"])
    if len(set(y)) < 2:
        raise ValueError("need both good and bad answers to fit (run the full answer eval)")
    folds = min(folds, y.count(0), y.count(1))  # each fold needs both kinds

    # Each question's probability comes from a model fitted WITHOUT that question.
    out_of_fold = cross_val_predict(
        LogisticRegression(C=C), X, y, cv=folds, method="predict_proba"
    )[:, 1]
    probs = [float(p) for p in out_of_fold]

    Calibrator.fit(X, y, c=C).save(out)
    return {
        "n": len(y),
        "positive_rate": sum(y) / len(y),
        "folds": folds,
        "ece": expected_calibration_error(probs, y, n_bins),
        "bins": [asdict(b) for b in reliability_bins(probs, y, n_bins)],
    }


def main(
    answers: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option(help="Where to save the calibrator.")] = DEFAULT_OUT,
    report: Annotated[
        Path, typer.Option(help="Where to save the calibration report (for the README plot).")
    ] = Path("results/calibration.json"),
    n_bins: Annotated[int, typer.Option(help="Reliability bins (small data: keep it small).")] = 5,
) -> None:
    """Fit the calibrator and print how well calibrated it is."""
    result = run(answers, out, n_bins=n_bins)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    typer.echo(
        f"fitted on {result['n']} answers ({result['positive_rate']:.0%} good), "
        f"{result['folds']}-fold out-of-fold check"
    )
    typer.echo(f"ECE {result['ece']:.3f}   (0 = perfectly calibrated)")
    typer.echo("  bin         n   confidence  accuracy")
    for b in result["bins"]:
        typer.echo(
            f"  {b['lower']:.1f}-{b['upper']:.1f}  {b['count']:>3}   "
            f"{b['confidence']:.2f}        {b['accuracy']:.2f}"
        )
    typer.echo(f"saved -> {out}  (report -> {report})")


if __name__ == "__main__":
    typer.run(main)
