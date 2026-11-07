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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer

GATES: dict[str, tuple[Literal["higher", "lower"], float]] = {
    "recall_at_5": ("higher", 0.02),
    "mrr": ("higher", 0.02),
    "page_recall_at_5": ("higher", 0.02),
    "abstain_recall": ("higher", 0.05),
    "false_abstain_rate": ("lower", 0.05),
}


@dataclass(frozen=True)
class MetricCheck:
    name: str
    baseline: float
    current: float | None  # None = missing from the new run
    delta: float | None  # current - baseline
    direction: str  # "higher" | "lower"
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class Comparison:
    checks: list[MetricCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)  # all checks passed


def compare(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    gates: Mapping[str, tuple[str, float]] = GATES,
) -> Comparison:
    """One check per gated metric: does ``current`` stay within tolerance of ``baseline``?"""
    checks: list[MetricCheck] = []
    for name, (direction, tolerance) in gates.items():
        if name not in baseline:
            continue  # the baseline never had it, so it can't regress

        if name not in current:
            # A metric that silently disappeared (e.g. abstain gate turned off) is a failure.
            checks.append(
                MetricCheck(
                    name=name,
                    baseline=baseline[name],
                    current=None,
                    delta=None,
                    direction=direction,
                    tolerance=tolerance,
                    passed=False,
                )
            )
            continue

        delta = current[name] - baseline[name]
        # 1e-9 so a change exactly equal to the tolerance passes despite float rounding
        if direction == "higher":
            passed = delta >= -tolerance - 1e-9  # allowed to drop by at most `tolerance`
        else:
            passed = delta <= tolerance + 1e-9  # allowed to rise by at most `tolerance`

        checks.append(
            MetricCheck(
                name=name,
                baseline=baseline[name],
                current=current[name],
                delta=delta,
                direction=direction,
                tolerance=tolerance,
                passed=passed,
            )
        )

    return Comparison(checks)  # after the loop, once every metric is checked


def format_report(
    comparison: Comparison,
    baseline_config: str | None = None,
    current_config: str | None = None,
) -> str:
    lines = [
        "| metric | baseline | current | change | status |",
        "|---|---|---|---|---|",
    ]
    for check in comparison.checks:  # each item is a MetricCheck
        status = "ok" if check.passed else "FAIL"
        if check.current is None:  # metric missing from the new run
            lines.append(f"| {check.name} | {check.baseline:.3f} | missing | | {status} |")
        else:
            lines.append(
                f"| {check.name} | {check.baseline:.3f} | {check.current:.3f} "
                f"| {check.delta:+.3f} | {status} |"
            )

    if baseline_config and current_config and baseline_config != current_config:
        lines.append("")
        lines.append(f"Note: baseline is `{baseline_config}`, this run is `{current_config}`.")

    lines.append("")
    lines.append("**PASS**" if comparison.passed else "**FAIL**")
    return "\n".join(lines)


def load_run(path: Path) -> tuple[str | None, dict[str, float]]:
    # (config name, metrics)
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if "metrics" not in data:
        raise ValueError(f"{path} has no 'metrics' section")

    return (data.get("config"), data["metrics"])


def write_baseline(results_path: Path, baseline_path: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    baseline = {
        "config": data["config"],  # e.g. "dense"
        "params": data["params"],  # index_version, model, k values...
        "metrics": data["metrics"],  # what compare() checks
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)  # e.g. creates eval/ if needed
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    results: Annotated[Path, typer.Argument(help="results/<config>.json from run_eval")],
    baseline: Annotated[Path, typer.Option(help="The committed baseline.")] = Path(
        "eval/baseline.json"
    ),
    update_baseline: Annotated[
        bool, typer.Option("--update-baseline", help="Overwrite the baseline with RESULTS.")
    ] = False,
    report: Annotated[
        Path | None, typer.Option(help="Also write the markdown report here (CI comment).")
    ] = None,
) -> None:
    """Exit 0 if RESULTS is no worse than the baseline, 1 if it regressed, 2 if no baseline."""
    if update_baseline:
        write_baseline(results, baseline)
        typer.echo(f"baseline written to {baseline}")
        raise typer.Exit(0)

    if not baseline.exists():
        typer.echo(f"no baseline at {baseline}; create one with --update-baseline")
        raise typer.Exit(2)

    baseline_config, baseline_metrics = load_run(baseline)
    current_config, current_metrics = load_run(results)
    comparison = compare(baseline_metrics, current_metrics)
    text = format_report(comparison, baseline_config, current_config)

    typer.echo(text)
    if report is not None:
        report.write_text(text + "\n", encoding="utf-8")
    raise typer.Exit(0 if comparison.passed else 1)


if __name__ == "__main__":
    app()
