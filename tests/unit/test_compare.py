"""Tests for eval/compare.py: the regression gate behind CI.

A run passes when no gated metric is worse than the committed baseline by more
than its tolerance. "Worse" means lower for most metrics and higher for
false_abstain_rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval.compare import (
    GATES,
    Comparison,
    MetricCheck,
    app,
    compare,
    format_report,
    load_run,
    write_baseline,
)

BASE = {
    "recall_at_5": 0.673,
    "mrr": 0.574,
    "page_recall_at_5": 0.873,
    "abstain_recall": 0.917,
    "false_abstain_rate": 0.127,
    "ndcg_at_10": 0.585,  # reported, not gated
}


def check(comparison: Comparison, name: str) -> MetricCheck:
    (found,) = (c for c in comparison.checks if c.name == name)
    return found


# =====================================================================
# GATES: which metrics are checked and how
# =====================================================================


def test_gates_cover_retrieval_and_abstention() -> None:
    assert set(GATES) == {
        "recall_at_5",
        "mrr",
        "page_recall_at_5",
        "abstain_recall",
        "false_abstain_rate",
    }


def test_false_abstain_rate_is_the_only_lower_is_better_metric() -> None:
    assert {name for name, (direction, _) in GATES.items() if direction == "lower"} == {
        "false_abstain_rate"
    }


def test_tolerances_are_small_positive_points() -> None:
    assert all(0 < tol <= 0.05 for _, tol in GATES.values())


# =====================================================================
# compare
# =====================================================================


def test_identical_run_passes() -> None:
    result = compare(BASE, dict(BASE))
    assert result.passed
    assert all(c.passed for c in result.checks)
    assert all(c.delta == pytest.approx(0.0) for c in result.checks)


def test_one_check_per_gated_metric_in_gate_order() -> None:
    assert [c.name for c in compare(BASE, BASE).checks] == list(GATES)


def test_improvement_passes() -> None:
    better = {**BASE, "recall_at_5": 0.75, "false_abstain_rate": 0.05}
    result = compare(BASE, better)
    assert result.passed
    assert check(result, "recall_at_5").delta == pytest.approx(0.077)
    assert check(result, "false_abstain_rate").delta == pytest.approx(-0.077)


def test_small_drop_within_tolerance_passes() -> None:
    _, tol = GATES["mrr"]
    current = {**BASE, "mrr": BASE["mrr"] - tol / 2}
    assert compare(BASE, current).passed


def test_drop_exactly_at_tolerance_passes() -> None:
    """Float noise must not fail a drop that equals the tolerance."""
    _, tol = GATES["recall_at_5"]
    current = {**BASE, "recall_at_5": BASE["recall_at_5"] - tol}
    assert compare(BASE, current).passed


def test_drop_beyond_tolerance_fails() -> None:
    _, tol = GATES["recall_at_5"]
    current = {**BASE, "recall_at_5": BASE["recall_at_5"] - tol - 0.01}
    result = compare(BASE, current)
    assert not result.passed
    failing = check(result, "recall_at_5")
    assert not failing.passed
    assert failing.baseline == pytest.approx(BASE["recall_at_5"])
    assert failing.current == pytest.approx(current["recall_at_5"])
    assert failing.delta == pytest.approx(-tol - 0.01)
    # every other metric still passes
    assert all(c.passed for c in result.checks if c.name != "recall_at_5")


def test_rise_in_false_abstain_rate_fails() -> None:
    _, tol = GATES["false_abstain_rate"]
    current = {**BASE, "false_abstain_rate": BASE["false_abstain_rate"] + tol + 0.01}
    result = compare(BASE, current)
    assert not result.passed
    assert not check(result, "false_abstain_rate").passed


def test_drop_in_false_abstain_rate_is_an_improvement() -> None:
    assert compare(BASE, {**BASE, "false_abstain_rate": 0.0}).passed


def test_ungated_metrics_never_fail_the_gate() -> None:
    assert compare(BASE, {**BASE, "ndcg_at_10": 0.0}).passed


def test_metric_missing_from_the_current_run_fails() -> None:
    """A metric that silently disappears (e.g. abstain gate turned off) is a regression."""
    current = {k: v for k, v in BASE.items() if k != "abstain_recall"}
    result = compare(BASE, current)
    assert not result.passed
    missing = check(result, "abstain_recall")
    assert (missing.passed, missing.current, missing.delta) == (False, None, None)


def test_metric_missing_from_the_baseline_is_skipped() -> None:
    """A metric the baseline never had can't regress; it's left out of the checks."""
    baseline = {k: v for k, v in BASE.items() if k != "abstain_recall"}
    result = compare(baseline, BASE)
    assert result.passed
    assert "abstain_recall" not in [c.name for c in result.checks]


def test_custom_gates_can_be_passed_in() -> None:
    gates = {"ndcg_at_10": ("higher", 0.01)}
    result = compare(BASE, {**BASE, "ndcg_at_10": 0.5}, gates=gates)
    assert [c.name for c in result.checks] == ["ndcg_at_10"]
    assert not result.passed


def test_records_direction_and_tolerance() -> None:
    c = check(compare(BASE, BASE), "false_abstain_rate")
    assert (c.direction, c.tolerance) == GATES["false_abstain_rate"]


# =====================================================================
# format_report (markdown for the PR comment)
# =====================================================================


def test_report_is_a_markdown_table_with_every_check() -> None:
    report = format_report(compare(BASE, BASE))
    assert "| metric |" in report.lower()
    for name in GATES:
        assert name in report


def test_report_says_pass_or_fail() -> None:
    assert "PASS" in format_report(compare(BASE, BASE))
    bad = compare(BASE, {**BASE, "mrr": 0.2})
    text = format_report(bad)
    assert "FAIL" in text
    assert "mrr" in text


def test_report_shows_signed_deltas() -> None:
    text = format_report(compare(BASE, {**BASE, "mrr": BASE["mrr"] + 0.05}))
    assert "+0.050" in text


def test_report_marks_a_missing_metric() -> None:
    current = {k: v for k, v in BASE.items() if k != "mrr"}
    assert "missing" in format_report(compare(BASE, current)).lower()


def test_report_notes_a_config_mismatch() -> None:
    text = format_report(
        compare(BASE, BASE), baseline_config="dense", current_config="hybrid_rerank"
    )
    assert "dense" in text and "hybrid_rerank" in text


# =====================================================================
# load_run / write_baseline
# =====================================================================


def results_file(tmp_path: Path, metrics: dict[str, float], config: str = "dense") -> Path:
    payload: dict[str, Any] = {
        "config": config,
        "params": {"config": config, "index_version": "abc123"},
        "metrics": metrics,
        "per_category": {"hooks": {"mrr": 0.9}},
        "rows": [{"id": "q001"}],
        "abstain_rows": [{"id": "oos001"}],
    }
    path = tmp_path / f"{config}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_run_returns_config_and_metrics(tmp_path: Path) -> None:
    config, metrics = load_run(results_file(tmp_path, BASE))
    assert config == "dense"
    assert metrics == pytest.approx(BASE)


def test_load_run_rejects_a_file_without_metrics(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"config": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_run(path)


def test_load_run_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope.json")


def test_write_baseline_keeps_config_params_metrics_only(tmp_path: Path) -> None:
    out = tmp_path / "eval" / "baseline.json"
    write_baseline(results_file(tmp_path, BASE), out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"config", "params", "metrics"}  # no per-question rows
    assert data["metrics"] == pytest.approx(BASE)
    assert data["params"]["index_version"] == "abc123"


def test_written_baseline_loads_back(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    write_baseline(results_file(tmp_path, BASE), out)
    assert load_run(out) == ("dense", pytest.approx(BASE))


# =====================================================================
# CLI: python -m eval.compare RESULTS [--baseline PATH] [--update-baseline]
# =====================================================================

runner = CliRunner()


def test_cli_exits_zero_on_pass(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    write_baseline(results_file(tmp_path, BASE), baseline)
    out = runner.invoke(app, [str(results_file(tmp_path, BASE)), "--baseline", str(baseline)])
    assert out.exit_code == 0, out.output
    assert "PASS" in out.output


def test_cli_exits_one_on_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    write_baseline(results_file(tmp_path, BASE), baseline)
    worse = results_file(tmp_path, {**BASE, "mrr": 0.3}, config="dense_worse")
    out = runner.invoke(app, [str(worse), "--baseline", str(baseline)])
    assert out.exit_code == 1
    assert "FAIL" in out.output


def test_cli_update_baseline_writes_and_exits_zero(tmp_path: Path) -> None:
    baseline = tmp_path / "eval" / "baseline.json"
    out = runner.invoke(
        app, [str(results_file(tmp_path, BASE)), "--baseline", str(baseline), "--update-baseline"]
    )
    assert out.exit_code == 0, out.output
    assert load_run(baseline)[0] == "dense"


def test_cli_missing_baseline_is_a_clear_error(tmp_path: Path) -> None:
    out = runner.invoke(
        app, [str(results_file(tmp_path, BASE)), "--baseline", str(tmp_path / "none.json")]
    )
    assert out.exit_code == 2
    assert "baseline" in out.output.lower()


def test_cli_writes_the_report_to_a_file_when_asked(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    write_baseline(results_file(tmp_path, BASE), baseline)
    report = tmp_path / "report.md"
    out = runner.invoke(
        app,
        [str(results_file(tmp_path, BASE)), "--baseline", str(baseline), "--report", str(report)],
    )
    assert out.exit_code == 0, out.output
    assert "PASS" in report.read_text(encoding="utf-8")
