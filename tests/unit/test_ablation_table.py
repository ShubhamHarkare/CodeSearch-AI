"""Tests for eval/ablation_table.py (results/*.json -> README table)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.ablation_table import (
    BLANK,
    END,
    START,
    load_results,
    ordered,
    render_table,
    replace_block,
)


def result(name: str, strategy: str = "v2", **metrics: float) -> dict[str, Any]:
    return {
        "config": name,
        "params": {"chunk_strategy": strategy, "n_questions": 55, "n_out_of_scope": 24},
        "metrics": metrics,
    }


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    (d / "dense.yaml").write_text("name: dense\ndescription: dense top-10\n")
    return d


def test_load_results_reads_every_json(tmp_path: Path) -> None:
    (tmp_path / "dense.json").write_text(json.dumps(result("dense", mrr=0.5)))
    (tmp_path / "bm25.json").write_text(json.dumps(result("bm25", mrr=0.3)))
    (tmp_path / "notes.json").write_text(json.dumps({"something": "else"}))  # ignored
    assert set(load_results(tmp_path)) == {"dense", "bm25"}


def test_rows_follow_the_story_order_then_unknown_alphabetically() -> None:
    assert ordered(["zeta", "dense", "v1", "alpha", "bm25"]) == [
        "v1",
        "bm25",
        "dense",
        "alpha",
        "zeta",
    ]


def test_table_has_one_row_per_config_and_marks_the_default(configs: Path) -> None:
    table = render_table(
        {"dense": result("dense", mrr=0.57), "bm25": result("bm25", mrr=0.38)},
        configs_dir=configs,
    )
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == 3  # header + 2 configs
    assert "**dense** (default)" in table
    assert "dense top-10" in table  # description from the YAML


def test_the_best_value_per_column_is_bold(configs: Path) -> None:
    table = render_table(
        {
            "dense": result("dense", mrr=0.574, false_abstain_rate=0.127),
            "hybrid": result("hybrid", mrr=0.554, false_abstain_rate=0.2),
        },
        configs_dir=configs,
    )
    assert "**0.574**" in table and "**0.554**" not in table
    assert "**0.127**" in table  # lowest false-abstain rate wins


def test_a_column_with_one_value_has_no_bold(configs: Path) -> None:
    table = render_table(
        {"dense": result("dense", mrr=0.5, abstain_recall=0.917), "bm25": result("bm25", mrr=0.3)},
        configs_dir=configs,
    )
    assert "0.917" in table and "**0.917**" not in table


def test_missing_metrics_are_blank(configs: Path) -> None:
    table = render_table({"bm25": result("bm25", mrr=0.4)}, configs_dir=configs)
    bm25_row = next(line for line in table.splitlines() if line.startswith("| bm25"))
    assert bm25_row.count(BLANK) == 8  # everything except MRR


def test_v1_section_metrics_are_blank_but_page_metrics_show(configs: Path) -> None:
    table = render_table(
        {
            "v1": result("v1", "v1", mrr=0.027, page_hit_at_5=0.855),
            "dense": result("dense", mrr=0.574, page_hit_at_5=0.982),
        },
        configs_dir=configs,
    )
    v1_row = next(line for line in table.splitlines() if line.startswith("| v1"))
    assert "0.027" not in v1_row  # not comparable: v1 chunks have no anchors
    assert "0.855" in v1_row
    assert "**0.982**" in table  # page columns compete: v2 beats v1
    assert "**0.574**" not in table  # v1's near-zero MRR doesn't compete for "best"


def test_footer_says_how_many_questions(configs: Path) -> None:
    table = render_table({"dense": result("dense", mrr=0.5)}, configs_dir=configs)
    assert "55 in-scope questions + 24 out-of-scope" in table


def test_replace_block_only_touches_the_marked_section() -> None:
    readme = f"# Title\n\nintro\n\n{START}\nold table\n{END}\n\n## Next\n"
    new = replace_block(readme, "NEW TABLE")
    assert "old table" not in new
    assert new.startswith("# Title\n\nintro\n\n")
    assert new.endswith(f"\n{END}\n\n## Next\n")
    assert f"{START}\nNEW TABLE\n{END}" in new


def test_replace_block_is_idempotent() -> None:
    readme = f"a\n{START}\nx\n{END}\nb\n"
    once = replace_block(readme, "T")
    assert replace_block(once, "T") == once


def test_replace_block_needs_both_markers() -> None:
    with pytest.raises(ValueError):
        replace_block("no markers here", "T")
    with pytest.raises(ValueError):
        replace_block(f"{END}\n{START}", "T")
