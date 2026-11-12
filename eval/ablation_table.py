"""Turn results/*.json into the README's ablation table.

Usage:
    uv run python -m eval.ablation_table                      # print the table
    uv run python -m eval.ablation_table --write README.md    # replace it in the README

The README keeps the table between two markers, so only that block is rewritten:
    <!-- ablation:start -->
    ...
    <!-- ablation:end -->

Section-level metrics are left blank for v1-chunked runs: v1 chunks have no section
anchors, so only the page_* metrics compare fairly against v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("eval/configs")
START = "<!-- ablation:start -->"
END = "<!-- ablation:end -->"

# Rows in reading order: baseline, single retrievers, fusion, rerankers.
ORDER = ["v1", "bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank", "hybrid_rerank_bge"]

# (header, metric key, section-level?) — section-level ones are blank for v1 chunks.
COLUMNS: list[tuple[str, str, bool]] = [
    ("R@1", "recall_at_1", True),
    ("R@5", "recall_at_5", True),
    ("MRR", "mrr", True),
    ("nDCG@10", "ndcg_at_10", True),
    ("Page hit@5", "page_hit_at_5", False),
    ("Page MRR", "page_mrr", False),
    ("Pool recall", "pool_recall", True),
    ("OOS abstain", "abstain_recall", False),
    ("False abstain", "false_abstain_rate", False),
]
BLANK = "n/a"


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    """name -> result dict, for every results/<name>.json."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "metrics" in data:
            results[str(data.get("config", path.stem))] = data
    return results


def describe(name: str, configs_dir: Path = CONFIGS_DIR) -> str:
    """The config's one-line description, or "" if there is no config file."""
    path = configs_dir / f"{name}.yaml"
    if not path.exists():
        return ""
    data = yaml.safe_load(path.read_text()) or {}
    return str(data.get("description", ""))


def ordered(names: list[str]) -> list[str]:
    """Known configs in ORDER, then any others alphabetically."""
    known = [n for n in ORDER if n in names]
    return known + sorted(n for n in names if n not in ORDER)


def render_table(
    results: dict[str, dict[str, Any]],
    *,
    default: str = "dense",
    configs_dir: Path = CONFIGS_DIR,
) -> str:
    """A Markdown table, one row per config; the default config's name is bold.

    The best value in each column is bold too (lowest for false abstain, highest
    for everything else), so the trade-offs are visible at a glance.
    """
    names = ordered(list(results))
    best: dict[str, float] = {}
    for _, key, section in COLUMNS:
        values = [
            results[n]["metrics"][key]
            for n in names
            if key in results[n]["metrics"] and not (section and _is_v1(results[n]))
        ]
        if len(values) > 1:  # "best" only means something with something to beat
            best[key] = min(values) if key == "false_abstain_rate" else max(values)

    header = "| Config | What it does | " + " | ".join(h for h, _, _ in COLUMNS) + " |"
    divider = "|---|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [header, divider]
    for name in names:
        result = results[name]
        label = f"**{name}** (default)" if name == default else name
        cells = [label, describe(name, configs_dir)]
        for _, key, section in COLUMNS:
            value = result["metrics"].get(key)
            if value is None or (section and _is_v1(result)):
                cells.append(BLANK)
                continue
            text = f"{value:.3f}"
            is_best = key in best and abs(value - best[key]) < 1e-12
            cells.append(f"**{text}**" if is_best else text)
        lines.append("| " + " | ".join(cells) + " |")

    n = max((r["params"].get("n_questions", 0) for r in results.values()), default=0)
    oos = max((r["params"].get("n_out_of_scope", 0) for r in results.values()), default=0)
    lines.append("")
    lines.append(
        f"{n} in-scope questions{f' + {oos} out-of-scope' if oos else ''} from "
        "`eval/golden.jsonl`. Best value per column in bold. "
        f"{BLANK} = not measured, or not comparable (v1 chunks have no section anchors, "
        "so compare v1 on the page columns)."
    )
    return "\n".join(lines)


def _is_v1(result: dict[str, Any]) -> bool:
    return bool(result.get("params", {}).get("chunk_strategy") == "v1")


def replace_block(text: str, table: str) -> str:
    """Swap whatever sits between the markers for the new table."""
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README needs both {START} and {END} markers, in that order")
    return text[: start + len(START)] + "\n" + table + "\n" + text[end:]


def main(
    write: Annotated[
        Path | None, typer.Option("--write", help="Replace the table in this file.")
    ] = None,
    default: Annotated[str, typer.Option(help="Config to mark as the default.")] = "dense",
) -> None:
    results = load_results()
    if not results:
        typer.echo("No results/*.json yet: run `make ablation` first.", err=True)
        raise typer.Exit(1)
    table = render_table(results, default=default)
    if write is None:
        typer.echo(table)
        return
    write.write_text(replace_block(write.read_text(), table))
    typer.echo(f"Updated the ablation table in {write} ({len(results)} configs).")


if __name__ == "__main__":
    typer.run(main)
