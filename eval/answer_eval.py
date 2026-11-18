"""End-to-end answer quality over the golden set (answer mode).

    uv run python -m eval.answer_eval eval/configs/dense.yaml --limit 10

Asks every golden question through the full pipeline (retrieval, LLM, NLI groundedness),
then reports answer rate, groundedness, citation precision, out-of-scope refusals and
latency. Writes results/answers_<name>.json and results/sentences_<name>.csv (one row
per graded sentence, with a blank "human" column to label by hand for threshold tuning).
"""

import asyncio
import csv
import json
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path
from typing import Annotated, Any

import typer

from codesearch.config import Settings
from codesearch.errors import TraceNotFoundError
from codesearch.factory import build_pipeline
from codesearch.schemas import Answer, ExplanationTrace
from eval.golden import GoldenRow, load_golden
from eval.metrics import to_page
from eval.run_eval import GOLDEN_PATH, load_config


def citation_precision(cited_urls: Sequence[str], expected_urls: Sequence[str]) -> float | None:
    if not cited_urls:
        return None
    unique_cited_urls = set(to_page(url) for url in cited_urls)
    unique_expected_urls = set(to_page(url) for url in expected_urls)

    common: int = len(unique_expected_urls.intersection(unique_cited_urls))

    return common / len(unique_cited_urls)


def percentile(values: Sequence[float], p: float) -> float:
    # nearest rank: sorted(values)[ceil(p * n) - 1];  empty -> ValueError
    if not values:
        raise ValueError

    return sorted(values)[ceil(p * len(values)) - 1]


def score_answer(row: GoldenRow, answer: Answer, trace: ExplanationTrace | None) -> dict[str, Any]:
    p = citation_precision([c.url for c in answer.citations], row.expected_urls)
    supported, partial, unsupported = 0, 0, 0
    if trace is not None:  # no trace (e.g. groundedness off) -> no sentences
        for sentence in trace.sentences:
            if sentence.label == "supported":
                supported += 1
            elif sentence.label == "partial":
                partial += 1
            else:
                unsupported += 1
    return {
        "id": row.id,
        "category": row.category,
        "question": row.question,
        "expected": row.expected,
        "answer": answer.text,
        "abstained": answer.abstained,
        "groundedness": answer.groundedness,
        "citation_precision": p,
        "cited_relevant": None if p is None else p > 0,
        "n_citations": len(answer.citations),
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "total_ms": answer.timings.get("total_ms"),
        "sentences": [
            {"sentence": s.sentence, "label": s.label, "score": s.score}
            for s in (trace.sentences if trace else [])
        ],
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    in_scope = [r for r in records if r["expected"] == "answer"]
    out_scope = [r for r in records if r["expected"] == "abstain"]
    answered = [r for r in in_scope if not r["abstained"]]

    if in_scope:
        metrics["answer_rate"] = len(answered) / len(in_scope)

    groundedness_vals = [r["groundedness"] for r in answered if r["groundedness"] is not None]
    if groundedness_vals:
        metrics["groundedness"] = sum(groundedness_vals) / len(groundedness_vals)

    precision_vals = [
        r["citation_precision"] for r in answered if r["citation_precision"] is not None
    ]
    if precision_vals:
        metrics["citation_precision"] = sum(precision_vals) / len(precision_vals)

    if answered:
        cited_hits = sum(1 for r in answered if r["cited_relevant"] is True)
        metrics["cited_hit_rate"] = cited_hits / len(answered)

    supported = sum(r["supported"] or 0 for r in answered)
    partial = sum(r["partial"] or 0 for r in answered)
    unsupported = sum(r["unsupported"] or 0 for r in answered)
    total = supported + partial + unsupported
    if total:
        metrics["supported_share"] = supported / total
        metrics["partial_share"] = partial / total
        metrics["unsupported_share"] = unsupported / total

    if out_scope:
        refusals = sum(1 for r in out_scope if r["abstained"])
        metrics["oos_refusal_rate"] = refusals / len(out_scope)

    latencies = [r["total_ms"] for r in records if r["total_ms"] is not None]
    if latencies:
        metrics["latency_p50_ms"] = percentile(latencies, 0.5)
        metrics["latency_p95_ms"] = percentile(latencies, 0.95)

    return metrics


def sentence_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"id": record["id"], "question": record["question"], **sentence, "human": ""}
        for record in records
        for sentence in record.get("sentences", [])
    ]
    return rows


async def run_answers(
    pipeline: Any,
    rows: Sequence[GoldenRow],
    *,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    todo = list(rows)[:limit] if limit is not None else list(rows)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(todo, start=1):
        answer = await pipeline.answer(row.question, bypass_cache=True)
        try:
            trace = pipeline.trace_store.get(answer.trace_id)  # sync: no await
        except TraceNotFoundError:
            trace = None

        records.append(score_answer(row, answer, trace))

        if on_progress is not None:
            on_progress(i, len(todo))
    return records


def run(
    config_path: Path,
    *,
    pipeline: Any = None,
    golden_path: Path = GOLDEN_PATH,
    results_dir: Path = Path("results"),
    limit: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if pipeline is None:
        settings = Settings().with_overrides(**cfg.overrides)
        settings = settings.with_overrides(
            cache={"enabled": False}, explain={"trace_db": ":memory:"}
        )
        pipeline = build_pipeline(settings)

    records = asyncio.run(run_answers(pipeline, load_golden(golden_path), limit=limit))
    return _write(cfg.name, records, results_dir)


def _write(name: str, records: list[dict[str, Any]], results_dir: Path) -> dict[str, Any]:
    """Summarize, then save the JSON results and the sentence CSV for hand-labelling."""
    metrics = summarize(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"answers_{name}.json").write_text(
        json.dumps({"config": name, "metrics": metrics, "rows": records}, indent=2)
    )
    with (results_dir / f"sentences_{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "question", "sentence", "label", "score", "human"]
        )
        writer.writeheader()
        writer.writerows(sentence_rows(records))
    return {"config": name, "metrics": metrics, "rows": records}


def main(
    config: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="eval/configs/*.yaml")
    ],
    limit: Annotated[int | None, typer.Option(help="Only the first N golden questions.")] = None,
) -> None:
    """Answer the golden questions end to end and report answer quality."""

    def progress(done: int, total: int) -> None:
        typer.echo(f"\r  {done}/{total}", nl=False)

    cfg = load_config(config)
    settings = Settings().with_overrides(**cfg.overrides)
    settings = settings.with_overrides(cache={"enabled": False}, explain={"trace_db": ":memory:"})
    pipeline = build_pipeline(settings)
    rows = load_golden(GOLDEN_PATH)
    records = asyncio.run(run_answers(pipeline, rows, limit=limit, on_progress=progress))
    typer.echo("")
    result = _write(cfg.name, records, Path("results"))
    typer.echo(f"{result['config']}: {len(records)} questions")
    for key, value in result["metrics"].items():
        typer.echo(f"  {key:<20} {value:.3f}")
    typer.echo(f"results -> results/answers_{cfg.name}.json, results/sentences_{cfg.name}.csv")


if __name__ == "__main__":
    typer.run(main)
