"""Runs one eval config over the golden set (retrieval mode) and records the results.

    uv run python -m eval.run_eval eval/configs/dense.yaml

For each in-scope golden question: retrieve, score at section level (url#anchor)
and page level (url only), then average overall and per category. Writes
``results/<name>.json`` and logs params, metrics and the per-question rows to MLflow.
Abstain rows are skipped until the abstain gate exists (Phase 1c).

Must not:
    - change golden.jsonl or baseline.json.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import BaseModel, ConfigDict

from codesearch.config import Settings
from codesearch.factory import build_embedder, build_reranker, load_index
from codesearch.ingest.manifest import read_manifest
from codesearch.ingest.urls import url_with_anchor
from codesearch.interfaces import Reranker, Retriever
from codesearch.retrieval.abstain import decide, top_signal
from codesearch.retrieval.hybrid import HybridRetriever
from codesearch.schemas import ScoredChunk
from eval.golden import load_golden
from eval.metrics import (
    abstain_precision_recall,
    aggregate,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    to_page,
)

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "codesearch-eval"


@dataclass(frozen=True)
class ThresholdChoice:
    """A suggested abstain threshold and what it would cost on the golden set."""

    threshold: float
    abstain_recall: float  # share of out-of-scope questions refused
    false_abstain_rate: float  # share of in-scope questions wrongly refused


def choose_threshold(
    in_scope: Sequence[float], out_of_scope: Sequence[float], target_recall: float = 0.9
) -> ThresholdChoice:
    """Lowest threshold that refuses at least ``target_recall`` of out-of-scope questions.

    Refuse means score < threshold, so a higher threshold only ever refuses more; the
    lowest one that meets the target therefore refuses the fewest in-scope questions.
    """
    if not out_of_scope:
        raise ValueError("need at least one out-of-scope score")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    candidates = sorted(set(in_scope) | set(out_of_scope))
    # Just above the highest score: refuses everything, so the loop always finds a value.
    candidates.append(math.nextafter(candidates[-1], math.inf))

    for t in candidates:
        recall = sum(s < t for s in out_of_scope) / len(out_of_scope)
        if recall >= target_recall:
            false_rate = sum(s < t for s in in_scope) / len(in_scope) if in_scope else 0.0
            return ThresholdChoice(float(t), float(recall), float(false_rate))
    raise AssertionError("unreachable: the last candidate refuses every question")


class EvalConfig(BaseModel):
    """One eval config file: a name and the Settings overrides it applies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    overrides: dict[str, dict[str, Any]] = {}


def load_config(path: Path) -> EvalConfig:
    """Read an eval config YAML; the name defaults to the file stem."""
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    ## 1. Handle empty or comment-only YAML files
    if data is None:
        data = {}

    ## 2. Check if the parsed YAML is a mapping (dict)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a YAML mapping, got {type(data).__name__}")

    # 3. Default 'name' to file stem if missing or empty
    if "name" not in data or data["name"] is None:
        data["name"] = path.stem

    # Validate and return Pydantic model
    return EvalConfig.model_validate(data)


def score_row(expected_urls: Sequence[str], retrieved_urls: Sequence[str]) -> dict[str, float]:
    """Section-level and page-level (``page_*``) ranking metrics for one question."""
    if not expected_urls:
        raise ValueError("expected_urls cannot be empty.")

    expected_pages = [to_page(url) for url in expected_urls]
    predicted_pages = [to_page(url) for url in retrieved_urls]

    def compute_metrics(target: Sequence[str], preds: Sequence[str]) -> dict[str, float]:
        return {
            "hit_at_5": hit_rate_at_k(preds, target, k=5),
            "recall_at_1": recall_at_k(preds, target, k=1),
            "recall_at_3": recall_at_k(preds, target, k=3),
            "recall_at_5": recall_at_k(preds, target, k=5),
            "recall_at_10": recall_at_k(preds, target, k=10),
            "mrr": mrr(preds, target, k=10),
            "ndcg_at_10": ndcg_at_k(preds, target, k=10),
        }

    # 3. Compute section-level metrics
    section_metrics = compute_metrics(expected_urls, retrieved_urls)

    # 4. Compute page-level metrics with 'page_' prefix
    page_metrics = {
        f"page_{key}": val for key, val in compute_metrics(expected_pages, predicted_pages).items()
    }

    # 5. Merge and return all 14 MLflow-safe metric keys
    return {**section_metrics, **page_metrics}


def pool_recall(expected_urls: Sequence[str], pool_urls: Sequence[str]) -> dict[str, float]:
    """Share of expected urls found anywhere in the candidate pool (section and page level)."""
    k = max(len(pool_urls), 1)
    expected_pages = [to_page(url) for url in expected_urls]
    pool_pages = [to_page(url) for url in pool_urls]
    return {
        "pool_recall": recall_at_k(pool_urls, expected_urls, k),
        "page_pool_recall": recall_at_k(pool_pages, expected_pages, k),
    }


def abstain_gate(results: Sequence[ScoredChunk], threshold: float) -> dict[str, Any]:
    """The abstain decision for one question, or all None when it can't be judged.

    BM25-only results carry no rerank or dense score, so the gate has nothing to read.
    """
    if results and top_signal(list(results)) is None:
        return {"top_score": None, "score_kind": None, "abstained": None}
    decision = decide(list(results), threshold)
    return {
        "top_score": decision.top_score,
        "score_kind": decision.score_kind,
        "abstained": decision.abstain,
    }


def abstain_metrics(
    answer_rows: Sequence[dict[str, Any]], abstain_rows: Sequence[dict[str, Any]]
) -> dict[str, float]:
    """Abstain precision/recall over every judged row, plus the in-scope false-abstain rate."""
    judged = [(r["abstained"], False) for r in answer_rows if r["abstained"] is not None]
    judged += [(r["abstained"], True) for r in abstain_rows if r["abstained"] is not None]
    if not judged:
        return {}
    predicted = [p for p, _ in judged]
    expected = [e for _, e in judged]
    precision, recall = abstain_precision_recall(predicted, expected)
    in_scope = [p for p, e in judged if not e]
    false_rate = sum(in_scope) / len(in_scope) if in_scope else 0.0
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "false_abstain_rate": false_rate,
    }


def suggest_threshold(
    answer_rows: Sequence[dict[str, Any]], abstain_rows: Sequence[dict[str, Any]]
) -> ThresholdChoice | None:
    """choose_threshold over the judged rows' top scores (None if nothing to tune on)."""

    def scores(rows: Sequence[dict[str, Any]]) -> list[float]:
        return [
            r["top_score"]
            for r in rows
            if r["abstained"] is not None and r["top_score"] is not None
        ]

    out_of_scope = scores(abstain_rows)
    if not out_of_scope:
        return None
    return choose_threshold(scores(answer_rows), out_of_scope)


@dataclass
class EvalResult:
    """What one eval run produced (also written to results/<name>.json)."""

    name: str
    metrics: dict[str, float]
    per_category: dict[str, dict[str, float]]
    rows: list[dict[str, Any]]
    params: dict[str, Any]
    results_path: Path
    abstain_rows: list[dict[str, Any]]
    suggested_threshold: ThresholdChoice | None


def run(
    config_path: Path,
    *,
    golden_path: Path = GOLDEN_PATH,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    results_dir: Path = Path("results"),
    log_mlflow: bool = True,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> EvalResult:
    """Run one config over the golden set; see the module docstring."""
    cfg = load_config(config_path)
    base_settings = settings or Settings()
    settings = base_settings.with_overrides(**cfg.overrides)

    golden = load_golden(golden_path)

    if retriever is None:
        retriever = HybridRetriever(
            load_index(settings),
            build_embedder(settings),
            settings.retrieval,
        )
    evaluated_rows: list[dict[str, Any]] = []
    all_scores: list[dict[str, float]] = []
    scores_by_category: dict[str, list[dict[str, float]]] = {}
    abstain_rows: list[dict[str, Any]] = []

    for row in golden:
        # Every row is retrieved (and reranked): the abstain gate needs the
        # out-of-scope scores as much as the in-scope ones.
        results = retriever.retrieve(row.question)
        # The candidate pool before reranking: the most any reranker could find.
        pool_urls = [url_with_anchor(r.chunk.url, r.chunk.anchor) for r in results]
        if settings.retrieval.reranker_model:
            if reranker is None:
                reranker = build_reranker(settings)  # built once, reused for every question
            results = reranker.rerank(row.question, results, settings.retrieval.rerank_top_n)

        gate = abstain_gate(results, settings.retrieval.abstain_threshold)
        if row.expected == "abstain":
            abstain_rows.append(
                {"id": row.id, "question": row.question, "category": row.category, **gate}
            )
            continue

        # Convert chunk results to anchor URLs
        retrieved_urls = [url_with_anchor(r.chunk.url, r.chunk.anchor) for r in results]

        # Score candidate matches against expected ground truth
        scores = score_row(row.expected_urls, retrieved_urls)
        scores.update(pool_recall(row.expected_urls, pool_urls))

        row_record = {
            "id": row.id,
            "question": row.question,
            "category": row.category,
            "expected_urls": list(row.expected_urls),
            "retrieved_urls": retrieved_urls[:10],
            "scores": scores,
            **gate,
        }

        evaluated_rows.append(row_record)
        all_scores.append(scores)
        scores_by_category.setdefault(row.category, []).append(scores)
    metrics = aggregate(all_scores)
    metrics.update(abstain_metrics(evaluated_rows, abstain_rows))
    suggested = suggest_threshold(evaluated_rows, abstain_rows)
    per_category = {cat: aggregate(cat_scores) for cat, cat_scores in scores_by_category.items()}

    # 7. Extract manifest metadata and assemble evaluation params
    manifest = read_manifest(settings.index.index_dir)
    index_version = manifest.index_version if manifest else "unknown"
    corpus_sha = manifest.corpus_sha if manifest else "unknown"

    params: dict[str, Any] = {
        "config": cfg.name,
        "retrieval_mode": settings.retrieval.mode,
        "dense_k": settings.retrieval.dense_k,
        "bm25_k": settings.retrieval.bm25_k,
        "rrf_k": settings.retrieval.rrf_k,
        "reranker_model": settings.retrieval.reranker_model or "none",
        "rerank_top_n": settings.retrieval.rerank_top_n,
        "chunk_strategy": settings.index.chunk_strategy,
        "embedding_model": settings.index.embedding_model,
        "index_version": index_version,
        "corpus_sha": corpus_sha,
        "n_questions": len(evaluated_rows),
        "n_out_of_scope": len(abstain_rows),
        "abstain_threshold": settings.retrieval.abstain_threshold,
    }

    payload: dict[str, Any] = {
        "config": cfg.name,
        "params": params,
        "metrics": metrics,
        "per_category": per_category,
        "rows": evaluated_rows,
        "abstain_rows": abstain_rows,
        "suggested_threshold": asdict(suggested) if suggested else None,
    }

    # 8. Create results directory and save output JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{cfg.name}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 9. Optional MLflow logging with dynamic import
    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name=cfg.name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(payload, "results.json")

    return EvalResult(
        name=cfg.name,
        metrics=metrics,
        per_category=per_category,
        rows=evaluated_rows,
        params=params,
        results_path=out_file,
        abstain_rows=abstain_rows,
        suggested_threshold=suggested,
    )


SUMMARY_KEYS = ("recall_at_5", "mrr", "ndcg_at_10", "page_recall_at_5", "page_mrr")


def main(
    config: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="eval/configs/*.yaml")
    ],
    no_mlflow: Annotated[bool, typer.Option("--no-mlflow", help="Skip MLflow logging.")] = False,
) -> None:
    """Run one eval config and print the headline metrics."""
    result = run(config, log_mlflow=not no_mlflow)
    typer.echo(f"{result.name}: {result.params['n_questions']} questions")
    for key in SUMMARY_KEYS:
        typer.echo(f"  {key:<18} {result.metrics[key]:.3f}")
    for key in ("abstain_recall", "false_abstain_rate"):
        if key in result.metrics:
            typer.echo(f"  {key:<18} {result.metrics[key]:.3f}")
    if result.suggested_threshold:
        t = result.suggested_threshold
        typer.echo(
            f"suggested abstain_threshold {t.threshold:.4f} "
            f"(refuses {t.abstain_recall:.0%} of out-of-scope, "
            f"{t.false_abstain_rate:.0%} of in-scope)"
        )
    typer.echo(f"results -> {result.results_path}")


if __name__ == "__main__":
    typer.run(main)
