"""Tests for eval/run_eval.py (retrieval mode).

The retriever is faked, so these run offline in milliseconds and test the
harness itself: config loading, per-question scoring, aggregation, the
results file and the MLflow run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codesearch.config import Settings
from codesearch.schemas import Chunk, ScoredChunk
from eval.run_eval import EvalConfig, EvalResult, load_config, run, score_row

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "eval" / "configs"

A = "https://react.dev/reference/react/useState"
B = "https://react.dev/learn/state-a-components-memory"
C = "https://react.dev/reference/react/useEffect"

SECTION_KEYS = {
    "hit_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
}
ALL_KEYS = SECTION_KEYS | {f"page_{k}" for k in SECTION_KEYS}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def scored(url_with_anchor: str) -> ScoredChunk:
    url, _, anchor = url_with_anchor.partition("#")
    chunk = Chunk(
        id=url_with_anchor,
        url=url,
        anchor=anchor,
        title="t",
        header_path=(),
        text="x",
        source_path="x.md",
        start_line=0,
    )
    return ScoredChunk(chunk=chunk, dense_rank=1, dense_score=0.5)


class FakeRetriever:
    """Returns canned url#anchor lists per question and records every call."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[ScoredChunk]:
        self.calls.append(query)
        return [scored(u) for u in self.answers.get(query, [])]


def golden_file(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def config_file(tmp_path: Path, text: str, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


GOLDEN = [
    {
        "id": "q001",
        "question": "Q1?",
        "expected_urls": [f"{A}#usestate"],
        "category": "hooks",
        "expected": "answer",
    },
    {
        "id": "q002",
        "question": "Q2?",
        "expected_urls": [f"{B}#adding-a-state-variable"],
        "category": "state",
        "expected": "answer",
    },
    {
        "id": "oos001",
        "question": "Vue?",
        "expected_urls": [],
        "category": "oos_frontend",
        "expected": "abstain",
    },
]
ANSWERS = {
    # q001: exact section at rank 1 -> every section metric is 1.0
    "Q1?": [f"{A}#usestate", f"{C}#x"],
    # q002: right page, wrong section, at rank 2 -> section 0.0, page mrr 0.5
    "Q2?": [f"{C}#y", f"{B}#other"],
    "Vue?": [f"{C}#z"],
}


def fresh_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={"index_dir": tmp_path / "no-index", "embedding_model": "hash"}
    )


def do_run(tmp_path: Path, **kwargs: Any) -> tuple[EvalResult, FakeRetriever]:
    retriever = FakeRetriever(ANSWERS)
    cfg = config_file(
        tmp_path,
        "name: demo\noverrides:\n  retrieval:\n    mode: dense\n    dense_k: 7\n",
    )
    options: dict[str, Any] = {
        "golden_path": golden_file(tmp_path, GOLDEN),
        "settings": fresh_settings(tmp_path),
        "retriever": retriever,
        "results_dir": tmp_path / "results",
        "log_mlflow": False,
    }
    options.update(kwargs)
    return run(cfg, **options), retriever


# =====================================================================
# load_config
# =====================================================================


def test_load_config_reads_every_field(tmp_path: Path) -> None:
    path = config_file(
        tmp_path,
        "name: dense\ndescription: dense only\noverrides:\n  retrieval:\n    mode: dense\n",
    )
    cfg = load_config(path)
    assert isinstance(cfg, EvalConfig)
    assert cfg.name == "dense"
    assert cfg.description == "dense only"
    assert cfg.overrides == {"retrieval": {"mode": "dense"}}


def test_name_defaults_to_the_file_stem(tmp_path: Path) -> None:
    path = config_file(tmp_path, "overrides: {}\n", name="hybrid_rerank.yaml")
    assert load_config(path).name == "hybrid_rerank"


def test_empty_config_file_means_no_overrides(tmp_path: Path) -> None:
    cfg = load_config(config_file(tmp_path, "", name="plain.yaml"))
    assert (cfg.name, cfg.description, cfg.overrides) == ("plain", "", {})


def test_comment_only_config_file_is_empty(tmp_path: Path) -> None:
    cfg = load_config(config_file(tmp_path, "# just a note\n", name="note.yaml"))
    assert cfg.overrides == {}


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, "name: x\noverride: {}\n"))


def test_config_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, "- just\n- a list\n"))


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_is_frozen(tmp_path: Path) -> None:
    cfg = load_config(config_file(tmp_path, "name: x\n"))
    with pytest.raises(ValidationError):
        cfg.name = "y"


def test_repo_v1_config() -> None:
    cfg = load_config(CONFIGS / "v1.yaml")
    assert cfg.name == "v1"
    assert cfg.overrides["index"]["chunk_strategy"] == "v1"
    assert cfg.overrides["retrieval"] == {"mode": "dense", "dense_k": 3}
    # v1 must never share an index directory with the v2 index
    assert cfg.overrides["index"]["index_dir"] != "index"


def test_repo_dense_config() -> None:
    cfg = load_config(CONFIGS / "dense.yaml")
    assert cfg.name == "dense"
    assert cfg.overrides["retrieval"]["mode"] == "dense"
    assert "index" not in cfg.overrides  # default v2 index


@pytest.mark.parametrize("name", ["v1.yaml", "dense.yaml"])
def test_repo_configs_apply_cleanly_to_settings(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    settings = Settings(_env_file=None).with_overrides(**cfg.overrides)  # type: ignore[call-arg]
    assert settings.retrieval.mode == "dense"


# =====================================================================
# score_row
# =====================================================================


def test_score_row_returns_section_and_page_metrics() -> None:
    assert set(score_row([f"{A}#usestate"], [f"{A}#usestate"])) == ALL_KEYS


def test_score_row_perfect_hit_scores_one_everywhere() -> None:
    scores = score_row([f"{A}#usestate"], [f"{A}#usestate", f"{C}#x"])
    assert all(v == pytest.approx(1.0) for v in scores.values())


def test_score_row_right_page_wrong_section() -> None:
    scores = score_row([f"{A}#usestate"], [f"{C}#x", f"{A}#other", f"{A}#usestate"])
    # section level: exact url at rank 3
    assert scores["recall_at_1"] == 0.0
    assert scores["recall_at_3"] == 1.0
    assert scores["mrr"] == pytest.approx(1 / 3)
    # page level: page A first appears at rank 2 (duplicates of A collapse)
    assert scores["page_recall_at_1"] == 0.0
    assert scores["page_mrr"] == pytest.approx(0.5)
    assert scores["page_hit_at_5"] == 1.0


def test_score_row_page_level_dedupes_before_ranking() -> None:
    """Three chunks of page C then page A: at page level A is rank 2, not rank 4."""
    retrieved = [f"{C}#a", f"{C}#b", f"{C}#c", f"{A}#x"]
    scores = score_row([f"{A}#usestate"], retrieved)
    assert scores["page_mrr"] == pytest.approx(0.5)
    assert scores["mrr"] == 0.0


def test_score_row_with_no_hits_is_all_zero() -> None:
    scores = score_row([f"{A}#usestate"], [f"{C}#x", f"{B}#y"])
    assert all(v == 0.0 for v in scores.values())


def test_score_row_with_nothing_retrieved_is_all_zero() -> None:
    assert all(v == 0.0 for v in score_row([f"{A}#usestate"], []).values())


def test_score_row_recall_counts_every_expected_url() -> None:
    expected = [f"{A}#usestate", f"{B}#adding-a-state-variable"]
    scores = score_row(expected, [f"{A}#usestate", f"{C}#x", f"{C}#y"])
    assert scores["recall_at_3"] == pytest.approx(0.5)
    assert scores["page_recall_at_3"] == pytest.approx(0.5)


def test_score_row_page_url_expected_matches_any_section_of_that_page() -> None:
    """A golden row may expect a bare page URL (its intro)."""
    scores = score_row([A], [f"{A}#usestate"])
    assert scores["page_recall_at_1"] == 1.0


def test_score_row_needs_expected_urls() -> None:
    with pytest.raises(ValueError):
        score_row([], [f"{A}#usestate"])


# =====================================================================
# run: scoring and aggregation
# =====================================================================


def test_run_scores_retrieval_only_on_answer_rows(tmp_path: Path) -> None:
    result, retriever = do_run(tmp_path)
    # every row is retrieved (the abstain gate needs the out-of-scope scores too) ...
    assert retriever.calls == ["Q1?", "Q2?", "Vue?"]
    # ... but retrieval metrics only cover rows that have expected urls
    assert [r["id"] for r in result.rows] == ["q001", "q002"]


def test_run_metrics_are_means_over_answer_rows(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    assert result.metrics["recall_at_5"] == pytest.approx(0.5)  # 1.0 and 0.0
    assert result.metrics["mrr"] == pytest.approx(0.5)
    assert result.metrics["page_mrr"] == pytest.approx(0.75)  # 1.0 and 0.5
    assert set(result.metrics) >= ALL_KEYS


def test_run_breaks_metrics_down_by_category(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    assert set(result.per_category) == {"hooks", "state"}
    assert result.per_category["hooks"]["recall_at_5"] == pytest.approx(1.0)
    assert result.per_category["state"]["recall_at_5"] == 0.0
    assert result.per_category["state"]["page_mrr"] == pytest.approx(0.5)


def test_run_records_what_was_retrieved_per_question(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    row = result.rows[1]
    assert row["id"] == "q002"
    assert row["question"] == "Q2?"
    assert row["category"] == "state"
    assert row["expected_urls"] == [f"{B}#adding-a-state-variable"]
    assert row["retrieved_urls"] == [f"{C}#y", f"{B}#other"]
    assert row["scores"]["page_mrr"] == pytest.approx(0.5)


def test_run_keeps_at_most_ten_retrieved_urls_per_row(tmp_path: Path) -> None:
    many = {"Q1?": [f"{C}#s{i}" for i in range(25)], "Q2?": []}
    result, _ = do_run(tmp_path, retriever=FakeRetriever(many))
    assert len(result.rows[0]["retrieved_urls"]) == 10


def test_anchorless_chunks_are_scored_by_their_page_url(tmp_path: Path) -> None:
    """v1 chunks have no anchor: retrieved url is the bare page url."""
    result, _ = do_run(tmp_path, retriever=FakeRetriever({"Q1?": [A], "Q2?": [B]}))
    assert result.rows[0]["retrieved_urls"] == [A]
    assert result.metrics["page_recall_at_1"] == pytest.approx(1.0)
    assert result.metrics["recall_at_1"] == 0.0


# =====================================================================
# run: params, results file, MLflow
# =====================================================================


def test_run_applies_config_overrides_and_records_params(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    assert result.name == "demo"
    assert result.params["config"] == "demo"
    assert result.params["retrieval_mode"] == "dense"
    assert result.params["dense_k"] == 7
    assert result.params["chunk_strategy"] == "v2"
    assert result.params["embedding_model"] == "hash"
    assert result.params["n_questions"] == 2
    assert result.params["n_out_of_scope"] == 1


def test_run_without_a_built_index_records_unknown_version(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    assert result.params["index_version"] == "unknown"
    assert result.params["corpus_sha"] == "unknown"


def test_run_writes_a_results_file(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path)
    path = tmp_path / "results" / "demo.json"
    assert result.results_path == path
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"] == "demo"
    assert data["metrics"] == pytest.approx(result.metrics)
    assert data["params"]["dense_k"] == 7
    assert len(data["rows"]) == 2
    assert set(data["per_category"]) == {"hooks", "state"}


def test_run_creates_the_results_dir(tmp_path: Path) -> None:
    do_run(tmp_path, results_dir=tmp_path / "deep" / "er" / "results")
    assert (tmp_path / "deep" / "er" / "results" / "demo.json").exists()


def test_run_does_not_touch_the_golden_file(tmp_path: Path) -> None:
    path = golden_file(tmp_path, GOLDEN)
    before = path.read_bytes()
    do_run(tmp_path, golden_path=path)
    assert path.read_bytes() == before


def test_run_without_mlflow_creates_no_tracking_store(tmp_path: Path) -> None:
    do_run(tmp_path, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert not (tmp_path / "mlflow.db").exists()


def test_run_logs_an_mlflow_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # artifacts land under ./mlruns
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result, _ = do_run(tmp_path, log_mlflow=True, tracking_uri=uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("codesearch-eval")
    assert experiment is not None
    (logged,) = client.search_runs([experiment.experiment_id])
    assert logged.info.run_name == "demo"
    assert logged.data.metrics["recall_at_5"] == pytest.approx(result.metrics["recall_at_5"])
    assert logged.data.params["dense_k"] == "7"
    assert logged.data.params["config"] == "demo"
    artifacts = {a.path for a in client.list_artifacts(logged.info.run_id)}
    assert "results.json" in artifacts


# =====================================================================
# run: reranking
# =====================================================================


class ReverseReranker:
    """Fake reranker: reverses the candidates and cuts to top_n; records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        self.calls.append((query, len(chunks), top_n))
        return list(reversed(chunks))[:top_n]


RERANK_CFG = (
    "name: rr\noverrides:\n  retrieval:\n    mode: hybrid\n"
    "    reranker_model: some/model\n    rerank_top_n: 1\n"
)


def test_reranker_is_applied_when_the_config_names_a_model(tmp_path: Path) -> None:
    reranker = ReverseReranker()
    result = run(
        config_file(tmp_path, RERANK_CFG),
        golden_path=golden_file(tmp_path, GOLDEN),
        settings=fresh_settings(tmp_path),
        retriever=FakeRetriever(ANSWERS),
        reranker=reranker,
        results_dir=tmp_path / "results",
        log_mlflow=False,
    )
    # out-of-scope rows are reranked too: the abstain gate reads their rerank scores
    assert reranker.calls == [("Q1?", 2, 1), ("Q2?", 2, 1), ("Vue?", 1, 1)]
    # reversed + cut to 1: q001 now gets C#x (miss), q002 gets B#other (page hit)
    assert result.rows[0]["retrieved_urls"] == [f"{C}#x"]
    assert result.rows[1]["retrieved_urls"] == [f"{B}#other"]
    assert result.metrics["page_mrr"] == pytest.approx(0.5)


def test_no_reranking_without_a_reranker_model(tmp_path: Path) -> None:
    reranker = ReverseReranker()
    result, _ = do_run(tmp_path, reranker=reranker)
    assert reranker.calls == []
    assert result.rows[0]["retrieved_urls"] == [f"{A}#usestate", f"{C}#x"]


def test_rerank_params_are_recorded(tmp_path: Path) -> None:
    result = run(
        config_file(tmp_path, RERANK_CFG),
        golden_path=golden_file(tmp_path, GOLDEN),
        settings=fresh_settings(tmp_path),
        retriever=FakeRetriever(ANSWERS),
        reranker=ReverseReranker(),
        results_dir=tmp_path / "results",
        log_mlflow=False,
    )
    assert result.params["reranker_model"] == "some/model"
    assert result.params["rerank_top_n"] == 1
    plain, _ = do_run(tmp_path)
    assert plain.params["reranker_model"] == "none"


def test_pool_recall_measures_the_candidates_before_reranking(tmp_path: Path) -> None:
    """pool_recall: share of expected urls anywhere in the candidate pool (rerank ceiling)."""
    result = run(
        config_file(tmp_path, RERANK_CFG),
        golden_path=golden_file(tmp_path, GOLDEN),
        settings=fresh_settings(tmp_path),
        retriever=FakeRetriever(ANSWERS),
        reranker=ReverseReranker(),
        results_dir=tmp_path / "results",
        log_mlflow=False,
    )
    # q001's pool holds A#usestate (pool hit) even though the reranked top-1 misses it
    assert result.rows[0]["scores"]["pool_recall"] == 1.0
    assert result.rows[0]["scores"]["recall_at_1"] == 0.0
    assert result.rows[1]["scores"]["pool_recall"] == 0.0
    assert result.rows[1]["scores"]["page_pool_recall"] == 1.0
    assert result.metrics["pool_recall"] == pytest.approx(0.5)


def test_pool_recall_is_zero_when_nothing_is_retrieved(tmp_path: Path) -> None:
    result, _ = do_run(tmp_path, retriever=FakeRetriever({}))
    assert result.metrics["pool_recall"] == 0.0
    assert result.metrics["page_pool_recall"] == 0.0


@pytest.mark.parametrize("name", ["hybrid_rerank.yaml", "dense_rerank.yaml"])
def test_repo_rerank_configs(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    settings = Settings(_env_file=None).with_overrides(**cfg.overrides)  # type: ignore[call-arg]
    assert settings.retrieval.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.retrieval.rerank_top_n == 10
