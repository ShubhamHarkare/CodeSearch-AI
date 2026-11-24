"""Tests for eval/answer_eval.py: end-to-end answer quality over the golden set.

The pipeline is faked (answers are scripted per question), so these run in
milliseconds; the real run needs Ollama and the NLI model:

    uv run python -m eval.answer_eval eval/configs/dense.yaml --limit 10
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from codesearch.errors import LLMRateLimitedError, LLMUnavailableError, TraceNotFoundError
from codesearch.schemas import (
    Answer,
    Chunk,
    Citation,
    ExplanationTrace,
    ScoredChunk,
    SentenceSupport,
)
from eval.answer_eval import (
    ask_with_retry,
    backoff,
    citation_precision,
    percentile,
    result_name,
    run,
    run_answers,
    score_answer,
    sentence_rows,
    summarize,
)
from eval.golden import GoldenRow

PAGE_A = "https://react.dev/learn/a"
PAGE_B = "https://react.dev/learn/b"


def row(qid: str, expected: str = "answer", urls: tuple[str, ...] = (f"{PAGE_A}#x",)) -> GoldenRow:
    return GoldenRow(
        id=qid,
        question=f"question {qid}?",
        expected_urls=urls if expected == "answer" else (),
        category="state" if expected == "answer" else "oos_general",
        expected=expected,  # type: ignore[arg-type]
    )


def answer(
    *urls: str,
    abstained: bool = False,
    groundedness: float | None = 1.0,
    total_ms: float = 100.0,
    trace_id: str = "t",
    text: str = "An answer [1].",
) -> Answer:
    return Answer(
        text=text,
        citations=[
            Citation(number=i, chunk_id=f"c{i}", url=u, title=f"T{i}")
            for i, u in enumerate(urls, start=1)
        ],
        abstained=abstained,
        groundedness=groundedness,
        timings={"total_ms": total_ms},
        trace_id=trace_id,
        index_version="iv",
        prompt_version="1",
    )


def cand(chunk_id: str, dense: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            url=PAGE_A,
            anchor="",
            title=chunk_id,
            header_path=(chunk_id,),
            text="t",
            source_path="x.md",
            start_line=1,
        ),
        dense_score=dense,
    )


def trace_with(*labels: str, trace_id: str = "t", best: float | None = None) -> ExplanationTrace:
    return ExplanationTrace(
        trace_id=trace_id,
        question="q",
        standalone_question="q",
        candidates=[],
        used_chunk_ids=[],
        sentences=[
            SentenceSupport(  # type: ignore[arg-type]
                sentence=f"S{i}.",
                cited_chunk_ids=["c1"],
                label=label,
                score=0.5,
                best_score=best,
                best_chunk_id=None if best is None else "c2",
            )
            for i, label in enumerate(labels)
        ],
    )


# =====================================================================
# citation_precision / percentile
# =====================================================================


def test_citation_precision_compares_pages_not_anchors() -> None:
    expected = [f"{PAGE_A}#updating-objects"]
    assert citation_precision([f"{PAGE_A}#something-else"], expected) == 1.0
    assert citation_precision([PAGE_A, PAGE_B], expected) == 0.5
    assert citation_precision([PAGE_B], expected) == 0.0


def test_citation_precision_with_no_citations_is_none() -> None:
    assert citation_precision([], [PAGE_A]) is None


def test_citation_precision_counts_each_cited_page_once() -> None:
    """Two chunks of the same page cited twice is one relevant page, not two."""
    assert citation_precision([f"{PAGE_A}#x", f"{PAGE_A}#y", PAGE_B], [PAGE_A]) == 0.5


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
        ([5.0], 0.95, 5.0),
        (list(map(float, range(1, 21))), 0.95, 19.0),
    ],
)
def test_percentile_nearest_rank(values: list[float], p: float, expected: float) -> None:
    assert percentile(values, p) == expected


def test_percentile_does_not_need_sorted_input() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)


# =====================================================================
# score_answer: one record per question
# =====================================================================


def test_score_answer_for_an_answered_question() -> None:
    record = score_answer(
        row("q1"),
        answer(f"{PAGE_A}#x", PAGE_B, groundedness=0.5, total_ms=8000.0),
        trace_with("supported", "unsupported"),
    )
    assert record["id"] == "q1"
    assert record["category"] == "state"
    assert record["expected"] == "answer"
    assert record["abstained"] is False
    assert record["groundedness"] == 0.5
    assert record["citation_precision"] == 0.5
    assert record["cited_relevant"] is True
    assert record["n_citations"] == 2
    assert (record["supported"], record["partial"], record["unsupported"]) == (1, 0, 1)
    assert record["total_ms"] == 8000.0
    assert record["answer"] == "An answer [1]."
    assert record["sentences"] == [
        {
            "sentence": "S0.",
            "label": "supported",
            "score": 0.5,
            "best_score": None,
            "best_chunk_id": None,
        },
        {
            "sentence": "S1.",
            "label": "unsupported",
            "score": 0.5,
            "best_score": None,
            "best_chunk_id": None,
        },
    ]


def test_score_answer_records_the_calibration_features() -> None:
    """The features the calibrator will be fitted on: from the chunks sent to the LLM."""
    trace = ExplanationTrace(
        trace_id="t",
        question="q",
        standalone_question="q",
        candidates=[cand("u1", 0.9), cand("u2", 0.7), cand("x", 0.95)],
        used_chunk_ids=["u1", "u2"],  # "x" was retrieved but not sent
        sentences=[],
    )
    record = score_answer(row("q1"), answer(PAGE_A, groundedness=0.5), trace)
    assert record["features"] == pytest.approx([0.9, 0.8, 0.5])


def test_score_answer_without_a_trace_has_no_features() -> None:
    assert score_answer(row("q1"), answer(PAGE_A), None)["features"] is None


def test_score_answer_without_citations() -> None:
    record = score_answer(row("q1"), answer(groundedness=None), None)
    assert record["citation_precision"] is None
    assert record["cited_relevant"] is None
    assert record["n_citations"] == 0
    assert record["sentences"] == []
    assert (record["supported"], record["partial"], record["unsupported"]) == (0, 0, 0)


def test_score_answer_for_an_abstain() -> None:
    record = score_answer(row("o1", "abstain"), answer(abstained=True, groundedness=None), None)
    assert record["expected"] == "abstain"
    assert record["abstained"] is True


def test_records_are_json_serialisable() -> None:
    record = score_answer(row("q1"), answer(PAGE_A), trace_with("supported"))
    assert json.loads(json.dumps(record)) == record


# =====================================================================
# summarize: the headline numbers
# =====================================================================


def test_summarize_answer_quality_over_answered_in_scope_questions() -> None:
    records = [
        score_answer(
            row("q1"), answer(PAGE_A, groundedness=1.0), trace_with("supported", "supported")
        ),
        score_answer(
            row("q2"), answer(PAGE_B, groundedness=0.5), trace_with("supported", "unsupported")
        ),
        score_answer(row("q3"), answer(abstained=True, groundedness=None), None),  # false abstain
        score_answer(row("q4"), answer(groundedness=0.0), trace_with("unsupported")),  # no citation
    ]
    m = summarize(records)
    assert m["answer_rate"] == 0.75  # 3 of 4 in-scope answered
    assert m["groundedness"] == pytest.approx(0.5)  # mean of 1.0, 0.5, 0.0
    assert m["citation_precision"] == pytest.approx(0.5)  # q1 1.0, q2 0.0 (q4 has none)
    assert m["cited_hit_rate"] == pytest.approx(1 / 3)  # only q1 cites a right page
    assert m["supported_share"] == pytest.approx(3 / 5)
    assert m["partial_share"] == 0.0
    assert m["unsupported_share"] == pytest.approx(2 / 5)
    assert "oos_refusal_rate" not in m  # no out-of-scope rows


def test_summarize_out_of_scope_refusals() -> None:
    records = [
        score_answer(row("o1", "abstain"), answer(abstained=True, groundedness=None), None),
        score_answer(row("o2", "abstain"), answer(abstained=True, groundedness=None), None),
        score_answer(row("o3", "abstain"), answer(PAGE_A), None),  # answered: bad
    ]
    m = summarize(records)
    assert m["oos_refusal_rate"] == pytest.approx(2 / 3)
    assert "answer_rate" not in m  # no in-scope rows
    assert "groundedness" not in m


def test_summarize_latency_over_every_question() -> None:
    records = [
        score_answer(row(f"q{i}"), answer(PAGE_A, total_ms=float(ms)), None)
        for i, ms in enumerate([100, 200, 300, 9000])
    ]
    m = summarize(records)
    assert m["latency_p50_ms"] == 200.0
    assert m["latency_p95_ms"] == 9000.0


def test_summarize_skips_metrics_with_no_data() -> None:
    m = summarize([score_answer(row("q1"), answer(groundedness=None), None)])
    assert "groundedness" not in m  # nothing graded
    assert "citation_precision" not in m  # nothing cited
    assert "supported_share" not in m  # no sentences
    assert m["answer_rate"] == 1.0


def test_summarize_of_nothing_is_empty() -> None:
    assert summarize([]) == {}


def test_summarize_mis_citation_when_the_diagnostic_ran() -> None:
    """Every sentence scored 0.5 against its citation (not supported) but 0.9 against
    some other chunk: all of them are mis-cited, all are supported somewhere."""
    records = [
        score_answer(row("q1"), answer(PAGE_A), trace_with("unsupported", "partial", best=0.9)),
        score_answer(row("q2"), answer(PAGE_A), trace_with("unsupported", best=0.2)),
    ]
    m = summarize(records)
    assert m["supported_anywhere_share"] == pytest.approx(2 / 3)
    assert m["miscited_share"] == pytest.approx(2 / 3)


def test_summarize_a_supported_sentence_is_never_miscited() -> None:
    records = [score_answer(row("q1"), answer(PAGE_A), trace_with("supported", best=0.95))]
    m = summarize(records)
    assert m["miscited_share"] == 0.0
    assert m["supported_anywhere_share"] == 1.0


def test_summarize_has_no_diagnostic_metrics_without_the_diagnostic() -> None:
    m = summarize([score_answer(row("q1"), answer(PAGE_A), trace_with("unsupported"))])
    assert "miscited_share" not in m
    assert "supported_anywhere_share" not in m


# =====================================================================
# sentence_rows: the CSV you label by hand to tune the thresholds
# =====================================================================


def test_sentence_rows_one_per_sentence_with_a_blank_human_column() -> None:
    records = [
        score_answer(row("q1"), answer(PAGE_A), trace_with("supported", "partial")),
        score_answer(row("q2"), answer(PAGE_A), trace_with("unsupported")),
    ]
    rows = sentence_rows(records)
    assert [(r["id"], r["sentence"], r["label"]) for r in rows] == [
        ("q1", "S0.", "supported"),
        ("q1", "S1.", "partial"),
        ("q2", "S0.", "unsupported"),
    ]
    assert all(r["human"] == "" for r in rows)
    assert set(rows[0]) == {
        "id",
        "question",
        "sentence",
        "label",
        "score",
        "best_score",
        "best_chunk_id",
        "human",
    }


# =====================================================================
# run_answers / run: the loop, with a scripted pipeline
# =====================================================================


class ScriptedPipeline:
    """Answers each question from a script; records how it was called."""

    def __init__(self, script: dict[str, tuple[Answer, ExplanationTrace | None]]) -> None:
        self.script = script
        self.calls: list[tuple[str, bool]] = []
        self.trace_store = self

    async def answer(
        self, question: str, history: Any = None, *, bypass_cache: bool = False
    ) -> Answer:
        self.calls.append((question, bypass_cache))
        return self.script[question][0]

    def get(self, trace_id: str) -> ExplanationTrace:  # the trace store
        for ans, trace in self.script.values():
            if ans.trace_id == trace_id and trace is not None:
                return trace
        raise TraceNotFoundError(trace_id)


def scripted() -> tuple[ScriptedPipeline, list[GoldenRow]]:
    rows = [row("q1"), row("q2"), row("o1", "abstain")]
    script = {
        "question q1?": (answer(PAGE_A, trace_id="t1"), trace_with("supported", trace_id="t1")),
        "question q2?": (
            answer(PAGE_B, groundedness=0.0, trace_id="t2"),
            trace_with("unsupported", trace_id="t2"),
        ),
        "question o1?": (answer(abstained=True, groundedness=None, trace_id="t3"), None),
    }
    return ScriptedPipeline(script), rows


async def test_run_answers_asks_every_question_without_the_cache() -> None:
    pipeline, rows = scripted()
    records = await run_answers(pipeline, rows)
    assert [r["id"] for r in records] == ["q1", "q2", "o1"]
    assert all(bypass for _, bypass in pipeline.calls)  # cached answers would skew the eval


async def test_run_answers_attaches_each_trace() -> None:
    pipeline, rows = scripted()
    records = await run_answers(pipeline, rows)
    assert records[0]["supported"] == 1
    assert records[1]["unsupported"] == 1
    assert records[2]["sentences"] == []  # no trace stored: fine


async def test_run_answers_limit() -> None:
    pipeline, rows = scripted()
    assert len(await run_answers(pipeline, rows, limit=2)) == 2


async def test_run_answers_reports_progress() -> None:
    pipeline, rows = scripted()
    seen: list[tuple[int, int]] = []
    await run_answers(pipeline, rows, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_run_writes_results_and_the_sentence_csv(tmp_path: Path) -> None:
    pipeline, rows = scripted()
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(r.model_dump_json() for r in rows) + "\n")
    config = tmp_path / "dense.yaml"
    config.write_text("name: dense\noverrides:\n  retrieval:\n    mode: dense\n")

    result = run(config, pipeline=pipeline, golden_path=golden, results_dir=tmp_path / "out")

    assert result["metrics"]["answer_rate"] == 1.0
    assert result["metrics"]["oos_refusal_rate"] == 1.0
    saved = json.loads((tmp_path / "out" / "answers_dense.json").read_text())
    assert saved["config"] == "dense"
    assert saved["metrics"] == result["metrics"]
    assert [r["id"] for r in saved["rows"]] == ["q1", "q2", "o1"]
    with (tmp_path / "out" / "sentences_dense.csv").open() as f:
        table = list(csv.DictReader(f))
    assert [t["label"] for t in table] == ["supported", "unsupported"]
    assert "best_score" in table[0]
    assert table[0]["human"] == ""


# =====================================================================
# Free-tier rate limits: retry with growing waits, pause between questions
# =====================================================================


class RateLimitedPipeline(ScriptedPipeline):
    """Refuses the first ``fail`` calls with a 429, then answers from the script."""

    def __init__(self, script: Any, fail: int, error: Exception | None = None) -> None:
        super().__init__(script)
        self.fail = fail
        self.error = error or LLMRateLimitedError("rate limited, retry after 7s")

    async def answer(
        self, question: str, history: Any = None, *, bypass_cache: bool = False
    ) -> Answer:
        if self.fail > 0:
            self.fail -= 1
            self.calls.append((question, bypass_cache))
            raise self.error
        return await super().answer(question, history, bypass_cache=bypass_cache)


class Clock:
    """A fake asyncio.sleep: records the waits instead of waiting."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def test_backoff_doubles_up_to_a_minute() -> None:
    assert [backoff(i) for i in range(7)] == [5, 10, 20, 40, 60, 60, 60]


async def test_a_rate_limited_question_is_retried_until_it_answers() -> None:
    script = {"question q1?": (answer(PAGE_A, trace_id="t1"), None)}
    pipeline, clock = RateLimitedPipeline(script, fail=2), Clock()
    got = await ask_with_retry(pipeline, "question q1?", sleep=clock)
    assert got.trace_id == "t1"
    assert len(pipeline.calls) == 3  # two refusals, then the answer
    assert clock.waits == [5, 10]


async def test_retrying_gives_up_after_the_last_attempt() -> None:
    # Still refused after ~4 minutes: probably the daily quota. Stop, don't loop forever.
    script = {"question q1?": (answer(PAGE_A), None)}
    pipeline, clock = RateLimitedPipeline(script, fail=99), Clock()
    with pytest.raises(LLMRateLimitedError):
        await ask_with_retry(pipeline, "question q1?", retries=3, sleep=clock)
    assert len(pipeline.calls) == 4  # the first try + 3 retries
    assert clock.waits == [5, 10, 20]


async def test_other_llm_errors_are_not_retried() -> None:
    # A wrong model or key won't fix itself by waiting.
    script = {"question q1?": (answer(PAGE_A), None)}
    pipeline = RateLimitedPipeline(script, fail=1, error=LLMUnavailableError("HTTP 404"))
    clock = Clock()
    with pytest.raises(LLMUnavailableError):
        await ask_with_retry(pipeline, "question q1?", sleep=clock)
    assert clock.waits == []


async def test_each_wait_is_reported() -> None:
    script = {"question q1?": (answer(PAGE_A), None)}
    seen: list[float] = []
    await ask_with_retry(
        RateLimitedPipeline(script, fail=1), "question q1?", sleep=Clock(), on_wait=seen.append
    )
    assert seen == [5]


async def test_run_answers_survives_rate_limits() -> None:
    pipeline, rows = scripted()
    limited = RateLimitedPipeline(pipeline.script, fail=2)
    records = await run_answers(limited, rows, sleep=Clock())
    assert [r["id"] for r in records] == ["q1", "q2", "o1"]


async def test_run_answers_pauses_between_questions_not_before_the_first() -> None:
    pipeline, rows = scripted()
    clock = Clock()
    await run_answers(pipeline, rows, pause=4.0, sleep=clock)
    assert clock.waits == [4.0, 4.0]  # 3 questions, 2 gaps


async def test_no_pause_by_default() -> None:
    pipeline, rows = scripted()
    clock = Clock()
    await run_answers(pipeline, rows, sleep=clock)
    assert clock.waits == []


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("", "dense"),
        ("gpt-oss-20b", "dense_gpt-oss-20b"),
        ("openai/gpt-oss-20b", "dense_openai-gpt-oss-20b"),  # "/" would make a folder
        ("  gemma3 4b ", "dense_gemma3-4b"),
    ],
)
def test_result_name(tag: str, expected: str) -> None:
    assert result_name("dense", tag) == expected


def test_a_tagged_run_does_not_overwrite_the_untagged_one(tmp_path: Path) -> None:
    pipeline, rows = scripted()
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    config = tmp_path / "dense.yaml"
    config.write_text("name: dense\noverrides:\n  retrieval:\n    mode: dense\n")
    run(config, pipeline=pipeline, golden_path=golden, results_dir=tmp_path / "r")
    run(config, pipeline=pipeline, golden_path=golden, results_dir=tmp_path / "r", tag="oss")
    assert (tmp_path / "r" / "answers_dense.json").exists()
    assert (tmp_path / "r" / "answers_dense_oss.json").exists()
