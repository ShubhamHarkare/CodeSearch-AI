"""Tests for codesearch/explain/trace.py: TraceBuilder (records one answer) and
TraceStore (keeps traces in SQLite so GET /v1/explain/{trace_id} can serve them)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from codesearch.errors import TraceNotFoundError
from codesearch.explain.trace import TraceBuilder, TraceStore
from codesearch.schemas import Chunk, ExplanationTrace, ScoredChunk, SentenceSupport


def scored(chunk_id: str, **scores: float) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        url=f"https://react.dev/{chunk_id}",
        anchor="",
        title=chunk_id,
        header_path=(chunk_id,),
        text=f"text of {chunk_id}",
        source_path=f"{chunk_id}.md",
        start_line=1,
    )
    return ScoredChunk(chunk=chunk, **scores)


def trace(trace_id: str = "t-1", question: str = "q?") -> ExplanationTrace:
    builder = TraceBuilder(trace_id, question)
    builder.add_candidates([scored("a", dense_score=0.9)])
    return builder.build()


# =====================================================================
# TraceBuilder
# =====================================================================


def test_a_bare_trace_has_the_question_and_sensible_defaults() -> None:
    t = TraceBuilder("t-1", "How do I use state?").build()
    assert t.trace_id == "t-1"
    assert t.question == "How do I use state?"
    assert t.standalone_question == "How do I use state?"  # no history: same question
    assert t.candidates == []
    assert t.used_chunk_ids == []
    assert t.sentences == []
    assert t.abstain_reason is None
    assert t.cache_tier is None


def test_the_standalone_question_is_recorded_separately() -> None:
    builder = TraceBuilder("t", "what about objects?")
    builder.set_standalone("How do I update an object in state?")
    t = builder.build()
    assert t.question == "what about objects?"
    assert t.standalone_question == "How do I update an object in state?"


def test_candidates_keep_their_order_and_scores() -> None:
    builder = TraceBuilder("t", "q")
    builder.add_candidates([scored("a", dense_score=0.9), scored("b", dense_score=0.8)])
    t = builder.build()
    assert [c.chunk.id for c in t.candidates] == ["a", "b"]
    assert t.candidates[0].dense_score == 0.9


def test_set_used_records_the_ids_sent_to_the_llm_in_order() -> None:
    builder = TraceBuilder("t", "q")
    builder.add_candidates([scored("a"), scored("b"), scored("c")])
    builder.set_used([scored("c"), scored("a")])
    assert builder.build().used_chunk_ids == ["c", "a"]


def test_set_used_merges_rerank_scores_into_the_candidates() -> None:
    """The reranker returns copies with rerank_score set: the trace should show them
    next to the retrieval scores, not lose them."""
    builder = TraceBuilder("t", "q")
    builder.add_candidates([scored("a", dense_score=0.9), scored("b", dense_score=0.8)])
    builder.set_used([scored("b", dense_score=0.8, rerank_score=5.0)])
    by_id = {c.chunk.id: c for c in builder.build().candidates}
    assert by_id["b"].rerank_score == 5.0
    assert by_id["b"].dense_score == 0.8
    assert by_id["a"].rerank_score is None  # not reranked into the top: unchanged
    assert [c.chunk.id for c in builder.build().candidates] == ["a", "b"]  # order kept


def test_set_used_ids_records_ids_without_chunks() -> None:
    """A cache hit has no chunks, only the cited chunk ids of the stored answer."""
    builder = TraceBuilder("t", "q")
    builder.set_used_ids(["c2", "c7"])
    t = builder.build()
    assert t.used_chunk_ids == ["c2", "c7"]
    assert t.candidates == []


def test_set_used_adds_chunks_that_were_not_candidates() -> None:
    builder = TraceBuilder("t", "q")
    builder.set_used([scored("x")])
    t = builder.build()
    assert t.used_chunk_ids == ["x"]
    assert [c.chunk.id for c in t.candidates] == ["x"]


def test_sentences_abstain_and_cache_tier() -> None:
    builder = TraceBuilder("t", "q")
    sentence = SentenceSupport(
        sentence="Use state [1].", cited_chunk_ids=["a"], label="supported", score=0.9
    )
    builder.set_sentences([sentence])
    builder.set_abstain("below_threshold (dense 0.512 < 0.755)")
    builder.set_cache_tier("semantic")
    t = builder.build()
    assert t.sentences == [sentence]
    assert t.abstain_reason == "below_threshold (dense 0.512 < 0.755)"
    assert t.cache_tier == "semantic"


def test_matched_terms_default_to_empty_and_are_recorded() -> None:
    builder = TraceBuilder("t", "q")
    assert builder.build().matched_terms == {}
    builder.set_matched_terms({"a": {"state": 1.2, "update": 0.4}, "b": {}})
    assert builder.build().matched_terms == {"a": {"state": 1.2, "update": 0.4}, "b": {}}


def test_matched_terms_are_copied_into_the_trace() -> None:
    terms = {"a": {"state": 1.2}}
    builder = TraceBuilder("t", "q")
    builder.set_matched_terms(terms)
    built = builder.build()
    terms["a"]["state"] = 99.0  # the caller changes its dict afterwards
    assert built.matched_terms == {"a": {"state": 1.2}}


def test_versions_are_recorded() -> None:
    builder = TraceBuilder("t", "q")
    builder.set_versions(index_version="iv", prompt_version="1", model="gemma3:4b")
    assert builder.build().versions == {
        "index_version": "iv",
        "prompt_version": "1",
        "model": "gemma3:4b",
    }


def test_time_records_a_step_in_milliseconds() -> None:
    builder = TraceBuilder("t", "q")
    with builder.time("retrieval"):
        time.sleep(0.01)
    t = builder.build()
    assert set(t.timings) == {"retrieval_ms"}
    assert t.timings["retrieval_ms"] >= 9.0


def test_time_records_even_when_the_step_raises() -> None:
    builder = TraceBuilder("t", "q")
    with pytest.raises(RuntimeError), builder.time("generation"):
        raise RuntimeError("llm down")
    assert "generation_ms" in builder.timings


def test_timings_is_the_live_dict_the_pipeline_can_share() -> None:
    """The pipeline puts the same timings on the Answer, so it reads builder.timings."""
    builder = TraceBuilder("t", "q")
    with builder.time("condense"):
        pass
    builder.timings["total_ms"] = 12.0
    assert builder.build().timings == builder.timings
    assert set(builder.timings) == {"condense_ms", "total_ms"}


def test_build_returns_a_snapshot() -> None:
    builder = TraceBuilder("t", "q")
    first = builder.build()
    builder.set_abstain("no_candidates")
    assert first.abstain_reason is None


# =====================================================================
# TraceStore
# =====================================================================


def test_save_then_get_round_trips() -> None:
    store = TraceStore()
    original = trace("t-1", "How do I use state?")
    store.save(original)
    assert store.get("t-1") == original


def test_get_unknown_raises_trace_not_found() -> None:
    with pytest.raises(TraceNotFoundError):
        TraceStore().get("nope")


def test_saving_the_same_id_again_replaces_it() -> None:
    store = TraceStore()
    store.save(trace("t-1", "old"))
    store.save(trace("t-1", "new"))
    assert store.get("t-1").question == "new"
    assert len(store) == 1


def test_recent_is_newest_first_and_limited() -> None:
    store = TraceStore()
    for i in range(5):
        store.save(trace(f"t-{i}"))
    assert [t.trace_id for t in store.recent(3)] == ["t-4", "t-3", "t-2"]
    assert len(store.recent(100)) == 5


def test_len_counts_traces() -> None:
    store = TraceStore()
    assert len(store) == 0
    store.save(trace("a"))
    store.save(trace("b"))
    assert len(store) == 2


def test_a_file_store_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "traces.db"  # parent folder is created
    first = TraceStore(path)
    first.save(trace("t-1"))
    first.close()
    assert TraceStore(path).get("t-1").trace_id == "t-1"


def test_the_stored_row_is_plain_json(tmp_path: Path) -> None:
    """JSON, not pickle: readable with sqlite3 and safe to load."""
    path = tmp_path / "traces.db"
    store = TraceStore(path)
    store.save(trace("t-1", "How do I use state?"))
    store.close()
    with sqlite3.connect(path) as db:
        (data,) = db.execute("SELECT data FROM traces WHERE trace_id = 't-1'").fetchone()
    assert json.loads(data)["question"] == "How do I use state?"


def test_the_store_works_from_another_thread() -> None:
    """FastAPI runs sync code in a thread pool, so the connection can't be thread-bound."""
    import threading

    store = TraceStore()
    errors: list[BaseException] = []

    def work() -> None:
        try:
            store.save(trace("from-thread"))
        except BaseException as e:
            errors.append(e)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert errors == []
    assert store.get("from-thread").trace_id == "from-thread"
