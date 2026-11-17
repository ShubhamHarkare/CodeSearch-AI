import math
from typing import Any

TIMING_ORDER = ["condense", "retrieval", "generation", "total"]


def format_ms(ms: float) -> str:
    if ms < 1000:
        return str(math.floor(ms)) + " ms"

    return str(round(ms / 1000, 1)) + " s"


def format_matched(terms: dict[str, float]) -> str:
    sorted_terms: dict[str, float] = dict(
        sorted(terms.items(), key=lambda item: item[1], reverse=True)
    )

    response: list[str] = []

    for word, weight in sorted_terms.items():
        response.append(f"{word} {weight:.1f}")

    return ", ".join(response)


def _round(value: float | None) -> float | None:
    """Round a score to 3 decimals; a missing score (None) stays None."""
    return None if value is None else round(value, 3)


def source_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    # chunk id -> candidate, in retrieval order (dicts keep insertion order)
    ids_to_chunk: dict[str, dict[str, Any]] = {}
    for chunk in trace.get("candidates", []):
        ids_to_chunk[chunk["chunk"]["id"]] = chunk
    matched_terms: dict[str, dict[str, float]] = trace.get("matched_terms", {})
    counter: int = 1
    response: list[dict[str, Any]] = []

    # Used sources first, in prompt order: n = 1, 2, ... matches the [n] citations.
    for used_ids in trace.get("used_chunk_ids", []):
        # pop: take it out so the second loop only sees unused candidates.
        # None on an exact cache hit (retrieval never ran, so no candidate).
        candidate = ids_to_chunk.pop(used_ids, None)
        if candidate is None:
            response.append(
                {
                    "n": counter,
                    "used": True,
                    "title": used_ids,  # all we know is the id
                    "url": None,
                    "dense": None,
                    "bm25": None,
                    "rrf": None,
                    "rerank": None,
                    "matched": format_matched(matched_terms.get(used_ids, {})),
                }
            )
        else:
            response.append(
                {
                    "n": counter,
                    "used": True,
                    "title": candidate["chunk"]["title"],
                    "url": candidate["chunk"]["url"],
                    "dense": _round(candidate.get("dense_score")),
                    "bm25": _round(candidate.get("bm25_score")),
                    "rrf": _round(candidate.get("rrf_score")),
                    "rerank": _round(candidate.get("rerank_score")),
                    "matched": format_matched(matched_terms.get(used_ids, {})),
                }
            )
        counter += 1

    # Then the candidates the LLM never saw, in retrieval order: no citation number.
    for ids, candidate in ids_to_chunk.items():
        response.append(
            {
                "n": None,
                "used": False,
                "title": candidate["chunk"]["title"],
                "url": candidate["chunk"]["url"],
                "dense": _round(candidate.get("dense_score")),
                "bm25": _round(candidate.get("bm25_score")),
                "rrf": _round(candidate.get("rrf_score")),
                "rerank": _round(candidate.get("rerank_score")),
                "matched": format_matched(matched_terms.get(ids, {})),
            }
        )

    return response


def summary_lines(trace: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    question = trace.get("question")
    standalone = trace.get("standalone_question")
    if standalone and standalone != question:
        lines.append(f"Searched as: {standalone}")

    if cache_tier := trace.get("cache_tier"):
        lines.append(f"Served from the {cache_tier} cache.")

    if abstain_reason := trace.get("abstain_reason"):
        lines.append(f"Abstained: {abstain_reason}")

    timings = trace.get("timings") or {}
    timing_parts = [
        f"{step} {format_ms(timings[f'{step}_ms'])}"
        for step in TIMING_ORDER
        if f"{step}_ms" in timings
    ]
    if timing_parts:
        lines.append(" · ".join(timing_parts))

    return lines
