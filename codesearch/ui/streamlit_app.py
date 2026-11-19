"""The chat interface. Replaces v1's app.py.

What goes here:
    - One ``handle_question(question)`` used by both the chat input and the
      example buttons (v1 duplicated this code).
    - Streaming: render the sources as soon as the ``retrieval`` event arrives,
      then ``st.write_stream`` the tokens, then citation links and sentence
      tags when ``citations`` / ``groundedness`` arrive.
    - "Why this answer?" expander per message: score table (BM25 / dense /
      RRF / rerank per source), matched query terms, sentence labels,
      confidence badge, and a "Deep explain" button.
    - Thumbs up / down per answer -> api_client.send_feedback.
    - Sidebar: readiness (cached 30 s), index version and corpus SHA, cache hit
      rate, p50/p95 latency, rolling groundedness.

Must not:
    - import the pipeline, models or Redis. Everything goes through ApiClient.
    - use auto-generated CSS class names (they break between Streamlit versions).

Phase: 1 (basic chat), 2 (explain panel), 3 (streaming, feedback)
"""

import os
from typing import Any, Literal

import streamlit as st

from codesearch.ui.api_client import ApiClient, ApiError
from codesearch.ui.explain_view import (
    confidence_badge,
    grounding_note,
    sentence_rows,
    source_rows,
    summary_lines,
)

API_URL = os.environ.get("CODESEARCH_API_URL", "http://localhost:8000")
HISTORY_TURNS = 10  # the API accepts at most 10 history messages
EXAMPLES = [
    "How do I update state based on the previous state?",
    "When should I use useEffect?",
    "How do I pass props to a component?",
]
BadgeColor = Literal["green", "orange", "red"]
STATUS_COLORS: dict[str, BadgeColor] = {"ok": "green", "degraded": "orange"}
# key in source_rows -> (column label, number format)
SCORE_COLUMNS = {
    "dense": ("Dense", "{:.3f}"),
    "bm25": ("BM25", "{:.2f}"),
    "rrf": ("RRF", "{:.4f}"),
    "rerank": ("Rerank", "{:.2f}"),
}
LABEL_COLORS = {"supported": "green", "partial": "orange", "unsupported": "red"}
AVATARS = {"user": ":material/person:", "assistant": ":material/menu_book:"}


@st.cache_resource
def get_client() -> ApiClient:
    """One client for the whole app (not one per rerun)."""
    return ApiClient(base_url=API_URL)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_status() -> dict[str, Any]:
    """/readyz, cached 30 s so every rerun doesn't hit the API."""
    try:
        return get_client().ready()
    except ApiError as e:
        return {"status": "offline", "error": str(e)}


@st.cache_data(show_spinner=False, max_entries=200)
def fetch_trace(trace_id: str) -> dict[str, Any] | None:
    """A trace never changes once saved, so fetch it once and keep it."""
    try:
        return get_client().explain(trace_id)
    except ApiError:
        return None  # API restarted with an empty store, or the trace was lost


def handle_question(question: str) -> None:
    """Shared by the chat input and the example buttons."""
    question = question.strip()
    if not question:
        return
    messages: list[dict[str, Any]] = st.session_state.messages
    history = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")]
    messages.append({"role": "user", "content": question})

    try:
        with st.spinner("Searching the React docs…"):
            result = get_client().ask(question, history=history[-HISTORY_TURNS:])
    except ApiError as e:
        # Stored as a message: a bare st.error would vanish on the st.rerun() below.
        messages.append({"role": "assistant", "content": str(e), "error": True})
        return

    messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result.get("citations", []),
            "abstained": result.get("abstained", False),
            "cache_tier": result.get("cache_tier"),
            "total_ms": result.get("timings", {}).get("total_ms"),
            "trace_id": result.get("trace_id"),
            "groundedness": result.get("groundedness"),
            "confidence": result.get("confidence"),
        }
    )


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("error"):
            st.error(msg["content"], icon=":material/error:")
            return
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return

        # The API's number matches the [n] markers in the answer text.
        sources = " · ".join(
            f"[{c['number']}] [{c['title']}]({c['url']})" for c in msg.get("citations", [])
        )
        if sources:
            st.caption(f"Sources: {sources}")

        badge = confidence_badge(msg.get("confidence"))
        if badge is not None:
            _, color, text = badge
            st.markdown(
                f":{color}-badge[{text}]",
                help="Calibrated on the golden set: how often answers like this one "
                "were actually right.",
            )

        meta = []
        if msg.get("abstained"):
            meta.append("Not covered in the React docs")
        if msg.get("cache_tier"):
            meta.append(f"cached ({msg['cache_tier']})")
        if msg.get("groundedness") is not None:
            meta.append(f"grounded {msg['groundedness']:.0%}")
        if msg.get("total_ms") is not None:
            meta.append(f"{msg['total_ms'] / 1000:.1f}s")
        if meta:
            st.caption(" · ".join(meta))

        if msg.get("trace_id"):
            trace = fetch_trace(msg["trace_id"])
            # The visible flag: claims a reader should double-check, outside the expander.
            note = grounding_note(trace) if trace else ""
            if note:
                st.caption(f":orange[:material/warning: {note}. See *Why this answer?*]")
            render_explanation(msg["trace_id"])


def render_explanation(trace_id: str) -> None:
    """The "Why this answer?" panel: what was searched, why, and every source's scores."""
    with st.expander("Why this answer?", icon=":material/manage_search:"):
        trace = fetch_trace(trace_id)
        if trace is None:
            st.caption("Explanation unavailable.")
            return
        for line in summary_lines(trace):
            st.caption(line)
        sentences = sentence_rows(trace)
        if sentences:
            st.markdown("**Sentence check**")
            for row in sentences:
                color = LABEL_COLORS[row["label"]]
                cites = f" · cites {row['sources']}" if row["sources"] else " · no citation"
                st.markdown(
                    f":{color}-badge[{row['label']} {row['score']:.2f}] {row['sentence']}"
                    f" :gray[{cites}]"
                )
            st.markdown("**Sources**")
        rows = source_rows(trace)
        if not rows:
            return
        # Show only the score columns this retrieval mode filled (dense mode has no RRF...).
        scores = [k for k in SCORE_COLUMNS if any(r[k] is not None for r in rows)]
        # Scores as text: a missing score shows as a blank cell, not "None".
        table = [
            {
                "n": f"[{r['n']}]" if r["n"] else "",
                "title": r["title"],
                "url": r["url"],
                "matched": r["matched"],
                **{k: "" if r[k] is None else SCORE_COLUMNS[k][1].format(r[k]) for k in scores},
            }
            for r in rows
        ]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_order=["n", "title", "url", "matched", *scores],
            column_config={
                "n": st.column_config.TextColumn("", width=36),
                "title": st.column_config.TextColumn("Source", width=190),
                "url": st.column_config.LinkColumn("", display_text="open", width=44),
                "matched": st.column_config.TextColumn(
                    "Matched words", width=170, help="Question words in this source, by BM25 weight"
                ),
                **{k: st.column_config.TextColumn(SCORE_COLUMNS[k][0], width=58) for k in scores},
            },
        )
        st.caption(
            "[n] = the citation number in the answer. Rows without one weren't sent to the model."
        )


def render_sidebar() -> None:
    status = fetch_status()
    state = status.get("status", "unknown")
    color: BadgeColor = STATUS_COLORS.get(state, "red")
    with st.sidebar:
        st.markdown("**CodeSearch**")
        st.badge(state, color=color)
        if "error" in status:
            st.caption(status["error"])
        versions = status.get("versions", {})
        components = status.get("components", {})
        st.caption(
            f"Index `{versions.get('index_version', 'n/a')}` · "
            f"prompt v{versions.get('prompt_version', '?')} · "
            f"cache {components.get('cache', 'n/a')}"
        )
        if st.button("New chat", icon=":material/add:", width="stretch"):
            st.session_state.messages = []
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="CodeSearch", page_icon=":material/search:", layout="centered")
    st.session_state.setdefault("messages", [])
    render_sidebar()

    messages: list[dict[str, Any]] = st.session_state.messages
    example_prompt = None
    if not messages:
        # Empty state: a short intro and a few questions to start from.
        st.title("CodeSearch")
        st.caption("Answers from the React docs, with sources. Off-topic questions are declined.")
        st.write("")
        for question in EXAMPLES:
            if st.button(question, type="tertiary", icon=":material/arrow_forward:"):
                example_prompt = question
    else:
        for msg in messages:
            render_message(msg)

    prompt = st.chat_input("Ask about React…") or example_prompt
    if prompt:
        handle_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
