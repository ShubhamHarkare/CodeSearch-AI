"""The chat interface: Streamlit, talking to the API only through ApiClient.

A question streams from POST /v1/ask/stream: the sources show as soon as retrieval
finishes, then the answer types itself out. When the ``done`` event arrives the page
re-renders the finished message: cleaned citations, confidence badge, grounding
note, timings (first words and total) and the "Why this answer?" panel with an
on-demand "Deep explain". Pure formatting lives in explain_view.py (unit tested).
"""

import os
from collections.abc import Iterator
from typing import Any, Literal

import streamlit as st

from codesearch.ui.api_client import ApiClient, ApiError
from codesearch.ui.explain_view import (
    attribution_rows,
    confidence_badge,
    grounding_note,
    message_from_answer,
    sentence_rows,
    source_rows,
    sources_line,
    summary_lines,
    timing_caption,
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
    """Shared by the chat input and the example buttons. Streams the answer live."""
    question = question.strip()
    if not question:
        return
    messages: list[dict[str, Any]] = st.session_state.messages
    history = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")]
    messages.append({"role": "user", "content": question})
    render_message(messages[-1])  # show it now; the whole page re-renders after the answer

    done: dict[str, Any] = {}
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status = st.empty()
        status.caption("Searching the React docs…")

        def tokens() -> Iterator[str]:
            for event in get_client().ask_stream(question, history=history[-HISTORY_TURNS:]):
                if event.type == "retrieval":
                    status.caption(f"Reading {sources_line(event.data)}")
                elif event.type == "token":
                    yield event.data
                elif event.type == "done":
                    done.update(event.data)

        try:
            st.write_stream(tokens())
        except ApiError as e:
            # Stored as a message: a bare st.error would vanish on the st.rerun() after this.
            messages.append({"role": "assistant", "content": str(e), "error": True})
            return
    messages.append(message_from_answer(done))


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("error"):
            st.error(msg["content"], icon=":material/error:")
            return
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return

        # The API's number matches the [n] markers in the answer text.
        sources = sources_line(msg.get("citations", []))
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
        if msg.get("cache_tier") == "shared":
            meta.append("shared with an identical question")
        elif msg.get("cache_tier"):
            meta.append(f"cached ({msg['cache_tier']})")
        if msg.get("groundedness") is not None:
            meta.append(f"grounded {msg['groundedness']:.0%}")
        if timing := timing_caption(msg.get("total_ms"), msg.get("ttft_ms")):
            meta.append(timing)
        if meta:
            st.caption(" · ".join(meta))

        if msg.get("trace_id"):
            trace = fetch_trace(msg["trace_id"])
            # The visible flag: claims a reader should double-check, outside the expander.
            note = grounding_note(trace) if trace else ""
            if note:
                st.caption(f":orange[:material/warning: {note}. See *Why this answer?*]")
            render_explanation(msg["trace_id"])
            render_feedback(msg["trace_id"])


def render_feedback(trace_id: str) -> None:
    """Thumbs up / down under an answer; a thumbs down can carry an optional comment.

    Streamlit reruns the script on every click, so remember what was already sent
    and only POST when the vote or the comment changed.
    """
    choice = st.feedback("thumbs", key=f"fb_{trace_id}")  # None, 0 = down, 1 = up
    if choice is None:
        return
    vote = 1 if choice == 1 else -1
    comment = None
    if vote == -1:
        comment = (
            st.text_input(
                "What was wrong? (optional)",
                key=f"fbc_{trace_id}",
                max_chars=500,
                placeholder="e.g. outdated API, wrong hook, missing example",
            ).strip()
            or None
        )
    sent: dict[str, tuple[int, str | None]] = st.session_state.setdefault("feedback_sent", {})
    if sent.get(trace_id) == (vote, comment):
        st.caption("Thanks for the feedback.")
        return
    try:
        get_client().send_feedback(trace_id, vote, comment)
    except ApiError as e:
        st.caption(f":orange[:material/warning: Feedback not saved: {e}]")
        return
    sent[trace_id] = (vote, comment)
    st.caption("Thanks for the feedback.")


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
        render_deep_explain(trace_id, trace)


def render_deep_explain(trace_id: str, trace: dict[str, Any]) -> None:
    """Leave-one-out attribution, on demand: which source actually drove the answer."""
    if trace.get("cache_tier") or trace.get("abstain_reason") or not trace.get("used_chunk_ids"):
        return  # nothing to attribute: the API would answer 409
    key = f"deep_{trace_id}"
    if key not in st.session_state:
        n = len(trace["used_chunk_ids"])
        if st.button(
            "Deep explain",
            key=f"btn_{trace_id}",
            icon=":material/science:",
            help=f"Re-asks the question {n} times, each time without one source, "
            "to see which source the answer depended on. Slow: ~5 s per source.",
        ):
            with st.spinner(f"Re-asking without each of the {n} sources…"):
                try:
                    st.session_state[key] = get_client().deep_explain(trace_id)["scores"]
                except ApiError as e:
                    st.session_state[key] = {"error": str(e)}
            st.rerun()
        return

    result = st.session_state[key]
    if "error" in result:
        st.caption(f":orange[:material/warning: {result['error']}]")
        return
    st.markdown("**Which source drove the answer**")
    if not any(result.values()):
        st.caption("Removing any single source left the answer unchanged.")
        return
    for row in attribution_rows(trace, result):
        st.progress(row["share"], text=f"[{row['n']}] {row['title']} — {row['share']:.0%}")
    st.caption("Share of the answer that changed when each source was left out.")


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
