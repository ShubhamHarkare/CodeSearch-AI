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

API_URL = os.environ.get("CODESEARCH_API_URL", "http://localhost:8000")
HISTORY_TURNS = 10  # the API accepts at most 10 history messages
EXAMPLES = [
    "How do I update state based on the previous state?",
    "When should I use useEffect?",
    "How do I pass props to a component?",
]
BadgeColor = Literal["green", "orange", "red"]
STATUS_COLORS: dict[str, BadgeColor] = {"ok": "green", "degraded": "orange"}
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

        meta = []
        if msg.get("abstained"):
            meta.append("Not covered in the React docs")
        if msg.get("cache_tier"):
            meta.append(f"cached ({msg['cache_tier']})")
        if msg.get("total_ms") is not None:
            meta.append(f"{msg['total_ms'] / 1000:.1f}s")
        if meta:
            st.caption(" · ".join(meta))


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
        if st.button("New chat", icon=":material/add:", use_container_width=True):
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
