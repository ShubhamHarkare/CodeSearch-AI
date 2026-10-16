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
