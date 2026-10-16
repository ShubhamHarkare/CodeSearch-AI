"""Shared data shapes passed between packages. Data only, no behaviour.

What goes here (pydantic models, frozen where possible):
    - ``Chunk``: id, url, anchor, title, header_path (list[str]), text,
      source_path, start_line.
    - ``ScoredChunk``: chunk plus bm25_rank, bm25_score, dense_rank,
      dense_score, rrf_score, rerank_score (all optional; None = stage not run).
    - ``Citation``: number (the [n] in the answer), chunk_id, url, title.
    - ``SentenceSupport``: sentence, cited_chunk_ids, label
      ("supported" | "partial" | "unsupported"), score.
    - ``Answer``: text, citations, abstained, confidence, groundedness,
      cache_tier (None | "exact" | "semantic"), timings (dict), trace_id,
      index_version, prompt_version.
    - ``ExplanationTrace``: trace_id, question, standalone_question,
      candidates (list[ScoredChunk]), used_chunk_ids, sentences
      (list[SentenceSupport]), abstain_reason, timings, versions.
    - ``StreamEvent``: type ("retrieval" | "token" | "citations" |
      "groundedness" | "done" | "error") and data.
    - ``RequestSignals``: what monitoring records per request.

Must not:
    - contain business logic or import feature packages. Everything imports
      this module; it imports only pydantic and the standard library.

Tested by: indirectly, through every other test.
Phase: 0
"""
