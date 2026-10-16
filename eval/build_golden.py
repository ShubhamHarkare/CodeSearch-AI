"""Drafts candidate golden-set questions for you to review by hand.

What goes here:
    - ``draft_candidates(index, llm, n=200, seed=0) -> list[dict]``:
        * sample chunks stratified by section (learn / reference / blog skipped)
        * ask the LLM for one realistic developer question per chunk that the
          chunk answers; expected_urls = that chunk's url#anchor
        * drop near-duplicate questions (embedding cosine > 0.9)
    - CLI: write ``eval/candidates.jsonl`` (gitignored). You then copy accepted
      rows into golden.jsonl, fixing questions and URLs as needed.

Must not:
    - write to golden.jsonl directly. Ground truth is always human-checked.

Phase: 1
"""
