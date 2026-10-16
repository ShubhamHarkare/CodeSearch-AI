"""Builds cache keys. Pure.

What goes here:
    - ``normalize(query) -> str``: lowercase, collapse whitespace, strip
      surrounding and trailing punctuation (the README promised this; v1 didn't do it).
    - ``make_key(query, index_version, prompt_version, model) -> str``:
      ``"codesearch:q:" + sha256(...)`` so a re-index, prompt change or model
      change never serves a stale answer.

Tested by: tests/unit/test_cache_keys.py (normalisation cases; key changes
when any version changes).
Phase: 1
"""
