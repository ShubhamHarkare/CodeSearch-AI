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

import hashlib
import json
import unicodedata

KEY_PREFIX = "codesearch:q:"


def normalize(query: str) -> str:
    temp: str = unicodedata.normalize("NFKC", query).casefold()

    text = " ".join(temp.split())
    return text.strip("\"'?!.,;: “")


def make_key(query: str, index_version: str, prompt_version: str, model: str) -> str:
    normalized = normalize(query)
    if not normalized:
        raise ValueError(f"cannot build a cache key from an empty question: {query!r}")
    data = json.dumps([normalized, index_version, prompt_version, model])
    return KEY_PREFIX + hashlib.sha256(data.encode()).hexdigest()
