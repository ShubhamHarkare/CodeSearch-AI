"""Tests for codesearch/cache/keys.py: turning a question into a cache key.

Two goals pull in opposite directions:
  * the same question typed slightly differently should hit the cache
    ("How do I use useState?" vs "how do i use usestate"), and
  * an answer must never be served after anything that shaped it changed:
    the index (re-ingest), the prompt (new version) or the model.
"""

from __future__ import annotations

import pytest

from codesearch.cache.keys import KEY_PREFIX, make_key, normalize

V = {"index_version": "ac462615f149", "prompt_version": "1", "model": "gemma3:4b"}

# =====================================================================
# normalize
# =====================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("How do I use useState?", "how do i use usestate"),
        ("  how do I use useState  ", "how do i use usestate"),
        ("How  do\tI\nuse   useState?", "how do i use usestate"),
        ("How do I use useState???", "how do i use usestate"),
        ("How do I use useState?!.", "how do i use usestate"),
        ('"How do I use useState?"', "how do i use usestate"),
        ("'How do I use useState'", "how do i use usestate"),
    ],
    ids=["basic", "outer-spaces", "inner-whitespace", "repeated-?", "mixed-end", "dq", "sq"],
)
def test_normalize_variants_collapse_to_one_form(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_inner_punctuation_is_kept() -> None:
    """useState() vs useState, or "props.children": inner punctuation carries meaning."""
    assert normalize("What does useState() return?") == "what does usestate() return"
    assert normalize("What is props.children?") == "what is props.children"


def test_different_questions_stay_different() -> None:
    assert normalize("How do I use useState?") != normalize("How do I use useReducer?")


def test_unicode_compatibility_forms_are_folded() -> None:
    """Full-width letters (pasted from some sites) normalize to plain ones."""
    full_width = "\uff55\uff53\uff45\uff33\uff54\uff41\uff54\uff45?"  # "useState?"
    assert normalize(full_width) == "usestate"


def test_only_punctuation_or_whitespace_normalizes_to_empty() -> None:
    assert normalize("  ?!  ") == ""
    assert normalize("") == ""


def test_normalize_is_idempotent() -> None:
    once = normalize("  How do I use useState?? ")
    assert normalize(once) == once


# =====================================================================
# make_key
# =====================================================================


def test_key_has_the_prefix_and_a_sha256_hex_digest() -> None:
    key = make_key("How do I use useState?", **V)
    assert KEY_PREFIX == "codesearch:q:"
    assert key.startswith(KEY_PREFIX)
    digest = key.removeprefix(KEY_PREFIX)
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_key_is_deterministic() -> None:
    assert make_key("How do I use useState?", **V) == make_key("How do I use useState?", **V)


def test_variants_of_one_question_share_a_key() -> None:
    assert make_key("How do I use useState?", **V) == make_key("  how do i use usestate ", **V)


def test_different_questions_get_different_keys() -> None:
    assert make_key("How do I use useState?", **V) != make_key("How do I use useReducer?", **V)


@pytest.mark.parametrize("field", ["index_version", "prompt_version", "model"])
def test_changing_any_version_changes_the_key(field: str) -> None:
    """Re-ingest, prompt edit or model switch: old answers must not be served."""
    changed = {**V, field: V[field] + "-new"}
    assert make_key("How do I use useState?", **V) != make_key("How do I use useState?", **changed)


def test_fields_cannot_bleed_into_each_other() -> None:
    """A naive "a|b" join would give these two the same key."""
    a = make_key("q", index_version="v1|2", prompt_version="3", model="m")
    b = make_key("q", index_version="v1", prompt_version="2|3", model="m")
    assert a != b


@pytest.mark.parametrize("query", ["", "   ", "?!"])
def test_a_question_with_nothing_left_after_normalizing_is_rejected(query: str) -> None:
    with pytest.raises(ValueError):
        make_key(query, **V)
