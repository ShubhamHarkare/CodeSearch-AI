"""Guards .env.example: every variable it sets must reach Settings unchanged.

The quick start copies .env.example to .env, so a bad line there breaks a
fresh clone (an empty value overriding a pinned default, or a name without the
CODESEARCH_ prefix that pydantic-settings silently ignores).
"""

from pathlib import Path

import pytest

from codesearch.config import IndexSettings, Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _assignments() -> dict[str, str]:
    """Return {NAME: value} for each uncommented line, inline comments removed."""
    out: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        out[name.strip()] = value.split("#", 1)[0].strip()
    return out


def test_every_variable_has_the_prefix() -> None:
    names = _assignments()
    assert names, ".env.example sets no variables"
    unprefixed = [n for n in names if not n.startswith("CODESEARCH_")]
    assert unprefixed == []  # pydantic-settings ignores these silently


def test_no_variable_is_set_to_empty() -> None:
    empty = [n for n, v in _assignments().items() if v == ""]
    assert empty == []  # an empty value overrides the default with ""


def test_every_variable_maps_to_a_real_setting() -> None:
    """CODESEARCH_A__B must name settings.a.b, or extra='ignore' drops it."""
    dumped = Settings().model_dump()
    for name in _assignments():
        node = dumped
        for part in name.removeprefix("CODESEARCH_").lower().split("__"):
            assert isinstance(node, dict) and part in node, f"{name} sets nothing"
            node = node[part]


def test_settings_from_env_example_keep_the_pinned_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(_assignments()):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert s.index.react_dev_sha == IndexSettings().react_dev_sha
    assert s.cache.redis_url == "redis://localhost:6379/0"
    assert (s.llm.provider, s.llm.model) == ("openai", "openai/gpt-oss-20b")  # Groq
    assert s.llm.reasoning_effort == "low"
    assert s.llm.api_key.get_secret_value() == ""  # the key is never in the example
