from pathlib import Path

import pytest
from pydantic import ValidationError

from codesearch.config import Settings, get_settings


def test_default_settings():
    """Verify default configurations are populated correctly."""
    settings = Settings()

    assert settings.index.react_dev_repo == "facebook/react"
    assert settings.index.corpus_dir == Path("data/")
    assert settings.retrieval.mode == "hybrid"
    assert settings.retrieval.bm25_k == 20
    assert settings.llm.provider == "fake"
    assert settings.cache.redis_url == "redis://localhost:6379/0"
    assert settings.monitoring.window_size == 500
    assert settings.admin.reindex_interval_hours == 6


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch):
    """Verify environment variables with double underscore prefix override defaults."""
    monkeypatch.setenv("CODESEARCH_RETRIEVAL__MODE", "dense")
    monkeypatch.setenv("CODESEARCH_LLM__PROVIDER", "groq")
    monkeypatch.setenv("CODESEARCH_LLM__API_KEY", "super-secret-key")

    settings = Settings()

    assert settings.retrieval.mode == "dense"
    assert settings.llm.provider == "groq"
    assert settings.llm.api_key.get_secret_value() == "super-secret-key"


def test_secret_str_masking():
    """Verify secret strings are masked when converted to string or logged."""
    settings = Settings().with_overrides(llm={"api_key": "my-secret-key"})
    secret_key = settings.llm.api_key

    assert str(secret_key) == "**********"
    assert repr(secret_key) == "SecretStr('**********')"
    assert secret_key.get_secret_value() == "my-secret-key"


def test_invalid_type_validation_error(monkeypatch: pytest.MonkeyPatch):
    """Verify invalid types fail fast at startup with a ValidationError."""
    monkeypatch.setenv("CODESEARCH_RETRIEVAL__MODE", "invalid_mode")

    with pytest.raises(ValidationError):
        Settings()


def test_with_overrides():
    """Verify method creates a new Settings instance with overrides applied."""
    base_settings = Settings()
    overridden = base_settings.with_overrides(
        retrieval={"mode": "bm25", "bm25_k": 50},
        llm={"provider": "ollama"},
    )

    # Overridden values changed
    assert overridden.retrieval.mode == "bm25"
    assert overridden.retrieval.bm25_k == 50
    assert overridden.llm.provider == "ollama"

    # Base settings remain untouched
    assert base_settings.retrieval.mode == "hybrid"
    assert base_settings.retrieval.bm25_k == 20


def test_get_settings_caching():
    """Verify get_settings returns a cached singleton instance."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
