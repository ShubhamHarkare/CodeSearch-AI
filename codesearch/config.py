from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IndexSettings(BaseModel):
    """
    Settings related to database ingestion and index storage
    """

    react_dev_repo: str = "https://github.com/reactjs/react.dev.git"
    react_dev_sha: str = "046f17d04295ba047bb5739026b4ac3110f17028"
    corpus_dir: Path = Path("data/")
    index_dir: Path = Path("index/")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_chars: int = 1000
    chunk_overlap: int = 100


class RetrievalSettings(BaseModel):
    """Settings for vector search using BM25, and reranking parameters"""

    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    bm25_k: int = 20
    dense_k: int = 20
    rrf_k: int = 60
    reranker_model: str | None = None
    rerank_top_n: int = 5
    abstain_threshold: float = 0.5
    history_turns: int = 3


class LLMSettings(BaseModel):
    """LLM provider and inference configurations."""

    provider: Literal["groq", "ollama", "fake"] = "fake"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 30.0
    max_retries: int = 3
    api_key: SecretStr = SecretStr("")


class CacheSettings(BaseModel):
    """Redis and semantic caching configurations."""

    redis_url: str = "redis://localhost:6379/0"
    ttl_hours: int = 24
    semantic_threshold: float = 0.95
    semantic_max_entries: int = 1000


class MonitoringSettings(BaseModel):
    """Metrics tracking and drift detection thresholds."""

    window_size: int = 500
    drift_threshold: float = 0.2


class AdminSettings(BaseModel):
    """Administrative secrets and background task scheduling."""

    webhook_secret: SecretStr = SecretStr("")
    reindex_interval_hours: int = 6


class Settings(BaseSettings):
    """Root configuration object for CodeSearch AI."""

    model_config = SettingsConfigDict(
        env_prefix="CODESEARCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    admin: AdminSettings = Field(default_factory=AdminSettings)

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a new Settings instance with specified field overrides.

        Useful for unit tests, evaluation setups, or benchmarking runs.
        Supports passing sub-dictionaries (e.g., llm={"provider": "groq"}).
        """
        current_data = self.model_dump()
        for key, value in kwargs.items():
            if (
                key in current_data
                and isinstance(current_data[key], dict)
                and isinstance(value, dict)
            ):
                current_data[key].update(value)
            else:
                current_data[key] = value
        return Settings.model_validate(current_data)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, singleton instance of the application Settings."""
    return Settings()
