"""Shared pytest fixtures. Tests must run offline in seconds.

Fixtures:
    - ``_isolate_env`` (automatic): keeps the developer's CODESEARCH_* env vars out.
    - ``mini_corpus_path``: tests/fixtures/mini_corpus (22 real react.dev pages).
    - ``hash_embedder``: HashEmbedder(dim=64), no downloads.
    - ``fake_llm``: FakeLLM with its default citing answer.
    - ``settings``: Settings with data/ and index/ under tmp_path, the hash
      embedder and the fake LLM, and the developer's .env ignored.
    - ``fake_react_dev``: a local git repo laid out like react.dev
      (src/content/...) holding the mini corpus, so build tests never touch
      the network. Use ``.url``, ``.head``, ``.commit({...})``.

Rules: no network, no real models, no Redis, unless a test is marked
``slow`` or ``needs_llm``.
"""

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from codesearch.config import Settings
from codesearch.generation.llm import FakeLLM
from codesearch.index.embedder import HashEmbedder

MINI_CORPUS = Path(__file__).parent / "fixtures" / "mini_corpus"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real env vars out of every test."""
    for key in list(os.environ):
        if key.startswith("CODESEARCH_"):
            monkeypatch.delenv(key)


@pytest.fixture
def mini_corpus_path() -> Path:
    return MINI_CORPUS


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={
            "corpus_dir": tmp_path / "data",
            "index_dir": tmp_path / "index",
            "embedding_model": "hash",
        },
        llm={"provider": "fake"},
    )


# ---------------------------------------------------------------------
# A fake react.dev git repository
# ---------------------------------------------------------------------


def run_git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` with a throwaway identity (CI runners have none)."""
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class FakeReactDev:
    """A local git repo shaped like react.dev, seeded with the mini corpus."""

    def __init__(self, root: Path) -> None:
        self.path = root
        self.content = root / "src" / "content"
        self.content.mkdir(parents=True)
        shutil.copytree(MINI_CORPUS, self.content, dirs_exist_ok=True)
        run_git(root, "init", "--quiet", "-b", "main")
        run_git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
        self.head = self.commit({}, "import mini corpus")

    @property
    def url(self) -> str:
        return f"file://{self.path}"

    def commit(self, files: Mapping[str, str | None], message: str = "update") -> str:
        """Write (or delete, for None) files under src/content, commit, return the SHA."""
        for rel, text in files.items():
            target = self.content / rel
            if text is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
        run_git(self.path, "add", "-A")
        run_git(self.path, "commit", "--quiet", "--allow-empty", "-m", message)
        self.head = run_git(self.path, "rev-parse", "HEAD")
        return self.head

    def read(self, rel: str) -> str:
        return (self.content / rel).read_text(encoding="utf-8")


@pytest.fixture
def fake_react_dev(tmp_path: Path) -> FakeReactDev:
    return FakeReactDev(tmp_path / "upstream")
