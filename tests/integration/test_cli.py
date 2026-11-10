"""Tests for codesearch/cli.py, through Typer's CliRunner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from codesearch.cli import app
from codesearch.config import get_settings
from tests.conftest import FakeReactDev

runner = CliRunner()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_react_dev: FakeReactDev) -> Path:
    """Point the CLI's settings at a temp folder and the fake react.dev."""
    monkeypatch.setenv("CODESEARCH_INDEX__CORPUS_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODESEARCH_INDEX__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("CODESEARCH_INDEX__REACT_DEV_REPO", fake_react_dev.url)
    monkeypatch.setenv("CODESEARCH_INDEX__REACT_DEV_SHA", fake_react_dev.head)
    get_settings.cache_clear()
    yield tmp_path / "index"
    get_settings.cache_clear()


def test_ingest_builds_the_index_and_reports_it(env: Path) -> None:
    result = runner.invoke(app, ["ingest", "--embedder", "hash"])
    assert result.exit_code == 0, result.output
    assert "Index built" in result.output
    assert "pages          22" in result.output
    assert (env / "manifest.json").exists()


def test_second_ingest_reports_no_changes(env: Path) -> None:
    runner.invoke(app, ["ingest", "--embedder", "hash"])
    result = runner.invoke(app, ["ingest", "--embedder", "hash"])
    assert result.exit_code == 0, result.output
    assert "+0 added, -0 removed" in result.output


def test_index_info_prints_the_manifest(env: Path) -> None:
    runner.invoke(app, ["ingest", "--embedder", "hash"])
    result = runner.invoke(app, ["index-info"])
    assert result.exit_code == 0
    assert '"embedding_model": "hash"' in result.output


def test_index_info_without_an_index(env: Path) -> None:
    result = runner.invoke(app, ["index-info"])
    assert result.exit_code == 1
    assert "codesearch ingest" in result.output


def test_ingest_failure_is_a_clean_error(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESEARCH_INDEX__REACT_DEV_SHA", "0" * 40)
    get_settings.cache_clear()
    result = runner.invoke(app, ["ingest", "--embedder", "hash"])
    assert result.exit_code == 1
    assert "Ingest failed" in result.output
    assert "Traceback" not in result.output


# =====================================================================
# codesearch ask
# =====================================================================


@pytest.fixture
def answering(env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A built hash index, the fake LLM, and a threshold that always answers."""
    monkeypatch.setenv("CODESEARCH_INDEX__EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("CODESEARCH_LLM__PROVIDER", "fake")
    monkeypatch.setenv("CODESEARCH_RETRIEVAL__ABSTAIN_THRESHOLD", "-2")
    get_settings.cache_clear()
    assert runner.invoke(app, ["ingest"]).exit_code == 0
    return env


def test_ask_prints_the_answer_and_its_sources(answering: Path) -> None:
    result = runner.invoke(app, ["ask", "How do I use useState?"])
    assert result.exit_code == 0, result.output
    assert "Use useState to add state" in result.output  # FakeLLM's default answer
    assert "Sources" in result.output
    assert "[1] https://react.dev/" in result.output
    assert "[2] https://react.dev/" in result.output


def test_ask_says_when_it_abstains(answering: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESEARCH_RETRIEVAL__ABSTAIN_THRESHOLD", "2")
    get_settings.cache_clear()
    result = runner.invoke(app, ["ask", "How do I write a computed property in Vue?"])
    assert result.exit_code == 0, result.output
    assert "couldn't find this in the React docs" in result.output
    assert "Sources" not in result.output


def test_ask_without_an_index_tells_you_to_ingest(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODESEARCH_LLM__PROVIDER", "fake")
    get_settings.cache_clear()
    result = runner.invoke(app, ["ask", "How do I use useState?"])
    assert result.exit_code == 1
    assert "codesearch ingest" in result.output


def test_ask_numbers_sources_by_the_citation_number(
    answering: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sources must use the [n] the answer cites, not 1, 2, 3 by position.

    An answer citing [3] and [1] must list "[3] ..." and "[1] ...", never "[2]".
    """
    from codesearch.generation.llm import FakeLLM

    monkeypatch.setattr(FakeLLM, "DEFAULT_ANSWER", "Queue it [3]. Then render [1].")
    result = runner.invoke(app, ["ask", "How do I use useState?"])
    assert result.exit_code == 0, result.output
    sources = result.output.split("Sources")[1]
    assert "[3] https://react.dev/" in sources
    assert "[1] https://react.dev/" in sources
    assert "[2]" not in sources
