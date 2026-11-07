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
