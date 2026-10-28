"""Tests for codesearch/ingest/manifest.py."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from codesearch.ingest.manifest import (
    PIPELINE_VERSION,
    ChangeLogEntry,
    ChunkParams,
    Manifest,
    compute_index_version,
    read_manifest,
    write_manifest,
)

SHA = "046f17d04295ba047bb5739026b4ac3110f17028"
MODEL = "BAAI/bge-small-en-v1.5"
PARAMS = ChunkParams(max_chars=1000, overlap=0)


def make_manifest(**overrides: object) -> Manifest:
    fields: dict[str, object] = {
        "index_version": compute_index_version(SHA, MODEL, PARAMS),
        "corpus_sha": SHA,
        "embedding_model": MODEL,
        "embedding_dim": 384,
        "chunk_params": PARAMS,
        "pipeline_version": PIPELINE_VERSION,
        "chunk_count": 809,
        "built_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
        "changelog": [
            ChangeLogEntry(
                at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
                corpus_sha=SHA,
                added=809,
                removed=0,
                unchanged=0,
            )
        ],
    }
    fields.update(overrides)
    return Manifest.model_validate(fields)


# =====================================================================
# compute_index_version
# =====================================================================


def test_same_inputs_give_the_same_version() -> None:
    assert compute_index_version(SHA, MODEL, PARAMS) == compute_index_version(
        SHA, MODEL, ChunkParams(max_chars=1000, overlap=0)
    )


def test_version_is_twelve_lowercase_hex_chars() -> None:
    assert re.fullmatch(r"[0-9a-f]{12}", compute_index_version(SHA, MODEL, PARAMS))


def test_default_pipeline_version_is_the_module_constant() -> None:
    assert compute_index_version(SHA, MODEL, PARAMS) == compute_index_version(
        SHA, MODEL, PARAMS, PIPELINE_VERSION
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"corpus_sha": "1" * 40},
        {"embedding_model": "hash"},
        {"chunk_params": ChunkParams(max_chars=800, overlap=0)},
        {"chunk_params": ChunkParams(max_chars=1000, overlap=150)},
        {"pipeline_version": "2"},
    ],
    ids=["sha", "model", "max_chars", "overlap", "pipeline_version"],
)
def test_any_input_change_gives_a_new_version(changed: dict[str, object]) -> None:
    base: dict[str, object] = {
        "corpus_sha": SHA,
        "embedding_model": MODEL,
        "chunk_params": PARAMS,
        "pipeline_version": PIPELINE_VERSION,
    }
    assert compute_index_version(**{**base, **changed}) != compute_index_version(**base)  # type: ignore[arg-type]


def test_swapped_values_do_not_collide() -> None:
    """Keys are part of the hash, so max_chars=150/overlap=1000 differs from the reverse."""
    a = compute_index_version(SHA, MODEL, ChunkParams(max_chars=1000, overlap=150))
    b = compute_index_version(SHA, MODEL, ChunkParams(max_chars=150, overlap=1000))
    assert a != b


# =====================================================================
# write_manifest / read_manifest
# =====================================================================


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    manifest = make_manifest()
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded == manifest
    assert loaded is not None and loaded.built_at.tzinfo is not None


def test_missing_manifest_is_none(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) is None
    assert read_manifest(tmp_path / "does" / "not" / "exist") is None


def test_write_creates_missing_folders(tmp_path: Path) -> None:
    index_dir = tmp_path / "a" / "b" / "index"
    write_manifest(index_dir, make_manifest())
    assert (index_dir / "manifest.json").exists()


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_second_write_replaces_the_first(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest(chunk_count=1))
    write_manifest(tmp_path, make_manifest(chunk_count=2))
    loaded = read_manifest(tmp_path)
    assert loaded is not None and loaded.chunk_count == 2


def test_stale_temp_file_from_a_crash_is_overwritten(tmp_path: Path) -> None:
    (tmp_path / "manifest.json.tmp").write_text("half-written garbage")
    write_manifest(tmp_path, make_manifest())
    assert read_manifest(tmp_path) == make_manifest()
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_written_file_is_readable_json(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest())
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["corpus_sha"] == SHA
    assert data["chunk_params"] == {"max_chars": 1000, "overlap": 0}


@pytest.mark.parametrize(
    "content",
    ["not json at all", "{}", '{"index_version": "x"}', ""],
    ids=["garbage", "empty-object", "missing-fields", "empty-file"],
)
def test_corrupt_manifest_raises_loudly(tmp_path: Path, content: str) -> None:
    (tmp_path / "manifest.json").write_text(content)
    with pytest.raises(ValidationError):
        read_manifest(tmp_path)


def test_changelog_defaults_to_empty_and_is_not_shared() -> None:
    first = make_manifest(changelog=[])
    second = Manifest.model_validate(
        {k: v for k, v in make_manifest().model_dump().items() if k != "changelog"}
    )
    assert second.changelog == []
    first.changelog.append(
        ChangeLogEntry(at=datetime.now(UTC), corpus_sha=SHA, added=1, removed=0, unchanged=0)
    )
    assert second.changelog == []
