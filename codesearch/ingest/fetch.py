"""Gets the react.dev source at a pinned commit. Replaces v1's dataScraper.py.

What goes here:
    - ``CorpusSnapshot``: what was fetched (repo dir, content root, exact commit,
      page paths, fetch time). ``build.py`` turns it into chunks and a manifest.
    - ``fetch_corpus(repo_url, sha, dest) -> CorpusSnapshot``: download exactly
      one commit with ``git fetch --depth 1`` and check it out. The first run
      creates the repo; later runs reuse it and just move to the new commit.
      ``sha=None`` means "latest main" (for live re-indexing later).
    - ``list_pages(content_root) -> list[Path]``: every ``.md`` / ``.mdx`` page,
      sorted so runs are reproducible, skipping README/LICENSE files.
    - ``run_git(args, cwd) -> str``: the only place that runs git. Every failure
      (bad exit code, timeout, git missing) becomes ``CorpusFetchError``.

Why ``init`` + ``fetch`` and not ``git clone``: clone cannot shallow-download a
specific commit; ``fetch --depth 1 origin <sha>`` can, and the same code path
serves the first run and every update.

Must not:
    - run anything at import time, or clean / chunk text (that is mdx_clean / chunker).

Tested by: tests/unit/test_fetch.py, against a tiny local git repo built in tmp_path.
Phase: 1
"""

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codesearch.errors import CorpusFetchError

GIT_TIMEOUT_S = 120
PAGE_SUFFIXES = (".md", ".mdx")
NON_PAGES = frozenset({"README.md", "LICENSE-DOCS.md"})
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class CorpusSnapshot:
    """One checked-out commit of react.dev and the pages it contains."""

    repo_dir: Path  # data/react.dev
    content_root: Path  # data/react.dev/src/content (what path_to_url needs)
    commit_sha: str  # the exact commit checked out; goes into the manifest
    page_paths: tuple[Path, ...]
    fetched_at: datetime


def fetch_corpus(repo_url: str, sha: str | None, dest: Path) -> CorpusSnapshot:
    """Check out exactly ``sha`` (or the latest ``main`` if None) of ``repo_url`` into ``dest``."""
    if sha is not None and not _FULL_SHA_RE.fullmatch(sha):
        raise CorpusFetchError(f"expected a full 40-character lowercase commit SHA, got {sha!r}")
    dest = Path(dest)

    if not (dest / ".git").exists():  # first run
        if dest.exists() and any(dest.iterdir()):
            raise CorpusFetchError(f"{dest} is not empty and is not a git repository")
        dest.mkdir(parents=True, exist_ok=True)
        run_git(["init", "--quiet"], cwd=dest)
        run_git(["remote", "add", "origin", repo_url], cwd=dest)
    else:  # later runs: the URL in config may have changed
        run_git(["remote", "set-url", "origin", repo_url], cwd=dest)

    ref = sha or "main"
    run_git(["fetch", "--quiet", "--depth", "1", "origin", ref], cwd=dest)  # only that commit
    run_git(["checkout", "--quiet", "--force", "--detach", "FETCH_HEAD"], cwd=dest)

    actual = run_git(["rev-parse", "HEAD"], cwd=dest)
    if sha is not None and actual != sha:
        raise CorpusFetchError(f"requested commit {sha} but checked out {actual}")

    content_root = dest / "src" / "content"
    if not content_root.is_dir():
        raise CorpusFetchError(f"commit {actual} has no src/content directory")
    pages = tuple(list_pages(content_root))
    if not pages:
        raise CorpusFetchError(f"commit {actual} has no .md/.mdx pages under src/content")

    return CorpusSnapshot(
        repo_dir=dest,
        content_root=content_root,
        commit_sha=actual,
        page_paths=pages,
        fetched_at=datetime.now(UTC),
    )


def list_pages(content_root: Path) -> list[Path]:
    """All .md/.mdx pages under ``content_root``, sorted; README/LICENSE files skipped."""
    if not content_root.is_dir():
        return []
    return sorted(
        path
        for path in content_root.rglob("*")
        if path.is_file() and path.suffix in PAGE_SUFFIXES and path.name not in NON_PAGES
    )


def run_git(args: list[str], cwd: Path | None = None) -> str:
    """Run one git command and return its stdout; any failure raises CorpusFetchError."""
    command = f"git {' '.join(args)}"
    # Never stop to ask for a username/password: a bad URL must fail, not hang.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        raise CorpusFetchError(f"{command} failed: {(e.stderr or '').strip()}") from e
    except subprocess.TimeoutExpired as e:
        # e.stderr is bytes here even with text=True, so don't print it.
        raise CorpusFetchError(f"{command} timed out after {GIT_TIMEOUT_S} seconds") from e
    except FileNotFoundError as e:
        raise CorpusFetchError(f"{command} failed: git is not installed or not on PATH") from e
    return result.stdout.strip()
