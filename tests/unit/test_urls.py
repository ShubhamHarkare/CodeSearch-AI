"""Tests for codesearch/ingest/urls.py."""

from pathlib import Path

import pytest

from codesearch.ingest.urls import Heading, parse_heading, path_to_url, slugify, url_with_anchor

# =====================================================================
# Standard URL Generation Tests
# =====================================================================


def test_standard_nested_path(tmp_path):
    content_root = tmp_path / "docs"
    file_path = content_root / "learn" / "hooks.md"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/learn/hooks"


def test_deeply_nested_path(tmp_path):
    content_root = tmp_path / "docs"
    file_path = content_root / "reference" / "react" / "useState.mdx"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/reference/react/useState"


# =====================================================================
# Index File Handling Tests
# =====================================================================


def test_root_index_file(tmp_path):
    content_root = tmp_path / "docs"
    file_path = content_root / "index.md"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/"


def test_nested_index_file(tmp_path):
    content_root = tmp_path / "docs"
    file_path = content_root / "learn" / "index.md"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/learn"


def test_deeply_nested_index_file(tmp_path):
    content_root = tmp_path / "docs"
    file_path = content_root / "reference" / "react" / "index.md"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/reference/react"


# =====================================================================
# Edge Cases & Boundary Conditions
# =====================================================================


def test_not_relative_path_raises_value_error(tmp_path):
    content_root = tmp_path / "docs"
    outside_file = tmp_path / "other_folder" / "page.md"

    with pytest.raises(ValueError):
        path_to_url(outside_file, content_root)


def test_files_containing_index_in_name(tmp_path):
    """Ensure files with 'index' as a substring don't accidentally get stripped."""
    content_root = tmp_path / "docs"

    # Files like 'my-index.md', 'indexation.md', or directory named 'index'
    f1 = content_root / "my-index.md"
    f2 = content_root / "indexation.md"
    f3 = content_root / "index" / "page.md"

    assert path_to_url(f1, content_root) == "https://react.dev/my-index"
    assert path_to_url(f2, content_root) == "https://react.dev/indexation"
    assert path_to_url(f3, content_root) == "https://react.dev/index/page"


def test_different_file_extensions(tmp_path):
    content_root = tmp_path / "docs"

    assert path_to_url(content_root / "page.html", content_root) == "https://react.dev/page"
    assert path_to_url(content_root / "page.txt", content_root) == "https://react.dev/page"
    assert path_to_url(content_root / "page", content_root) == "https://react.dev/page"


def test_filename_with_multiple_dots(tmp_path):
    """pathlib with_suffix() only removes the last extension suffix."""
    content_root = tmp_path / "docs"
    file_path = content_root / "blog" / "v1.0.release.md"

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/blog/v1.0.release"


def test_windows_path_separator_conversion():
    """Verify that backslashes on Windows are converted to posix forward slashes."""
    content_root = Path("C:/Users/docs")
    file_path = Path("C:/Users/docs/learn/hooks.md")

    url = path_to_url(file_path, content_root)
    assert url == "https://react.dev/learn/hooks"
    assert "\\" not in url


# =====================================================================
# Real mini-corpus files
# =====================================================================

MINI_CORPUS = Path(__file__).parents[1] / "fixtures" / "mini_corpus"


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("reference/react/useState.md", "https://react.dev/reference/react/useState"),
        ("learn/index.md", "https://react.dev/learn"),
        (
            "reference/react-dom/components/input.md",
            "https://react.dev/reference/react-dom/components/input",
        ),
    ],
)
def test_mini_corpus_urls(rel_path, expected):
    path = MINI_CORPUS / rel_path
    assert path.exists(), f"fixture missing: {rel_path}"
    assert path_to_url(path, MINI_CORPUS) == expected


# =====================================================================
# parse_heading
# =====================================================================


def test_explicit_anchor_wins_over_slug():
    assert parse_heading("## Creating and nesting components {/*components*/}") == (
        2,
        "Creating and nesting components",
        "components",
    )


def test_backticks_are_stripped_from_text():
    heading = parse_heading("### `useState(initialState)` {/*usestate*/}")
    assert heading == Heading(level=3, text="useState(initialState)", anchor="usestate")


def test_fallback_slug_when_no_explicit_anchor():
    heading = parse_heading("## Why these pages")
    assert heading is not None
    assert heading.anchor == "why-these-pages"


@pytest.mark.parametrize(
    "line",
    ["Some paragraph", "#nospace", "####### seven hashes", "", "   ## indented"],
)
def test_non_headings_return_none(line):
    assert parse_heading(line) is None


def test_every_heading_in_the_mini_corpus_parses_with_an_anchor():
    """Real-data check: all '## ' / '### ' lines outside code fences have an anchor."""
    for page in MINI_CORPUS.rglob("*.md"):
        if page.name in {"README.md", "LICENSE-DOCS.md"}:
            continue
        in_fence = False
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                in_fence = not in_fence
            if in_fence or not line.startswith("#"):
                continue
            heading = parse_heading(line)
            assert heading is not None, f"{page.name}: {line!r}"
            assert heading.anchor, f"{page.name}: empty anchor for {line!r}"


# =====================================================================
# slugify / url_with_anchor
# =====================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Why these pages", "why-these-pages"),
        ("Adding state to a component!", "adding-state-to-a-component"),
        ("useState(initialState)", "usestateinitialstate"),
        ("  extra   spaces  ", "extra-spaces"),
    ],
)
def test_slugify(text, expected):
    assert slugify(text) == expected


def test_url_with_anchor():
    url = "https://react.dev/reference/react/useState"
    assert url_with_anchor(url, "usestate") == f"{url}#usestate"
    assert url_with_anchor(url, "") == url
