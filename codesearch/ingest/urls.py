"""Maps a react.dev source file to its live URL and heading anchors. Pure.

What goes here:
    - ``path_to_url(path: Path, content_root: Path) -> str``, e.g.
      ``src/content/reference/react/useState.md`` ->
      ``https://react.dev/reference/react/useState``; ``index.md`` maps to the
      folder URL.
    - ``parse_heading(line: str) -> Heading | None``: split a markdown heading
      into level, clean text (backticks removed) and anchor. The explicit
      ``{/*anchor*/}`` id wins; otherwise ``slugify(text)`` is the fallback.
      Returns None for non-headings. Code fences are the chunker's job: this
      function only ever sees one line.
    - ``Heading``: NamedTuple(level, text, anchor).
    - ``url_with_anchor(url, anchor) -> str``.

Why: v1 cited local paths like react.dev/src/content/...; v2 citations must be
links a user can click.

Tested by: tests/unit/test_urls.py (reference page, learn page, index page,
explicit anchor, slugified anchor).
Phase: 1
"""

import re
from pathlib import Path
from typing import NamedTuple

BASE_URL = "https://react.dev"


def path_to_url(path: Path, content_root: Path) -> str:
    """Turn a react.dev source file path into its live page URL."""

    relative = path.relative_to(content_root)

    slug = relative.with_suffix("").as_posix()

    if slug == "index":
        slug = ""
    elif slug.endswith("/index"):
        slug = slug.removesuffix("/index")

    # Glue it onto the website address.
    return f"{BASE_URL}/{slug}"


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{/\*(.*?)\*/\})?\s*$")


class Heading(NamedTuple):
    level: int
    text: str
    anchor: str


def slugify(text: str) -> str:
    """Fallback anchor: 'Why these pages!' -> 'why-these-pages'."""
    text = text.lower()
    # Drop anything that isn't a letter, digit, space or hyphen.
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)  # spaces (and runs of hyphens) -> one hyphen
    return text.strip("-")


def parse_heading(line: str) -> Heading | None:
    """Split a markdown heading line into (level, text, anchor); None if not a heading."""
    match = HEADING_RE.match(line)
    if match is None:
        return None
    hashes, raw_text, explicit_anchor = match.groups()
    text = raw_text.replace("`", "").strip()
    anchor = explicit_anchor or slugify(text)  # explicit {/*...*/} wins
    return Heading(level=len(hashes), text=text, anchor=anchor)


def url_with_anchor(url: str, anchor: str) -> str:
    """'https://react.dev/learn' + 'components' -> 'https://react.dev/learn#components'."""
    return f"{url}#{anchor}" if anchor else url
