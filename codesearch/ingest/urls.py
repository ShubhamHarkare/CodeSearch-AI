"""Maps a react.dev source file to its live URL and heading anchors. Pure.

What goes here:
    - ``path_to_url(path: Path, content_root: Path) -> str``, e.g.
      ``src/content/reference/react/useState.md`` ->
      ``https://react.dev/reference/react/useState``; ``index.md`` maps to the
      folder URL.
    - ``heading_anchor(heading_line: str) -> str``: prefer the explicit
      ``{/*anchor*/}`` id; otherwise slugify the way react.dev does.
    - ``url_with_anchor(url, anchor) -> str``.

Why: v1 cited local paths like react.dev/src/content/...; v2 citations must be
links a user can click.

Tested by: tests/unit/test_urls.py (reference page, learn page, index page,
explicit anchor, slugified anchor).
Phase: 1
"""
