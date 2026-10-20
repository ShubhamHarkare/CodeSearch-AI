"""Cleans one react.dev MDX page into plain markdown. Pure function, no I/O.

What goes here:
    - ``CleanPage``: the cleaned page (``title`` from frontmatter, markdown ``body``).
    - ``clean_mdx(raw) -> CleanPage``, which:
        * parses and removes the YAML frontmatter, keeping ``title``
        * keeps code fences byte-for-byte, except: the opening line is reduced to
          its language (```` ```js src/App.js active ```` -> ```` ```js ````), and
          fences marked ``hidden`` (sandbox config, package.json) are dropped
        * unwraps container components (``<Sandpack>``, ``<Recipes>``,
          ``<Challenges>``, ``<Intro>``, ...) so only their content remains
        * turns callouts into labelled text: ``<Pitfall>`` -> "Pitfall:",
          ``<Note>`` -> "Note:", ``<DeepDive>`` -> "Deep dive:", ...
        * keeps image alt text: ``<Diagram alt="...">`` -> "Diagram: ..."
        * unwraps inline components such as ``<CodeStep step={1}>x</CodeStep>``,
          but never touches anything inside `inline code`
        * removes MDX comments ``{/* ... */}``, EXCEPT on heading lines, where they
          are the explicit anchor that ``urls.parse_heading`` needs
        * collapses runs of blank lines (outside code fences) to one

What it must do:
    - Be idempotent: ``clean_mdx(clean_mdx(x).body)`` gives the same body.

Must not:
    - read files, or know about chunks or URLs.

Tested by: tests/unit/test_mdx_clean.py (small snippets + all mini-corpus pages).
Phase: 1
"""

import re
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class CleanPage:
    """A cleaned page: its frontmatter title and plain-markdown body."""

    title: str
    body: str


# ``` or ~~~ fences, possibly indented, with optional info string after them.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
# A line that is exactly one component tag: <Note>, </Note>, <InlineToc />, <Recipes a="b">
_BLOCK_TAG_RE = re.compile(
    r"^\s*<(?P<close>/?)(?P<name>[A-Z][A-Za-z0-9]*)\b(?P<attrs>[^>]*?)\s*(?P<self>/?)>\s*$"
)
_ATTR_RE = re.compile(r"""(?P<key>\w+)=(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)')""")
_MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}")
_INLINE_COMPONENT_RE = re.compile(r"<(?P<name>[A-Z][A-Za-z0-9]*)\b[^>]*>(?P<inner>.*?)</(?P=name)>")
_INLINE_CODE_RE = re.compile(r"(`+)(?:.+?)\1")
_HEADING_RE = re.compile(r"^#{1,6}\s")

# Callout components whose opening tag becomes a text label.
_LABELS = {
    "Pitfall": "Pitfall:",
    "Note": "Note:",
    "DeepDive": "Deep dive:",
    "Deprecated": "Deprecated:",
    "Hint": "Hint:",
    "Solution": "Solution:",
    "Recap": "Recap:",
    "YouWillLearn": "You will learn:",
}
# Visual components: keep their alt text, since it describes the picture.
_IMAGE_COMPONENTS = {"Diagram", "Illustration"}


def clean_mdx(raw: str) -> CleanPage:
    """Turn one raw react.dev ``.md`` page into a title and plain-markdown body."""
    title, text = _split_frontmatter(raw)
    out: list[str] = []
    fence: str | None = None  # the opening marker while inside a fence
    skipping_fence = False  # inside a ``hidden`` fence we drop every line

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)

        if fence is not None:  # ---- inside a code fence
            closes = fence_match is not None and (
                fence_match["marker"].startswith(fence) and not fence_match["info"].strip()
            )
            if not skipping_fence:
                out.append(line)
            if closes:
                fence, skipping_fence = None, False
            continue

        if fence_match is not None:  # ---- opening a code fence
            fence = fence_match["marker"]
            info = fence_match["info"].split()
            if "hidden" in info:
                skipping_fence = True
                continue
            language = info[0] if info and info[0][0] not in "{[" else ""
            out.append(f"{fence_match['indent']}{fence}{language}")
            continue

        tag_match = _BLOCK_TAG_RE.match(line)  # ---- a line that is only a component tag
        if tag_match is not None:
            _append(out, _replace_block_tag(tag_match))
            continue

        _append(out, _clean_prose_line(line))  # ---- ordinary text

    body = "\n".join(out).strip("\n")
    return CleanPage(title=title, body=body + "\n" if body else "")


def _split_frontmatter(raw: str) -> tuple[str, str]:
    """Return (title, text without frontmatter). No frontmatter -> ("", raw)."""
    if not raw.startswith("---"):
        return "", raw
    lines = raw.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            data = yaml.safe_load("".join(lines[1:i])) or {}
            title = str(data.get("title", "")) if isinstance(data, dict) else ""
            return title.strip(), "".join(lines[i + 1 :])
    return "", raw  # opening --- with no closing one: not frontmatter


def _replace_block_tag(match: re.Match[str]) -> str:
    """What a whole-line component tag becomes ("" means drop the line)."""
    name, is_closing, is_self_closing = match["name"], bool(match["close"]), bool(match["self"])
    if is_closing:
        return ""
    attrs = {
        m["key"]: (m["dq"] if m["dq"] is not None else m["sq"]).strip()
        for m in _ATTR_RE.finditer(match["attrs"])
    }
    if name in _IMAGE_COMPONENTS and attrs.get("alt"):
        return f"{name}: {attrs['alt']}"
    if is_self_closing:
        return ""  # <InlineToc />, <Solution /> (a separator inside <Recipes>)
    if name == "Recipes" and attrs.get("titleText"):
        return f"Examples: {attrs['titleText']}"
    return _LABELS.get(name, "")  # unknown containers (<Sandpack>, <Intro>, ...) vanish


def _clean_prose_line(line: str) -> str:
    """Strip MDX comments and unwrap inline components, leaving `inline code` alone."""
    if _HEADING_RE.match(line):
        return line  # keep {/*anchor*/}: parse_heading needs it

    # Swap inline code spans for placeholders so nothing below can touch them.
    spans: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    text = _INLINE_CODE_RE.sub(_stash, line)
    text = _MDX_COMMENT_RE.sub("", text)
    previous = None
    while previous != text:  # repeat to unwrap nested inline components
        previous, text = text, _INLINE_COMPONENT_RE.sub(lambda m: m["inner"], text)
    text = re.sub("\x00(\\d+)\x00", lambda m: spans[int(m.group(1))], text)

    return text.rstrip() if text.strip() else ""


def _append(out: list[str], line: str) -> None:
    """Append a line, never letting two blank lines stack up."""
    if line == "" and (not out or out[-1] == ""):
        return
    out.append(line)
