"""Splits a cleaned page into heading-aware chunks with stable IDs. Pure.

How it works (three small functions):
    1. ``split_into_sections(page)``: walk the page once and cut it at every
       #, ## and ### heading outside code blocks. Each ``Section`` remembers its
       header path, e.g. ["useState", "Usage", "Adding state"], and its anchor.
    2. ``split_long_section(lines, max_chars)``: if a section is too long, cut it
       only at blank lines OUTSIDE code blocks. A code block is never split; one
       that is bigger than ``max_chars`` becomes a piece on its own.
    3. ``chunk_page(page, url, source_path, max_chars)``: turn every piece into a
       ``Chunk`` whose text starts with the header path
       ("useState > Usage > Adding state"), so the embedder and BM25 see the
       context, with ``id = sha256(url | path | piece index | text)``.

Why not LangChain's splitters (v1): on useState.md they cut a code example in
half in 26 of 55 chunks, kept the header path only in metadata, and gave no IDs.

Stable IDs: the same content always gets the same id, and editing one section
changes only that section's chunk ids. ingest/build.py relies on this to
re-embed only what changed.

Not yet: ``overlap`` is accepted but not applied (sections already start with
their header path); measure it with the eval harness before turning it on.

Tested by: tests/unit/test_chunker.py (unit cases + every mini-corpus page).
Phase: 1
"""

import hashlib
from dataclasses import dataclass

from codesearch.ingest.mdx_clean import CleanPage
from codesearch.ingest.urls import parse_heading
from codesearch.schemas import Chunk


@dataclass
class Section:
    """One heading's worth of lines, before size splitting."""

    path: list[str]
    anchor: str
    lines: list[str]


def split_into_sections(page: CleanPage) -> list[Section]:
    """Cut a page into sections at #/##/### headings outside code blocks."""
    title = [page.title] if page.title else []
    path: list[str] = list(title)
    open_headings: list[tuple[int, str]] = []  # (level, text) of the headings we're under
    anchor: str = ""
    current: list[str] = []
    in_code: bool = False
    sections: list[Section] = []

    lines = page.body.splitlines()

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            current.append(line)

            continue

        if in_code:
            current.append(line)
            continue

        heading = parse_heading(line)
        if heading and heading.level <= 3:
            if any(ln.strip() for ln in current):
                sections.append(Section(path=path, anchor=anchor, lines=current))

            ## Starting a new section here
            # Close every heading at this level or deeper, then open this one.
            # (Tracking levels, not list positions: react.dev pages start at ##.)
            open_headings = [h for h in open_headings if h[0] < heading.level]
            open_headings.append((heading.level, heading.text))
            path = title + [text for _, text in open_headings]
            anchor = heading.anchor
            current = []
        else:
            current.append(line)

    if any(ln.strip() for ln in current):
        sections.append(Section(path=path, anchor=anchor, lines=current))

    return sections


def split_long_section(lines: list[str], max_chars: int, overlap: int = 0) -> list[str]:
    """Split a section into pieces of up to max_chars, never inside a code block."""
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return [text]

    # Step - 1 Converting lines into blocks
    blocks: list[str] = []
    current: list[str] = []
    in_code = False

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code  # no `continue`: the fence line itself must be kept
        if not line.strip() and not in_code:
            # A blank line outside code is a safe place to cut.
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:  # save whatever is left after the last blank line
        blocks.append("\n".join(current))

    # A prose block with no blank lines (a long list) can still exceed max_chars:
    # cut it at line breaks. Code blocks are never cut, even when oversized.
    blocks = [part for block in blocks for part in _split_prose_block(block, max_chars)]

    # Step-2 convert these blocks to pieces
    pieces: list[str] = []
    piece: str = ""

    for block in blocks:
        if not piece:
            piece = block

        elif len(piece) + 2 + len(block) <= max_chars:
            piece = piece + "\n\n" + block
        else:
            # The block doesn't fit: save the current piece, start a new one with this block.
            pieces.append(piece)
            piece = block
    if piece:  # `piece` is a str, so check "non-empty", not "is not None"
        pieces.append(piece)

    return pieces


def chunk_page(
    page: CleanPage, url: str, source_path: str, max_chars: int = 1000, overlap: int = 0
) -> list[Chunk]:
    """Turn one cleaned page into Chunks with header-path prefixes and stable IDs."""
    chunks: list[Chunk] = []
    for section in split_into_sections(page):
        prefix = " > ".join(section.path)  # "useState > Usage > Adding state"
        pieces = split_long_section(section.lines, max_chars, overlap)
        for index, piece in enumerate(pieces):
            text = f"{prefix}\n\n{piece}" if prefix else piece
            # The piece index keeps ids unique when react.dev repeats the same example
            # twice in one section (useMemo.md does).
            chunk_id = hashlib.sha256(f"{url}|{prefix}|{index}|{text}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    url=url,
                    anchor=section.anchor,
                    title=page.title,
                    header_path=tuple(section.path),
                    text=text,
                    source_path=source_path,
                    start_line=0,
                )
            )
    return chunks


def _split_prose_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized non-code block at line breaks; code blocks pass through whole."""
    if len(block) <= max_chars or block.lstrip().startswith("```"):
        return [block]
    parts: list[str] = []
    part = ""
    for line in block.splitlines():
        if part and len(part) + 1 + len(line) > max_chars:
            parts.append(part)
            part = line
        else:
            part = f"{part}\n{line}" if part else line
    if part:
        parts.append(part)
    return parts
