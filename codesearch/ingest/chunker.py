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
import re
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
    heading: str = ""  # the heading line that opened it (v1 chunks keep it in the text)


def split_into_sections(page: CleanPage, *, keep_empty: bool = False) -> list[Section]:
    """Cut a page into sections at #/##/### headings outside code blocks.

    A heading with no text under it ("## Reference" right before a ###) is dropped
    unless ``keep_empty`` is set; the v1 baseline needs it.
    """
    title = [page.title] if page.title else []
    path: list[str] = list(title)
    open_headings: list[tuple[int, str]] = []  # (level, text) of the headings we're under
    anchor: str = ""
    heading_line: str = ""
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
            if (keep_empty and heading_line) or any(ln.strip() for ln in current):
                sections.append(
                    Section(path=path, anchor=anchor, lines=current, heading=heading_line)
                )

            ## Starting a new section here
            # Close every heading at this level or deeper, then open this one.
            # (Tracking levels, not list positions: react.dev pages start at ##.)
            open_headings = [h for h in open_headings if h[0] < heading.level]
            open_headings.append((heading.level, heading.text))
            path = title + [text for _, text in open_headings]
            anchor = heading.anchor
            heading_line = line
            current = []
        else:
            current.append(line)

    if (keep_empty and heading_line) or any(ln.strip() for ln in current):
        sections.append(Section(path=path, anchor=anchor, lines=current, heading=heading_line))

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


# ---------------------------------------------------------------------------
# v1 baseline: the chunking the original v1 app used (vectorstore.py on the v1
# branch): LangChain's MarkdownHeaderTextSplitter on #/##/### with
# strip_headers=False, then RecursiveCharacterTextSplitter(chunk_size=1000,
# chunk_overlap=150). Ported here, quirks included, so the eval can compare v2
# against what v1 really did without depending on LangChain.
# ---------------------------------------------------------------------------

V1_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")


def v1_section_text(section: Section) -> str:
    """A section as MarkdownHeaderTextSplitter emitted it.

    Every line is stripped (code indentation included) and loses non-printable
    characters such as tabs, the heading line is kept,
    and paragraphs are joined with two spaces and a newline instead of a blank
    line. Blank lines inside code blocks survive as empty lines.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    in_code = False
    for raw in ([section.heading] if section.heading else []) + section.lines:
        line = "".join(ch for ch in raw.strip() if ch.isprintable())  # drops tabs too
        if line.startswith("```"):
            in_code = not in_code
            current.append(line)
        elif in_code or line:
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return "  \n".join(paragraphs)


def _split_keep_start(text: str, separator: str) -> list[str]:
    """Split on a literal separator, keeping it at the start of the next piece."""
    if not separator:
        return list(text)
    parts = re.split(f"({re.escape(separator)})", text)
    pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
    return [p for p in pieces if p]


def _merge_pieces(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces up to ``size``; start the next chunk with ~``overlap`` chars."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for piece in pieces:
        if total + len(piece) > size and current:
            joined = "".join(current).strip()
            if joined:
                chunks.append(joined)
            # Drop pieces from the front until what's left fits as the overlap.
            while total > overlap or (total + len(piece) > size and total > 0):
                total -= len(current.pop(0))
        current.append(piece)
        total += len(piece)
    joined = "".join(current).strip()
    if joined:
        chunks.append(joined)
    return chunks


def recursive_split(
    text: str, size: int, overlap: int, separators: tuple[str, ...] = V1_SEPARATORS
) -> list[str]:
    """RecursiveCharacterTextSplitter: try paragraph, line, word, then character cuts."""
    separator: str = separators[-1]
    finer: tuple[str, ...] = ()
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator, finer = candidate, separators[i + 1 :]
            break
    out: list[str] = []
    fitting: list[str] = []
    for piece in _split_keep_start(text, separator):
        if len(piece) < size:
            fitting.append(piece)
            continue
        if fitting:
            out.extend(_merge_pieces(fitting, size, overlap))
            fitting = []
        out.extend(recursive_split(piece, size, overlap, finer) if finer else [piece])
    if fitting:
        out.extend(_merge_pieces(fitting, size, overlap))
    return out


def chunk_page_v1(
    page: CleanPage, url: str, source_path: str, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """v1's chunks: heading split, then a recursive 1,000-char split with 150 overlap.

    Unlike v2 there is no header-path prefix in the text and code blocks can be cut
    anywhere. The anchor and header path are kept as metadata (v1 kept the headers
    in metadata too), so the eval can score v1 at section level.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if not 0 <= overlap < max_chars:
        raise ValueError(f"overlap must be in [0, max_chars), got {overlap}")

    sections = split_into_sections(page, keep_empty=True)
    chunks: list[Chunk] = []
    pending = ""  # a heading-only section waiting to be glued onto the next one
    for i, section in enumerate(sections):
        text = v1_section_text(section)
        if pending:
            text = f"{pending}  \n{text}"
            pending = ""
        following = sections[i + 1] if i + 1 < len(sections) else None
        heading_only = not any(ln.strip() for ln in section.lines)
        if heading_only and following and len(following.path) > len(section.path):
            # LangChain appends a lone heading to the deeper section after it.
            pending = text
            continue
        for index, piece in enumerate(recursive_split(text, max_chars, overlap)):
            # "v1|" keeps these ids apart from v2 ids for the same text
            chunk_id = hashlib.sha256(
                f"v1|{url}|{section.anchor}|{index}|{piece}".encode()
            ).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    url=url,
                    anchor=section.anchor,
                    title=page.title,
                    header_path=tuple(section.path),
                    text=piece,
                    source_path=source_path,
                    start_line=0,
                )
            )
    return chunks
