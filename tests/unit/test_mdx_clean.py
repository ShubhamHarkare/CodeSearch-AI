"""Tests for codesearch/ingest/mdx_clean.py.

Two layers: small hand-written snippets (one rule each), then checks that run
over every real page in tests/fixtures/mini_corpus.
"""

import re
from pathlib import Path

import pytest

from codesearch.ingest.mdx_clean import CleanPage, clean_mdx
from codesearch.ingest.urls import parse_heading

MINI_CORPUS = Path(__file__).parents[1] / "fixtures" / "mini_corpus"
PAGES = sorted(
    p for p in MINI_CORPUS.rglob("*.md") if p.name not in {"README.md", "LICENSE-DOCS.md"}
)


def body(raw: str) -> str:
    return clean_mdx(raw).body


# =====================================================================
# Frontmatter
# =====================================================================


def test_frontmatter_title_is_extracted_and_removed() -> None:
    page = clean_mdx("---\ntitle: useState\n---\n\nHello\n")
    assert page == CleanPage(title="useState", body="Hello\n")


def test_no_frontmatter_gives_empty_title() -> None:
    assert clean_mdx("Just text\n") == CleanPage(title="", body="Just text\n")


def test_unclosed_frontmatter_is_left_as_text() -> None:
    page = clean_mdx("---\ntitle: broken\nno closing line\n")
    assert page.title == ""
    assert "title: broken" in page.body


def test_empty_input() -> None:
    assert clean_mdx("") == CleanPage(title="", body="")


# =====================================================================
# Code fences
# =====================================================================


def test_sandpack_is_unwrapped_and_code_kept() -> None:
    raw = "<Sandpack>\n\n```js\nconst [count, setCount] = useState(0);\n```\n\n</Sandpack>\n"
    assert body(raw) == "```js\nconst [count, setCount] = useState(0);\n```\n"


def test_hidden_fences_are_dropped() -> None:
    raw = (
        "<Sandpack>\n\n```js src/App.js\nexport default App;\n```\n\n"
        '```json package.json hidden\n{"dependencies": {}}\n```\n\n</Sandpack>\n'
    )
    out = body(raw)
    assert "export default App;" in out
    assert "dependencies" not in out


@pytest.mark.parametrize(
    ("opening", "expected"),
    [
        ("```js src/App.js active", "```js"),
        ("```js {3}", "```js"),
        ('```js [[1, 4, "age"], [2, 4, "setAge"]]', "```js"),
        ("```css", "```css"),
        ("```", "```"),
    ],
)
def test_fence_opening_is_reduced_to_language(opening: str, expected: str) -> None:
    out = body(f"{opening}\nx = 1\n```\n")
    assert out.splitlines()[0] == expected


def test_text_inside_fences_is_never_changed() -> None:
    code = (
        "```js\n"
        "<Note>\n"
        "{/* a JSX comment */}\n"
        "<CodeStep step={1}>x</CodeStep>\n"
        "\n\n\n"
        "</Note>\n"
        "```\n"
    )
    assert body(code) == code


def test_longer_closing_fence_and_tilde_fences() -> None:
    assert body("~~~py\nprint(1)\n~~~\n") == "~~~py\nprint(1)\n~~~\n"
    assert body("```js\na\n````\nafter\n") == "```js\na\n````\nafter\n"


# =====================================================================
# Block components
# =====================================================================


@pytest.mark.parametrize(
    ("tag", "label"),
    [
        ("Pitfall", "Pitfall:"),
        ("Note", "Note:"),
        ("DeepDive", "Deep dive:"),
        ("Deprecated", "Deprecated:"),
        ("Hint", "Hint:"),
        ("Solution", "Solution:"),
        ("Recap", "Recap:"),
        ("YouWillLearn", "You will learn:"),
    ],
)
def test_callouts_become_labels(tag: str, label: str) -> None:
    raw = f"<{tag}>\n\nCareful here.\n\n</{tag}>\n"
    assert body(raw) == f"{label}\n\nCareful here.\n"


def test_containers_disappear_but_keep_content() -> None:
    raw = (
        "<Intro>\n\nIntro text.\n\n</Intro>\n\n<InlineToc />\n\n"
        "<Challenges>\n\nTry it.\n\n</Challenges>\n"
    )
    assert body(raw) == "Intro text.\n\nTry it.\n"


def test_recipes_title_kept_and_solution_separators_dropped() -> None:
    raw = (
        '<Recipes titleText="Basic useState examples" titleId="examples-basic">\n\n'
        "#### Counter {/*counter*/}\n\nOne.\n\n<Solution />\n\n"
        "#### Text field {/*text-field*/}\n\nTwo.\n\n<Solution />\n\n</Recipes>\n"
    )
    assert body(raw) == (
        "Examples: Basic useState examples\n\n"
        "#### Counter {/*counter*/}\n\nOne.\n\n"
        "#### Text field {/*text-field*/}\n\nTwo.\n"
    )


def test_alt_text_whitespace_is_trimmed() -> None:
    assert body('<Diagram name="x" alt="Trailing space. " />\n') == "Diagram: Trailing space.\n"


def test_diagram_and_illustration_keep_alt_text() -> None:
    raw = (
        '<Diagram name="tree" height={367} width={407} alt="A tree of three components." >\n\n'
        "Initially, count is 0\n\n</Diagram>\n\n"
        "<Illustration src=\"/i.png\" alt='A puzzle-like Card tile' />\n"
    )
    assert body(raw) == (
        "Diagram: A tree of three components.\n\n"
        "Initially, count is 0\n\n"
        "Illustration: A puzzle-like Card tile\n"
    )


def test_console_block_keeps_message() -> None:
    raw = '<ConsoleBlock level="error">\n\nWarning: Each child needs a key.\n\n</ConsoleBlock>\n'
    assert body(raw) == "Warning: Each child needs a key.\n"


# =====================================================================
# Prose lines
# =====================================================================


def test_heading_anchor_is_kept() -> None:
    line = "### `useState(initialState)` {/*usestate*/}"
    assert body(line + "\n") == line + "\n"


def test_mdx_comments_outside_headings_are_removed() -> None:
    raw = "Before\n\n{/* TODO(@poteto) - investigate */}\n\nAfter {/* inline */} text\n"
    assert body(raw) == "Before\n\nAfter  text\n"


def test_inline_codestep_is_unwrapped() -> None:
    raw = (
        "Render <CodeStep step={1}>some JSX</CodeStep> "
        "and a <CodeStep step={2}>DOM node</CodeStep>.\n"
    )
    assert body(raw) == "Render some JSX and a DOM node.\n"


def test_components_inside_inline_code_are_untouched() -> None:
    line = "* Nested JSX like `<Card><Avatar /></Card>` will appear as `children`.\n"
    assert body(line) == line


def test_blank_lines_are_collapsed() -> None:
    assert body("a\n\n\n\n\nb\n") == "a\n\nb\n"


def test_trailing_whitespace_is_trimmed_on_prose() -> None:
    assert body("text   \n") == "text\n"


# =====================================================================
# Every real page in the mini corpus
# =====================================================================

TAG_LINE = re.compile(r"^\s*</?[A-Z][A-Za-z0-9]*\b[^>]*>\s*$")


def _outside_fences(text: str) -> list[str]:
    lines, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return lines


def _visible_fences(text: str) -> int:
    """Count fence blocks whose opening line is not marked hidden."""
    count, in_fence = 0, False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            if not in_fence and "hidden" not in stripped.split():
                count += 1
            in_fence = not in_fence
    return count


def test_corpus_has_pages() -> None:
    assert len(PAGES) == 22


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(MINI_CORPUS).as_posix())
def test_real_page(page: Path) -> None:
    raw = page.read_text(encoding="utf-8")
    cleaned = clean_mdx(raw)

    # Title came from the frontmatter.
    assert cleaned.title

    prose = _outside_fences(cleaned.body)
    # No component tag lines survive outside code.
    leftover = [line for line in prose if TAG_LINE.match(line)]
    assert not leftover, leftover[:3]
    # No stray MDX comments outside headings.
    assert not [line for line in prose if "{/*" in line and not line.startswith("#")]
    # Every heading still parses with an anchor.
    for line in prose:
        if line.startswith("#"):
            heading = parse_heading(line)
            assert heading is not None and heading.anchor, line
    # Every non-hidden code block survived.
    assert _visible_fences(cleaned.body) == _visible_fences(raw)
    # Cleaning is idempotent.
    assert clean_mdx(cleaned.body).body == cleaned.body


def test_use_state_page_reads_well() -> None:
    cleaned = clean_mdx((MINI_CORPUS / "reference/react/useState.md").read_text(encoding="utf-8"))
    assert cleaned.title == "useState"
    assert "Pitfall:" in cleaned.body
    assert "Examples: Basic useState examples" in cleaned.body
    assert "const [age, setAge] = useState(28);" in cleaned.body
    assert "<Sandpack>" not in cleaned.body
