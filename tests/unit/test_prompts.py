"""Tests for codesearch/generation/prompts.py and the prompt files in prompts/.

Prompts are versioned: the version goes into cache keys and traces, so changing
the text without bumping the version would serve stale cached answers and make
eval runs impossible to compare. test_prompt_text_changes_need_a_version_bump
enforces that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codesearch.generation.prompts import (
    PROMPTS_DIR,
    PromptTemplate,
    format_context,
    load_prompt,
    prompt_fingerprint,
)
from codesearch.schemas import Chunk, ScoredChunk

# When you change a prompt's text: bump its `version` in the YAML file, run the
# tests, and copy the new fingerprint from the failure message into this table.
KNOWN_PROMPTS = {
    "answer": ("1", "12f861a0ed85"),
    "condense": ("1", "d6e2353dacbd"),
}


def sc(url: str, anchor: str, text: str, title: str = "useState") -> ScoredChunk:
    chunk = Chunk(
        id=f"{url}#{anchor}",
        url=url,
        anchor=anchor,
        title=title,
        header_path=(title,),
        text=text,
        source_path="x.md",
        start_line=0,
    )
    return ScoredChunk(chunk=chunk, dense_rank=1, dense_score=0.8)


def write(tmp_path: Path, name: str, text: str) -> Path:
    (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")
    return tmp_path


# =====================================================================
# PromptTemplate.render
# =====================================================================


def make(template: str, system: str = "sys") -> PromptTemplate:
    return PromptTemplate(name="t", version="1", system=system, template=template)


def test_render_fills_every_placeholder() -> None:
    t = make("Q: {question}\nC: {context}")
    assert t.render(question="Why?", context="[1] text") == "Q: Why?\nC: [1] text"


def test_render_inserts_values_literally() -> None:
    """Code in the context is full of braces; it must not be treated as placeholders."""
    code = "function App() { return <div>{count}</div>; }"
    assert make("{context}").render(context=code) == code


def test_render_without_placeholders() -> None:
    assert make("fixed text").render() == "fixed text"


def test_missing_value_is_an_error_naming_it() -> None:
    with pytest.raises(ValueError, match="question"):
        make("{context} {question}").render(context="c")


def test_unexpected_value_is_an_error_naming_it() -> None:
    """A typo like contxt= must fail loudly, not silently drop the context."""
    with pytest.raises(ValueError, match="contxt"):
        make("{context}").render(context="c", contxt="oops")


def test_placeholders_lists_the_template_fields() -> None:
    assert make("{context} then {question} then {context}").placeholders == {
        "context",
        "question",
    }


def test_template_is_frozen() -> None:
    t = make("x")
    with pytest.raises((AttributeError, TypeError)):
        t.version = "2"  # type: ignore[misc]


# =====================================================================
# load_prompt
# =====================================================================


def test_default_prompts_dir_is_the_repo_prompts_folder() -> None:
    assert PROMPTS_DIR.name == "prompts"
    assert (PROMPTS_DIR / "answer.yaml").exists()


def test_loads_the_answer_prompt() -> None:
    p = load_prompt("answer")
    assert (p.name, p.version) == ("answer", KNOWN_PROMPTS["answer"][0])
    assert p.placeholders == {"context", "question"}
    assert p.system.strip()


def test_loads_the_condense_prompt() -> None:
    p = load_prompt("condense")
    assert p.placeholders == {"history", "question"}
    assert p.system.strip()


def test_load_works_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_prompt("answer").name == "answer"


def test_load_from_a_custom_dir(tmp_path: Path) -> None:
    d = write(tmp_path, "mine", 'version: "3"\nsystem: be brief\ntemplate: "Q: {question}"\n')
    p = load_prompt("mine", prompts_dir=d)
    assert (p.name, p.version, p.system, p.template) == ("mine", "3", "be brief", "Q: {question}")


def test_numeric_version_becomes_a_string(tmp_path: Path) -> None:
    d = write(tmp_path, "num", "version: 2\ntemplate: x\n")
    assert load_prompt("num", prompts_dir=d).version == "2"


def test_system_is_optional(tmp_path: Path) -> None:
    d = write(tmp_path, "nosys", 'version: "1"\ntemplate: x\n')
    assert load_prompt("nosys", prompts_dir=d).system == ""


@pytest.mark.parametrize(
    "text",
    [
        "template: x\n",  # no version
        'version: ""\ntemplate: x\n',  # empty version
        'version: "1"\n',  # no template
        'version: "1"\ntemplate: ""\n',  # empty template
        'version: "1"\ntemplate: x\ntemplates: y\n',  # unknown key (typo)
        "- just\n- a list\n",  # not a mapping
        "",  # empty file
    ],
    ids=[
        "no-version",
        "empty-version",
        "no-template",
        "empty-template",
        "unknown-key",
        "list",
        "empty",
    ],
)
def test_invalid_prompt_files_are_rejected(tmp_path: Path, text: str) -> None:
    d = write(tmp_path, "bad", text)
    with pytest.raises(ValueError):
        load_prompt("bad", prompts_dir=d)


def test_missing_prompt_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nope", prompts_dir=tmp_path)


# =====================================================================
# Prompt content (the rules the answer prompt must state)
# =====================================================================


def test_answer_prompt_asks_for_numbered_citations() -> None:
    assert "[1]" in load_prompt("answer").system


def test_answer_prompt_allows_saying_it_is_not_in_the_docs() -> None:
    assert "isn't covered in the React docs" in load_prompt("answer").system


def test_answer_prompt_renders_with_real_context() -> None:
    chunks = [sc("https://react.dev/reference/react/useState", "usestate", "useState > ...\n\nx")]
    text = load_prompt("answer").render(context=format_context(chunks), question="How?")
    assert "[1]" in text
    assert "Question: How?" in text


# =====================================================================
# format_context
# =====================================================================

A = "https://react.dev/reference/react/useState"
B = "https://react.dev/learn/state-a-components-memory"


def test_numbers_chunks_from_one_in_order() -> None:
    ctx = format_context([sc(A, "usestate", "first"), sc(B, "", "second")])
    assert ctx.index("[1]") < ctx.index("first") < ctx.index("[2]") < ctx.index("second")
    assert "[3]" not in ctx


def test_each_entry_shows_its_url_with_anchor() -> None:
    ctx = format_context([sc(A, "usestate", "first"), sc(B, "", "second")])
    assert f"{A}#usestate" in ctx
    assert B in ctx and f"{B}#" not in ctx  # no dangling '#' for anchorless chunks


def test_chunk_text_is_included_verbatim() -> None:
    text = "useState > Usage\n\n```js\nconst [a, setA] = useState(0);\n```"
    assert text in format_context([sc(A, "usage", text)])


def test_entries_are_separated_by_a_blank_line() -> None:
    ctx = format_context([sc(A, "a", "first"), sc(B, "b", "second")])
    assert "first\n\n[2]" in ctx


def test_empty_context() -> None:
    assert format_context([]) == ""


def test_number_matches_position_not_rank() -> None:
    """[n] is the position in the list we send, whatever the chunk's retrieval rank was."""
    chunks = [sc(B, "b", "second"), sc(A, "a", "first")]
    ctx = format_context(chunks)
    assert ctx.index("[1]") < ctx.index(f"{B}#b") < ctx.index("[2]") < ctx.index(f"{A}#a")


# =====================================================================
# prompt_fingerprint and the version guard
# =====================================================================


def test_fingerprint_is_stable_and_short() -> None:
    t = make("{question}", system="s")
    assert prompt_fingerprint(t) == prompt_fingerprint(make("{question}", system="s"))
    fp = prompt_fingerprint(t)
    assert len(fp) == 12 and all(ch in "0123456789abcdef" for ch in fp)


def test_fingerprint_changes_with_system_or_template() -> None:
    base = prompt_fingerprint(make("{question}", system="s"))
    assert prompt_fingerprint(make("{question}", system="s2")) != base
    assert prompt_fingerprint(make("{question}!", system="s")) != base


def test_fingerprint_ignores_the_version() -> None:
    """It fingerprints the text; the guard below compares it with the version."""
    a = PromptTemplate(name="t", version="1", system="s", template="x")
    b = PromptTemplate(name="t", version="9", system="s", template="x")
    assert prompt_fingerprint(a) == prompt_fingerprint(b)


def test_fingerprint_separates_system_from_template() -> None:
    assert prompt_fingerprint(make("bc", system="a")) != prompt_fingerprint(make("c", system="ab"))


@pytest.mark.parametrize("name", sorted(KNOWN_PROMPTS))
def test_prompt_text_changes_need_a_version_bump(name: str) -> None:
    known_version, known_fp = KNOWN_PROMPTS[name]
    prompt = load_prompt(name)
    fp = prompt_fingerprint(prompt)
    if fp != known_fp:
        assert prompt.version != known_version, (
            f"prompts/{name}.yaml changed but its version is still {known_version!r}. "
            f"Bump the version, then set KNOWN_PROMPTS[{name!r}] = "
            f"({prompt.version!r}, {fp!r})."
        )
        pytest.fail(
            f"prompts/{name}.yaml is now version {prompt.version!r}: update "
            f"KNOWN_PROMPTS[{name!r}] = ({prompt.version!r}, {fp!r})."
        )
    assert prompt.version == known_version
