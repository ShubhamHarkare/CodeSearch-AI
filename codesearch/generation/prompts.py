"""Loads versioned prompt templates and formats retrieved context.

What goes here:
    - ``PromptTemplate`` (dataclass): name, version, system, template;
      ``render(**values) -> str``.
    - ``load_prompt(name, prompts_dir="prompts") -> PromptTemplate``: reads
      prompts/<name>.yaml; fails if ``version`` is missing.
    - ``format_context(chunks: list[ScoredChunk]) -> str``: numbers chunks
      [1]..[n], each with title, header path and URL, then the text.
    - ``prompt_fingerprint(template) -> str``: hash of the text, used by a test
      that fails when the text changes but ``version`` does not.

Tested by: tests/unit/test_prompts.py (render, numbering, version guard).
Phase: 1
"""
