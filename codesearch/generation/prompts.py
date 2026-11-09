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

import string
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from codesearch.ingest.urls import url_with_anchor
from codesearch.schemas import ScoredChunk

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"  # repo's prompts/, from any cwd


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    template: str

    @property
    def placeholders(self) -> set[str]:
        parsed_template = string.Formatter().parse(self.template)
        unique_field_set = set()

        for _text, field_name, _spec, _conversion in parsed_template:
            if field_name:
                unique_field_set.add(field_name)

        return unique_field_set

    def render(self, **values: Any) -> str:
        formatter = string.Formatter()
        template_placeholders = {
            field_name
            for _, field_name, _, _ in formatter.parse(self.template)
            if field_name is not None
        }

        provided_keys = set(values.keys())

        # 1. Missing values for placeholders in template
        missing_placeholders = template_placeholders - provided_keys
        if missing_placeholders:
            missing_str = ", ".join(sorted(missing_placeholders))
            raise ValueError(f"Missing required template placeholder(s): {missing_str}")

        # 2. Extra values provided that are not placeholders in template
        extra_keys = provided_keys - template_placeholders
        if extra_keys:
            extra_str = ", ".join(sorted(extra_keys))
            raise ValueError(f"Unexpected argument(s) not in template: {extra_str}")

        # 3. Format and return
        return self.template.format(**values)


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    file_path = prompts_dir / f"{name}.yaml"
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file '{file_path.name}' is empty or not a mapping.")

    # Allowed keys validation
    allowed_keys = {"version", "system", "template"}
    if not set(data.keys()).issubset(allowed_keys):
        invalid_keys = set(data.keys()) - allowed_keys
        raise ValueError(f"Prompt file contained invalid key(s): {', '.join(invalid_keys)}")

    # Required key presence and truthiness checks
    if data.get("version") in (None, ""):
        raise ValueError("Missing or empty 'version' key.")

    if not data.get("template"):
        raise ValueError("Missing or empty 'template' key.")

    return PromptTemplate(
        name=name,
        version=str(data["version"]),
        system=data.get("system") or "",
        template=data["template"],
    )


def format_context(chunks: list[ScoredChunk]) -> str:
    entries: list[str] = []

    for number, scored in enumerate(chunks, start=1):
        url = url_with_anchor(scored.chunk.url, scored.chunk.anchor)

        entries.append(f"[{number}] {url}\n{scored.chunk.text}")

    return "\n\n".join(entries)


def prompt_fingerprint(template: PromptTemplate) -> str:
    return sha256(f"{template.system}\0{template.template}".encode()).hexdigest()[:12]
