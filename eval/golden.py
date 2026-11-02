"""The golden set: row model, loader and URL check.

``golden.jsonl`` holds one JSON object per line (format in eval/README.md).
``load_golden`` parses and validates it; ``missing_urls`` checks the expected
URLs against the url#anchor values that actually exist in the index.
"""

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

REACT_DEV = "https://react.dev/"


class GoldenRow(BaseModel):
    """One golden-set question and the sections that answer it."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_urls: tuple[str, ...]
    category: str = Field(min_length=1)
    expected: Literal["answer", "abstain"]
    notes: str | None = None

    @field_validator("expected_urls")
    @classmethod
    def _check_urls(cls, urls: tuple[str, ...]) -> tuple[str, ...]:
        for url in urls:
            if not url.startswith(REACT_DEV):
                raise ValueError(f"expected_urls must start with {REACT_DEV}, got {url!r}")
        if len(set(urls)) != len(urls):
            raise ValueError("expected_urls contains duplicates")
        return urls

    @model_validator(mode="after")
    def _check_urls_match_expected(self) -> Self:
        if self.expected == "answer" and not self.expected_urls:
            raise ValueError("an 'answer' row needs at least one expected URL")
        if self.expected == "abstain" and self.expected_urls:
            raise ValueError("an 'abstain' row must have no expected URLs")
        return self


def load_golden(path: Path) -> list[GoldenRow]:
    """Read and validate golden.jsonl; errors name the line number as an editor shows it."""
    rows: list[GoldenRow] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip blank lines; enumerate still counts them, so line numbers stay true
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON - {exc}") from exc

            if not isinstance(data, dict):
                raise ValueError(
                    f"line {line_number}: expected a JSON object, got {type(data).__name__}"
                )

            try:
                row = GoldenRow.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"line {line_number}: invalid row - {exc}") from exc

            if row.id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {row.id!r}")

            seen_ids.add(row.id)
            rows.append(row)
    return rows


def missing_urls(rows: Sequence[GoldenRow], known_urls: Collection[str]) -> dict[str, list[str]]:
    """Map row id -> its expected URLs that are not in ``known_urls`` (rows with none left out)."""
    known = set(known_urls)
    result: dict[str, list[str]] = {}

    for row in rows:
        missing = [url for url in row.expected_urls if url not in known]
        if missing:
            result[row.id] = missing

    return result
