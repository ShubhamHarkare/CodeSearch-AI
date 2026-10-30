"""Keyword search with per-term explanations.

What goes here:
    - ``tokenize(text) -> list[str]``: lowercase words, AND keep API names whole
      plus their camelCase parts (``useLayoutEffect`` -> ``uselayouteffect``,
      ``use``, ``layout``, ``effect``); drop markdown punctuation.
    - ``BM25Index.build(chunks) -> BM25Index`` (rank_bm25.BM25Okapi).
    - ``search(query, k) -> list[tuple[str, float]]`` (chunk_id, score).
    - ``term_contributions(query, chunk_id) -> dict[str, float]``: how much each
      query term added to that chunk's score (for the "why this source" panel).
    - ``save(path)`` / ``load(path)`` so the API starts without rebuilding.

Why: exact API names are where dense embeddings are weakest; this is half of
hybrid retrieval and one row of the ablation table.

Tested by: tests/unit/test_bm25.py (exact API name ranks first, camelCase
split, contributions sum to the score).
Phase: 1 (search), 2 (term_contributions)
"""

import json
import os
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from codesearch.schemas import Chunk


def tokenize(text: str) -> list[str]:
    # Words = runs of letters/digits in any language ([^\W_] is \w without "_",
    # so snake_case still splits). [a-zA-Z0-9] would turn "naïve" into "na", "ve".
    words = re.findall(r"[^\W_]+", text)

    tokens: list[str] = []
    for word in words:
        tokens.append(word.lower())

        parts = re.findall(
            r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word
        )  # camelCase / PascalCase parts: useLayoutEffect -> use, layout, effect

        if len(parts) > 1 and word.isascii():  # camelCase is an ASCII-code convention
            tokens.extend(part.lower() for part in parts)

    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    def __init__(self, ids: list[str], tokens: list[list[str]]) -> None:
        if len(ids) != len(tokens):
            raise ValueError(f"got {len(ids)} ids but {len(tokens)} token lists")
        self.ids = ids
        self.tokens = tokens
        self.token_set = [set(t) for t in tokens]
        self.bm25 = BM25Okapi(tokens) if tokens else None
        self.positions = {cid: i for i, cid in enumerate(ids)}

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        ids = [chunk.id for chunk in chunks]
        text = [tokenize(chunk.text) for chunk in chunks]
        return cls(ids, text)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q or self.bm25 is None or k <= 0:
            return []
        scores = self.bm25.get_scores(q)
        wanted = set(q)

        matches = [i for i, words in enumerate(self.token_set) if words & wanted]
        matches.sort(key=lambda i: scores[i], reverse=True)
        return [(self.ids[i], float(scores[i])) for i in matches[:k]]

    def term_contributions(self, query: str, chunk_id: str) -> dict[str, float]:
        i = self.positions[chunk_id]
        if self.bm25 is None:
            return {}
        contributions: dict[str, float] = {}
        chunk_words = self.token_set[i]  # the set of words in THIS chunk only
        for unique_word in dict.fromkeys(tokenize(query)):
            if unique_word in chunk_words:
                contributions[unique_word] = float(self.bm25.get_scores([unique_word])[i])

        return contributions

    def __len__(self) -> int:
        """Number of chunks in the index (so len(index) works)."""
        return len(self.ids)

    def save(self, path: Path) -> None:
        """Write ids and tokens as JSON (no pickle: loading a pickle can run code)."""
        # 1. make path.parent if missing
        path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Prepare data payload
        data = {"ids": self.ids, "tokens": self.tokens}

        # 3. Write to .tmp, then atomically replace
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Read a saved index and rebuild the scorer."""
        # 1. Read and parse JSON data
        data = json.loads(path.read_text(encoding="utf-8"))

        # 2. Instantiate and return the class
        return cls(data["ids"], data["tokens"])
