"""Checks whether each answer sentence is supported by the chunk it cites.

What goes here:
    - ``split_sentences(answer) -> list[str]``: skips code blocks; keeps
      citation markers attached to their sentence.
    - ``NLIScorer(model_name="cross-encoder/nli-deberta-v3-base")``:
      ``entailment(premise=chunk_text, hypothesis=sentence) -> float``.
    - ``assess(answer, citations, chunks, scorer) -> tuple[list[SentenceSupport], float]``:
        * label each sentence supported (>= 0.7) / partial (>= 0.4) / unsupported
          (thresholds in config; tune them on the golden set)
        * a sentence with no citation is "unsupported" unless it is purely
          connective (e.g. "Here's how:")
        * overall groundedness = share of supported sentences

Why: explainability that is measured, not only displayed; feeds the
confidence score and the eval report.

Tested by: tests/unit/test_groundedness.py with a fake scorer (labels,
uncited sentences, code blocks skipped).
Phase: 2
"""

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from codesearch.generation.citations import extract_citations
from codesearch.interfaces import EntailmentScorer
from codesearch.schemas import Citation, ScoredChunk, SentenceSupport

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# End of a sentence: . ! or ? (not right after a digit, so "18.2" survives), plus an
# optional citation such as " [3]" or " [1, 2]", then the whitespace we split on.
_SENTENCE_END = re.compile(r"((?<!\d)[.!?](?:\s*\[\d+(?:\s*,\s*\d+)*\])?)\s+")


def split_sentences(text: str) -> list[str]:
    # 1. Remove fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Replace inline `code` with placeholders and store originals
    inline_codes: list[str] = []

    def _replace_code(match: re.Match[str]) -> str:
        idx = len(inline_codes)
        inline_codes.append(match.group(0))
        return f"\x00CODE_{idx}\x00"

    text = re.sub(r"`[^`\n]+`", _replace_code, text)

    results: list[str] = []

    # 3. Split into lines, strip bullet markers, and skip empty lines
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip leading '- ', '* ', or '1. ' (any number)
        line = re.sub(r"^(?:[-*]|\d+\.)(?:\s+|$)", "", line).strip()
        if not line:
            continue

        marked = re.sub(_SENTENCE_END, lambda m: m.group(1) + "\x01", line)
        sentences = [s.strip() for s in marked.split("\x01") if s.strip()]

        # 5. Restore inline code placeholders
        for sentence in sentences:
            restored = re.sub(
                r"\x00CODE_(\d+)\x00",
                lambda m: inline_codes[int(m.group(1))],
                sentence,
            )
            results.append(restored)

    return results


def is_connective(sentence: str) -> bool:
    """A lead-in like "Key points:" is not a claim, so it isn't graded."""
    return sentence.rstrip().endswith(":") and not extract_citations(sentence)


def assess(
    text: str,
    citations: Sequence[Citation],
    chunks: Sequence[ScoredChunk],
    scorer: EntailmentScorer,
    *,
    supported_at: float = 0.7,
    partial_at: float = 0.4,
    all_sources: bool = False,
) -> tuple[list[SentenceSupport], float | None]:
    number_to_id = {c.number: c.chunk_id for c in citations}
    id_to_text = {c.chunk.id: c.chunk.text for c in chunks}

    # Pass 1: plan sentences and collect pairs without calling the model yet
    plan: list[tuple[str, list[str], str]] = []
    for sentence in split_sentences(text):
        ids = list(
            dict.fromkeys(
                number_to_id[num]
                for num in extract_citations(sentence)
                if num in number_to_id and number_to_id[num] in id_to_text
            )
        )
        hypothesis = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", sentence).strip()
        # A lone "[3]" (empty once the markers are gone) or a lead-in like
        # "Key points:" is not a claim: don't grade it.
        if not hypothesis or (not ids and is_connective(sentence)):
            continue
        plan.append((sentence, ids, hypothesis))

    if not plan:
        return [], None

    # Single batched entailment call. Each (chunk, hypothesis) pair is sent once;
    # `position` remembers where its score will be. With all_sources, every sentence is
    # also paired with every chunk (the mis-citation diagnostic).
    pairs: list[tuple[str, str]] = []
    position: dict[tuple[str, str], int] = {}
    for _, ids, hypothesis in plan:
        for chunk_id in list(id_to_text) if all_sources else ids:
            if (chunk_id, hypothesis) not in position:
                position[(chunk_id, hypothesis)] = len(pairs)
                pairs.append((id_to_text[chunk_id], hypothesis))
    scores = scorer.entailment(pairs) if pairs else []
    if len(scores) != len(pairs):
        raise ValueError(f"Expected {len(pairs)} entailment scores, got {len(scores)}")

    # Pass 2: map scores back to sentences and assign labels
    results: list[SentenceSupport] = []
    for sentence, ids, hypothesis in plan:
        # The label judges only the CITED sources, with or without the diagnostic.
        score = max((scores[position[(cid, hypothesis)]] for cid in ids), default=0.0)
        label: Literal["supported", "partial", "unsupported"]
        if score >= supported_at:
            label = "supported"
        elif score >= partial_at:
            label = "partial"
        else:
            label = "unsupported"

        best_score: float | None = None
        best_chunk_id: str | None = None
        if all_sources and id_to_text:
            best_chunk_id = max(id_to_text, key=lambda cid: scores[position[(cid, hypothesis)]])
            best_score = scores[position[(best_chunk_id, hypothesis)]]

        results.append(
            SentenceSupport(
                sentence=sentence,
                cited_chunk_ids=ids,
                label=label,
                score=score,
                best_score=best_score,
                best_chunk_id=best_chunk_id,
            )
        )

    groundedness = sum(1 for r in results if r.label == "supported") / len(results)
    return results, groundedness


class NLIScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        entailment_index: int = 1,
        device: str | None = None,
        batch_size: int = 16,
        predict: Callable[[list[tuple[str, str]]], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.entailment_index = entailment_index
        self.device = device
        self.batch_size = batch_size
        self.predict = predict
        self.model: Any = None  # the CrossEncoder, loaded on first use

    def logits(self, pairs: list[tuple[str, str]]) -> Any:
        if self.predict:
            return self.predict(pairs)

        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)

        return self.model.predict(pairs, batch_size=self.batch_size)

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        z = np.asarray(self.logits(list(pairs)), dtype=np.float64)
        z = z - z.max(axis=1, keepdims=True)

        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

        return [float(x) for x in p[:, self.entailment_index]]
