"""Checks whether each answer sentence is supported by the chunk it cites.

What goes here:
    - ``split_sentences(answer) -> list[str]``: skips code blocks; keeps
      citation markers attached to their sentence.
    - ``NLIScorer(model_name="cross-encoder/nli-deberta-v3-small")``:
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
