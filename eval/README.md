<!-- TODO(v2): explain how evaluation works. Write this once eval runs end to end. -->
# Evaluation

`eval/` measures answer quality; `tests/` checks code correctness. Keep them separate.

## golden.jsonl format (one JSON object per line, no comments allowed in JSONL)

```json
{"id": "q001", "question": "How do I update state based on the previous state?",
 "expected_urls": ["https://react.dev/reference/react/useState#updating-state-based-on-the-previous-state"],
 "category": "hooks", "expected": "answer"}
{"id": "oos001", "question": "How do I define a computed property in Vue?",
 "expected_urls": [], "category": "out_of_scope", "expected": "abstain"}
```

Target: ~150 in-scope rows (seeded from the 20 questions in v1's test_queries.py)
plus ~25 out-of-scope rows. Every row is checked by hand before it is added.

## Files

- `golden.jsonl`: ground truth (starts empty)
- `build_golden.py`: drafts candidate rows for you to review
- `configs/*.yaml`: one file per pipeline variant (the ablation)
- `metrics.py`: scoring functions
- `run_eval.py`: runs one config, logs to MLflow
- `compare.py`: CI gate against `baseline.json` (generated, then committed)
