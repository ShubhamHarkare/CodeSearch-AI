"""Runs one pipeline configuration over the golden set and records the results.

What goes here:
    - ``run(config_path, mode="retrieval" | "answer", limit=None) -> EvalResult``:
        * load Settings and apply the config's overrides
        * retrieval mode: no LLM; for each question record retrieved URLs and
          rerank scores -> Recall@5, MRR@10, nDCG@10, abstain P/R
        * answer mode: full pipeline -> also groundedness, citation precision,
          latency (needs GROQ_API_KEY; marked needs_llm)
        * log params (every Settings field that matters + corpus SHA + prompt
          version), metrics and the per-question rows to MLflow
        * write ``results/<config_name>.json``
    - ``run_ablation(configs_dir) -> markdown table`` for the README.

Must not:
    - change golden.jsonl or baseline.json.

Phase: 1
"""
