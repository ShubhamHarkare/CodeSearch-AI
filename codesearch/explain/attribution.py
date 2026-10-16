"""On-demand leave-one-out attribution: how much did each source drive the answer?

What goes here:
    - ``async leave_one_out(question, chunks, answer, llm, embedder, prompt)
      -> dict[str, float]``:
        * for each chunk i, regenerate the answer WITHOUT chunk i
        * score_i = 1 - cosine(embed(original_answer), embed(answer_without_i))
        * normalise so scores sum to 1; return {chunk_id: score}
    - Run regenerations concurrently (asyncio.gather) with a cap.

Cost: k extra LLM calls, so it only runs from
POST /v1/explain/{trace_id}/deep ("Deep explain" button), never by default.
Connects to your Attribution-Guided Masking paper: same question, asked at
the context level instead of inside the model.

Tested by: tests/unit/test_attribution.py with a FakeLLM whose answer depends
on which chunks are present.
Phase: 2
"""
