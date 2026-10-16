"""Rewrites a follow-up question into a standalone one before retrieval.

What goes here:
    - ``async condense(question, history, llm, prompt, max_turns=3) -> str``:
        * empty history -> return the question unchanged (no LLM call)
        * otherwise render prompts/condense.yaml with the last ``max_turns``
          turns and return the model's single-line rewrite
        * on LLM failure, fall back to the original question and log a warning

Example: history about useState, question "what about in a class component?"
-> "How do I manage state in a React class component instead of useState?"

Tested by: tests/unit/test_condense.py with FakeLLM (no history -> no call;
history -> rewritten; LLM error -> original question).
Phase: 1
"""
