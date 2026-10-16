"""End-to-end pipeline test with fakes over the mini corpus.

Cases: an in-scope question returns citations pointing at react.dev URLs;
an off-topic question abstains without calling the LLM (check FakeLLM's
recorded prompts); the same question twice hits the exact cache; stream()
yields events in order retrieval -> token... -> citations -> groundedness -> done.
"""
