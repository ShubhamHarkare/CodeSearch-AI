"""Tests for codesearch/ingest/mdx_clean.py.

Cases: frontmatter title extracted and removed; anchor comments removed;
<Sandpack> unwrapped with code fences intact; <Pitfall> becomes "Pitfall: ...";
text inside a code fence is never modified; output is stable (idempotent).
"""
