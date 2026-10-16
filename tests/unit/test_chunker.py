"""Tests for codesearch/ingest/chunker.py.

Cases: header path prepended; no chunk splits a code fence; chunks respect
max_chars except a single oversized fence; ids are identical across runs; an
edited sentence changes only its own chunk's id.
"""
