"""Command-line entry point (Typer). Thin: parses arguments, calls library code.

Commands to write:
    - ``codesearch ingest [--sha SHA] [--full]``: fetch, chunk, embed, write
      index + manifest; print an IngestReport (added / changed / removed).
    - ``codesearch eval --config eval/configs/X.yaml [--mode retrieval|answer]``
    - ``codesearch ask "question"``: one answer with citations and timings.
    - ``codesearch serve``: run uvicorn on codesearch.api.main:create_app.
    - ``codesearch index-info``: print the manifest.

Must not:
    - contain logic beyond argument parsing and printing.

Tested by: tests/integration/test_cli.py using Typer's CliRunner.
Phase: 1
"""
