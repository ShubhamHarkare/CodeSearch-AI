"""Command-line entry point (Typer). Thin: parses arguments, calls library code.

Commands so far:
    - ``codesearch ingest [--full] [--embedder NAME]``: fetch react.dev at the
      pinned commit, chunk it, embed what's new, write the index + manifest, and
      print what changed.
    - ``codesearch index-info``: print the manifest of the built index.

Still to come: ``eval``, ``ask``, ``serve``.

Must not:
    - contain logic beyond argument parsing and printing.

Tested by: tests/integration/test_cli.py (Typer's CliRunner).
Phase: 1
"""

import asyncio
from typing import Annotated

import typer

from codesearch.config import Settings, get_settings
from codesearch.errors import CodeSearchError, IndexNotReadyError, LLMUnavailableError
from codesearch.factory import build_embedder, build_pipeline
from codesearch.ingest.build import build_index
from codesearch.ingest.manifest import read_manifest
from codesearch.log import configure_logging

app = typer.Typer(help="CodeSearch AI: explainable RAG over the React docs.", no_args_is_help=True)


def _settings(embedder: str | None) -> Settings:
    settings = get_settings()
    if embedder:
        settings = settings.with_overrides(index={"embedding_model": embedder})
    return settings


@app.command()
def ingest(
    full: Annotated[bool, typer.Option(help="Re-embed every chunk, not just new ones.")] = False,
    embedder: Annotated[
        str | None,
        typer.Option(help='Override the embedding model, e.g. "hash" for a quick offline run.'),
    ] = None,
    json_logs: Annotated[bool, typer.Option(help="Log as JSON instead of readable text.")] = False,
) -> None:
    """Build or update the search index from react.dev."""
    configure_logging(json=json_logs)
    settings = _settings(embedder)
    try:
        report = build_index(settings, build_embedder(settings), full=full)
    except CodeSearchError as error:
        typer.secho(f"Ingest failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.secho("Index built", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  index_version  {report.index_version}")
    typer.echo(f"  react.dev      {report.corpus_sha}")
    typer.echo(f"  pages          {report.pages}")
    typer.echo(f"  chunks         {report.chunks}")
    typer.echo(
        f"  changes        +{report.added} added, -{report.removed} removed, "
        f"{report.unchanged} unchanged" + ("  (full rebuild)" if report.full_rebuild else "")
    )
    typer.echo(f"  took           {report.duration_s}s")
    typer.echo(f"  written to     {settings.index.index_dir}")


@app.command("index-info")
def index_info() -> None:
    """Show what the current index was built from."""
    settings = get_settings()
    manifest = read_manifest(settings.index.index_dir)
    if manifest is None:
        typer.echo(f"No index in {settings.index.index_dir}. Run: codesearch ingest")
        raise typer.Exit(code=1)
    typer.echo(manifest.model_dump_json(indent=2))


@app.command()
def ask(question: str) -> None:
    try:
        settings = get_settings()
        pipeline = build_pipeline(settings)
        answer = asyncio.run(pipeline.answer(question))
    except IndexNotReadyError as e:
        typer.echo(
            f"Index is not ready. Run `codesearch ingest` to build it first.\nDetails: {e}",
            err=True,
        )
        raise typer.Exit(code=1) from e
    except LLMUnavailableError as e:
        typer.echo(f"LLM Service Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    # Output the generated text
    typer.echo(answer.text)

    # Output citations if available
    if answer.citations:
        typer.echo("\nSources:")
        for citation in answer.citations:
            typer.echo(f"[{citation.number}] {citation.url}")


if __name__ == "__main__":
    app()
