# TODO(v2): short names for every common command, so README and CI use the same ones.
#
# Targets to write (each one line, calling uv / the codesearch CLI):
#   setup     uv sync --all-extras && pre-commit install
#   lint      ruff check . && ruff format --check . && mypy codesearch
#   test      pytest -m "not needs_llm"
#   ingest    codesearch ingest
#   eval      codesearch eval --config eval/configs/hybrid_rerank.yaml
#   ablation  run every config in eval/configs/ and write the README table
#   serve     uvicorn codesearch.api.main:create_app --factory --reload
#   ui        streamlit run codesearch/ui/streamlit_app.py
#   up        docker compose -f deploy/docker-compose.yml up --build
#
# Phase 0 (setup, lint, test); the rest as each phase lands.
