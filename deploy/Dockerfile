# TODO(v2): multi-stage image for the API (the UI can reuse it with another CMD).
#
# Stage 1 "builder" (python:3.11-slim):
#   - install uv; copy pyproject.toml + uv.lock; `uv sync --frozen --no-dev`
#   - pre-download the embedder, reranker and NLI models into /models
#     (so containers start fast and work offline)
# Stage 2 "runtime" (python:3.11-slim):
#   - copy the venv and /models from builder; copy codesearch/ and prompts/ only
#   - create and switch to a non-root user
#   - ENV PORT=8000, HF_HOME=/models; EXPOSE 8000
#   - HEALTHCHECK on http://localhost:8000/healthz
#   - CMD uvicorn codesearch.api.main:create_app --factory --host 0.0.0.0 --port $PORT
# The index is NOT baked in: mount it, or download the release asset matching
# the manifest's index_version at startup.
# Phase 4.
