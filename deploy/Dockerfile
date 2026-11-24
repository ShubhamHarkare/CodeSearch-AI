# CodeSearch AI: one image for the API and the UI (compose picks the command).
#
#   docker build -f deploy/Dockerfile -t codesearch .        (from the repo root)
#
# Two stages: "builder" has the build tools and downloads the models; "runtime" gets
# only the finished virtualenv, the models and the code, so it's smaller and has no
# compilers or package managers in it.

# ---------------------------------------------------------------- builder
FROM python:3.13-slim AS builder

RUN pip install --no-cache-dir "uv>=0.8,<1"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models

WORKDIR /app

# 1. Dependencies first, from the lock file only. This layer is rebuilt only when
#    pyproject.toml or uv.lock change, not on every code edit.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api --extra ui --extra explain

# 2. Then the project itself.
COPY codesearch/ codesearch/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --extra ui --extra explain

# 3. Bake the models into the image: containers start without downloading anything
#    and keep working if Hugging Face is down. Must match codesearch/config.py.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-base
RUN .venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${NLI_MODEL}')"

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

# git: re-indexing clones react.dev. Nothing else is installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user, not root: a bug in the app can't take over the container.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /models /models
COPY --chown=app:app codesearch/ codesearch/
COPY --chown=app:app prompts/ prompts/
COPY --chown=app:app models/calibrator.json models/calibrator.json
COPY --chown=app:app eval/golden.jsonl eval/golden.jsonl

# index/, data/ and traces/ are volumes (see docker-compose.yml): created here so
# the volumes start out owned by the app user.
RUN mkdir -p index data traces && chown app:app index data traces

ARG GIT_SHA=unknown
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    CODESEARCH_IMAGE_SHA=${GIT_SHA}

USER app
EXPOSE 8000

# Healthy = the API answers. start-period: the models load before uvicorn listens.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"

# One worker on purpose: the caches, single-flight and the metrics window live in
# this process. More capacity = more containers behind a load balancer (later).
CMD ["uvicorn", "codesearch.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
