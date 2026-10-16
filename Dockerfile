# =============================================================================
# TODO(v2): REPLACED by deploy/Dockerfile (multi-stage, non-root, models
# downloaded at build time, /healthz health check). Delete this one after.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

#TODO: Installing dependencies
# TODO(v2): build tools end up in the runtime image. Multi-stage: build in one stage,
#     copy only the installed packages into a slim runtime stage.
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TODO(v2): copies everything not in .dockerignore. v2 copies only codesearch/ and prompts/.
COPY . .

RUN mkdir -p logs

#! Exposing the streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# TODO(v2): $PORT has no default, the container runs as root, and the app needs an
#     Ollama server that isn't in the image. v2 runs uvicorn for the API; the UI
#     is its own service in docker-compose.
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
