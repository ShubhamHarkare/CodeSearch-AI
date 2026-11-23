.PHONY: setup lint test ingest eval ablation serve ui up down index logs

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -m "not needs_llm"

ingest:
	uv run codesearch ingest

eval:
	uv run python -m eval.run_eval eval/configs/dense.yaml

# Every config, then the README table. A failing config is reported and skipped, so one
# bad model download doesn't stop the rest. v1.yaml needs its own index first:
#   CODESEARCH_INDEX__CHUNK_STRATEGY=v1 CODESEARCH_INDEX__CHUNK_OVERLAP=150 \
#   CODESEARCH_INDEX__INDEX_DIR=index/v1 uv run codesearch ingest
ablation:
	@failed=""; for config in eval/configs/*.yaml; do \
		uv run python -m eval.run_eval "$$config" || failed="$$failed $$config"; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED:$$failed (table uses their previous results)"; fi
	uv run python -m eval.ablation_table --write README.md

# Throughput and latency per serving path; needs `make serve` running in another terminal.
# Writes docs/loadtest/{hit,miss,burst,mixed,stream}.json (committed, so README numbers can be re-run).
loadtest:
	uv run python -m eval.loadtest hit
	uv run python -m eval.loadtest miss
	uv run python -m eval.loadtest burst
	uv run python -m eval.loadtest mixed
	uv run python -m eval.loadtest stream

serve:
	uv run uvicorn codesearch.api.main:create_app --factory --reload

ui:
	uv run streamlit run codesearch/ui/streamlit_app.py

# --- Docker: the whole stack (API, UI, Redis, Prometheus, Grafana) -------------
# --env-file: compose reads ${VARS} from the repo's .env, not deploy/.env.
COMPOSE = GIT_SHA=$$(git rev-parse --short HEAD) docker compose --env-file .env -f deploy/docker-compose.yml

up:  ## build and start everything in the background
	$(COMPOSE) up --build -d
	@echo "UI http://localhost:8501   Grafana http://localhost:3000   API http://localhost:8000"

down:  ## stop everything (volumes, so the index and traces, are kept)
	$(COMPOSE) down

index:  ## build or update the search index inside the stack (first run: a few minutes)
	$(COMPOSE) run --rm api codesearch ingest

logs:  ## follow the API's logs
	$(COMPOSE) logs -f api
