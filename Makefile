.PHONY: setup lint test ingest eval ablation serve ui up

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
#   CODESEARCH_INDEX__CHUNK_STRATEGY=v1 CODESEARCH_INDEX__INDEX_DIR=index/v1 uv run codesearch ingest
ablation:
	@failed=""; for config in eval/configs/*.yaml; do \
		uv run python -m eval.run_eval "$$config" || failed="$$failed $$config"; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED:$$failed (table uses their previous results)"; fi
	uv run python -m eval.ablation_table --write README.md

serve:
	uv run uvicorn codesearch.api.main:create_app --factory --reload

ui:
	uv run streamlit run codesearch/ui/streamlit_app.py

up:
	docker compose -f deploy/docker-compose.yml up --build
