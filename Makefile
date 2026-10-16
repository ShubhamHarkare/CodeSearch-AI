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
	uv run codesearch eval --config eval/configs/hybrid_rerank.yaml

ablation:
	@for config in eval/configs/*.yaml; do \
		uv run codesearch eval --config "$$config"; \
	done

serve:
	uv run uvicorn codesearch.api.main:create_app --factory --reload

ui:
	uv run streamlit run codesearch/ui/streamlit_app.py

up:
	docker compose -f deploy/docker-compose.yml up --build
