.PHONY: setup db ingest questions check-questions eval eval-retrieval guardrails demo test lint

# Install dependencies and create a blank .env if there isn't one.
setup:
	uv sync
	@test -f .env || printf '%s\n' \
		'# Get a key at https://aistudio.google.com/apikey' 'GEMINI_API_KEY=' \
		'# SEC asks for a contact User-Agent, e.g. "Jane Doe jane@example.com"' 'SEC_USER_AGENT=' \
		'DATABASE_URL=postgresql://filings:filings@localhost:5432/filings' \
		'ANSWER_MODEL=gemini-3.8-flash' 'JUDGE_MODEL=gemini-3.8-flash' > .env
	@echo "Fill in GEMINI_API_KEY and SEC_USER_AGENT in .env"

# Postgres + pgvector in Docker.
db:
	docker compose up -d db
	uv run python -m filings.db

# Load the filings pinned in filings.json. ARGS=--force rebuilds chunks;
# ARGS="--latest --years 2" uses the newest 10-Ks instead.
ingest:
	uv run python -m filings.ingest $(ARGS)

# Draft more eval questions with Gemini.
questions:
	uv run python evals/generate_questions.py --n 60

# Sanity checks on the eval questions (no API calls).
check-questions:
	uv run python evals/check_questions.py

# Guardrail cases (uses the Gemini API).
guardrails:
	uv run python evals/run_guardrails.py

# Retrieval metrics only (no API calls).
eval-retrieval:
	uv run python evals/run_eval.py --retrieval-only

# Answers and grading for the full question set (uses the Gemini API).
eval:
	uv run python evals/run_eval.py --modes rerank --no-filters

# Demo at http://localhost:8501.
demo:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
