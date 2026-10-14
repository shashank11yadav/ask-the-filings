# Makefile - short commands for common tasks. Run e.g. `make ingest` from this folder.
# (`uv run` runs a command inside this project's virtual environment.)

# Companies and number of years to download. Override like: make ingest TICKERS="AAPL MSFT" YEARS=1
TICKERS ?= AAPL MSFT NVDA AMZN GOOGL JPM XOM UNH WMT TSLA
YEARS ?= 2

.PHONY: setup db ingest questions check-questions eval eval-retrieval guardrails demo test lint

# Install Python dependencies and create a blank .env to fill in (won't overwrite an existing one).
setup:
	uv sync
	@test -f .env || printf '%s\n' \
		'# Get a key at https://aistudio.google.com/apikey' 'GEMINI_API_KEY=' \
		'# SEC asks for a contact User-Agent, e.g. "Jane Doe jane@example.com"' 'SEC_USER_AGENT=' \
		'DATABASE_URL=postgresql://filings:filings@localhost:5432/filings' \
		'ANSWER_MODEL=gemini-3.8-flash' 'JUDGE_MODEL=gemini-3.8-flash' > .env
	@echo "Fill in GEMINI_API_KEY and SEC_USER_AGENT in .env"

# Start Postgres+pgvector in Docker and create the table.
db:
	docker compose up -d db
	uv run python -m filings.db

# Download, chunk, embed and store the 10-K filings. Already-loaded filings are skipped.
# Extra options: make ingest ARGS=--force   (rebuild chunks, e.g. after changing chunk size)
#                make ingest ARGS=--refresh (check the SEC for newer filings)
ingest:
	uv run python -m filings.ingest $(TICKERS) --years $(YEARS) $(ARGS)

# Draft more eval questions with Gemini (then run make check-questions).
questions:
	uv run python evals/generate_questions.py --n 60

# Automatic checks on the eval questions (no API calls).
check-questions:
	uv run python evals/check_questions.py

# Guardrail tests: advice, predictions, off-topic and prompt-injection cases. Uses Gemini credits.
guardrails:
	uv run python evals/run_guardrails.py

# Free eval: search quality only, no Gemini calls.
eval-retrieval:
	uv run python evals/run_eval.py --retrieval-only

# Full eval: answers + judge. Uses Gemini credits.
eval:
	uv run python evals/run_eval.py --modes rerank --no-filters

# Start the demo web page at http://localhost:8501 (needs `make db` running and data ingested).
demo:
	uv run streamlit run app/streamlit_app.py

# Run the unit tests.
test:
	uv run pytest -q

# Check code style.
lint:
	uv run ruff check .
