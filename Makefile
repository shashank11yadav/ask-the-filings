# Makefile - short commands for common tasks. Run e.g. `make ingest` from this folder.
# (`uv run` runs a command inside this project's virtual environment.)

# Companies and number of years to download. Override like: make ingest TICKERS="AAPL MSFT" YEARS=1
TICKERS ?= AAPL MSFT NVDA AMZN GOOGL JPM XOM UNH WMT TSLA
YEARS ?= 2

.PHONY: setup db ingest questions eval eval-retrieval serve demo test lint

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

# Draft eval questions with Gemini (check them by hand afterwards!).
questions:
	uv run python evals/generate_questions.py --n 60

# Free eval: search quality only, no Gemini calls.
eval-retrieval:
	uv run python evals/run_eval.py --retrieval-only

# Full eval: answers + judge. Uses Gemini credits.
eval:
	uv run python evals/run_eval.py

# Start the web API at http://127.0.0.1:8000 (auto-reloads when you edit code).
serve:
	uv run uvicorn filings.api:app --reload

# Start the demo web page at http://localhost:8501 (needs `make db` running and data ingested).
demo:
	uv run streamlit run app/streamlit_app.py

# Run the unit tests.
test:
	uv run pytest -q

# Check code style.
lint:
	uv run ruff check .
