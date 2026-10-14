# Ask the Filings

Ask questions about a company's annual report (SEC 10-K) and get an answer with the exact passages
it came from. If the filings don't contain the answer, it says so instead of guessing.

I built this to learn how to make a retrieval system that is measurably good, not just one that
demos well. Every change below was kept or dropped based on an eval.

## Results

60 questions over 10 companies' 10-Ks (2 years each), answered by `gemini-3.8-flash`:

| Retrieval | Found the right passage (recall@8) | Answer correct | Cost / question | Median latency |
|---|---|---|---|---|
| Dense (embeddings only) | 0.78 | 0.82 | $0.007 | 4.1 s |
| Keyword (Postgres full-text) | 0.77 | 0.82 | $0.007 | 3.9 s |
| Hybrid (both, merged with RRF) | 0.87 | 0.88 | $0.007 | 4.0 s |
| **Hybrid + reranker (default)** | **0.92** | **0.95** | $0.007 | 4.8 s |

The default setup answered 57 of 60 correctly. The other 3 were honest "not in the excerpts"
replies where retrieval missed a table. It never gave a confident wrong answer on this set.

Things to keep in mind when reading these numbers:
- The questions were drafted by Gemini from the filings and Gemini also grades the answers.
  I haven't hand-checked all of them yet, and a same-family judge tends to be generous.
- 60 questions is small. A score like 0.95 has an uncertainty of several points either way.
- None of the 60 questions is unanswerable, so how often it wrongly refuses isn't measured yet.

## How it works

```
EDGAR 10-K HTML -> clean text, tables flattened to "label | value" rows
                -> 400-word chunks tagged with their 10-K Item (1A Risk Factors, 7 MD&A, ...)
                -> bge-small embeddings + Postgres full-text index (one Postgres with pgvector)

question -> detect company name -> dense top-k + keyword top-k -> Reciprocal Rank Fusion
         -> bge-reranker-base re-sorts the top 30 -> top 8 to Gemini
         -> JSON answer: {answerable, answer, citations}
```

Embeddings and reranking run locally, so the only paid call is the final Gemini request.

## Demo

```bash
make setup    # installs dependencies and creates a blank .env
              # then add GEMINI_API_KEY and SEC_USER_AGENT to .env
make db       # Postgres + pgvector in Docker
make ingest   # downloads 10 companies x 2 years of 10-Ks (skips anything already loaded)
make demo     # web page at http://localhost:8501
```

`make serve` starts the same thing as a JSON API (`POST /ask`, docs at http://127.0.0.1:8000/docs).

## Running the eval

```bash
make questions        # draft new eval questions with Gemini, then review them by hand
make eval-retrieval   # retrieval metrics only, no API calls
make eval             # full run: answers + grading
uv run python evals/compare.py <run A>.jsonl <run B>.jsonl   # is B really better, or luck?
```

`compare.py` does a paired McNemar test and a bootstrap confidence interval on per-question
results. I added it after a change that looked like +8 points turned out to be noise.

## What I tried

Retrieval only (does the right passage reach the top 8?), on the 60-question set:

| Change | recall@8 | MRR | Notes |
|---|---|---|---|
| Keyword search requiring every word | 0.23 | | Bug: found nothing for 42 of 60 questions |
| Keyword search matching any word | 0.77 | | |
| Hybrid (dense + keyword) | 0.88 | 0.62 | |
| + reranker `ms-marco-MiniLM-L-6-v2` | 0.88 | | No gain. It's trained on short web snippets |
| + reranker `bge-reranker-base` (top 30) | **0.92** | **0.81** | About 1.2 s extra on CPU. Top 50 wasn't better |
| Same, but company/year not given to search | 0.77 | 0.66 | Similar wording from other companies crowds out the answer |
| + detect the company from the question | 0.82 | 0.71 | Simple rules, no extra API call |

Chunking, re-tested on 121 newer questions sampled from the filing text instead of from chunks:

| Chunker | recall (company/year given) | recall (realistic) |
|---|---|---|
| 400 words (kept) | 0.89 | 0.88 |
| 400 tokens | 0.92 | 0.88 |
| 400 tokens, tables kept whole, headers repeated | 0.93 | 0.90 |

I didn't adopt the table-aware chunker. It's slightly ahead everywhere, but none of the
differences are significant (realistic: fixed 9, broke 7, p = 0.80). On the first 60 questions it
looked like an 8-point win. It's still available with `CHUNKER=tokens_tables`.

One finding that surprised me: about half of the 400-word chunks are longer than the 512 tokens
the embedding model and reranker can read, so the end of those chunks is never seen by them.
Fixing that alone didn't change the scores.

## Limitations

- Numbers buried in large tables are the main source of misses.
- When a question names a year but not a filing, search can pull the wrong year's table
  (e.g. Walmart's FY2026 advertising costs get answered from the FY2025 filing).
- Section labels are approximate. A few financial-statement chunks are tagged with the wrong Item.
- The eval questions are only partly reviewed by hand, and there are no unanswerable ones yet.

## Next

See [docs/ROADMAP.md](docs/ROADMAP.md). Short version: grow and hand-check the eval set, check
numeric answers with code instead of an LLM judge, add chunk headers (company / year / section),
then deploy the demo.

## Project layout

| Path | What it does |
|---|---|
| `src/filings/ingest.py` | Downloads 10-Ks from EDGAR, chunks, embeds and stores them |
| `src/filings/chunking.py` | HTML to text, table flattening, the three chunkers |
| `src/filings/embed.py`, `rerank.py` | Local embedding model and cross-encoder reranker |
| `src/filings/retrieve.py` | Dense, keyword, hybrid and reranked search |
| `src/filings/query_parse.py` | Finds the company named in a question |
| `src/filings/answer.py`, `llm.py` | Builds the prompt and calls Gemini with a JSON schema |
| `src/filings/api.py`, `app/streamlit_app.py` | JSON API and demo page |
| `evals/` | Question generation, eval runner, paired comparison, question sets |
| `tests/` | Unit tests (offline, no API key needed) |
