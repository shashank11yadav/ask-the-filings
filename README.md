# Ask the Filings

Ask a question about a company's annual report (SEC 10-K) and get an answer along with the exact
passages it came from. If the filings don't contain the answer, it says so instead of guessing.

I built this to learn how to make a retrieval system that is measurably good, not just one that
demos well. Every design choice below was kept or dropped based on the eval.

## Results

121 questions over 10 companies' 10-Ks (two fiscal years each). The question doesn't say which
company or year to search; the system works that out itself.

| | |
|---|---|
| **Correct answers** | **115 / 121 (95%)**, 95% CI roughly 90-98% |
| Confident wrong answers | **0**. The other 6 were "not found in the excerpts" replies |
| Answers supported by their cited passages | 115 / 115 |
| Table questions / text questions correct | 91 / 96 and 24 / 25 |
| Cost per question | $0.008 (answer + grading, `gemini-3.8-flash`) |
| Latency | 5.6 s median, 9.5 s p95 |

Correct answers (95%) come out higher than retrieval recall (88%) because many facts appear in
both years' filings, so the answer can come from a passage other than the one the question was
written from.

How much each part of the search contributes (did the right passage make the top 8?):

| Search | Company/year given | Realistic: company detected from the question |
|---|---|---|
| Embeddings only | 0.69 | 0.60 |
| Keywords only (Postgres full-text) | 0.82 | 0.67 |
| Hybrid (both, merged with RRF) | 0.81 | 0.74 |
| **Hybrid + reranker (default)** | **0.89** | **0.88** |
| Hybrid + reranker, no company detection | | 0.77 |

On this question set, keyword search beats the embedding model (bge-small doesn't handle
numbers and line items well), and the cross-encoder reranker does most of the work.

**Caveats.** The questions were drafted by Gemini from passages of the filings, and Gemini also
grades the answers, which tends to be generous. I haven't hand-checked every question yet, and the
set has no unanswerable questions, so wrong refusals aren't measured.

## How it works

```mermaid
flowchart TD
    subgraph Ingest["Ingest (once)"]
        A[SEC EDGAR 10-K HTML] --> B["Clean text<br/>tables flattened to 'label | value' rows"]
        B --> C["Split into 400-word chunks<br/>tagged with 10-K Item"]
        C --> D["Embed with bge-small<br/>(local)"]
        C --> E[Postgres full-text index]
        D --> F[(Postgres + pgvector)]
        E --> F
    end

    subgraph Ask["Answer a question"]
        Q[Question] --> P["Detect company name<br/>e.g. Walmart's -> WMT"]
        P --> S1[Embedding search<br/>top 120]
        P --> S2[Keyword search<br/>top 120]
        F -.-> S1
        F -.-> S2
        S1 --> R[Reciprocal Rank Fusion]
        S2 --> R
        R --> RR["Rerank top 30<br/>bge-reranker-base (local)"]
        RR --> G["Gemini 3.8 Flash<br/>top 8 passages, JSON schema"]
        G --> O["answerable + answer + citations"]
    end
```

Embedding and reranking run locally, so the only paid call is the final Gemini request.

## Run it

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make setup     # installs dependencies and creates a blank .env
               # then fill in GEMINI_API_KEY and SEC_USER_AGENT in .env
make db        # Postgres + pgvector in Docker
make ingest    # downloads 10 companies x 2 years of 10-Ks (re-runs skip what's already loaded)
make demo      # web page at http://localhost:8501
```

The demo page shows the answer, the cited passages, and a link to the original filing on EDGAR.

## Eval

```bash
make eval-retrieval   # retrieval metrics only, no API calls
make eval             # answers + grading (about $1 for 121 questions)
make questions        # draft more questions with Gemini, then review them by hand
uv run python evals/compare.py <run A>.jsonl <run B>.jsonl
```

`compare.py` checks whether one configuration really beats another, using a paired McNemar test
and a bootstrap confidence interval over the per-question results.

## What I tried

- **Keyword search bug.** My first version required every word of the question to appear in a
  chunk, so it found nothing for most questions (recall 0.23 on an earlier 60-question set).
  Matching any word fixed it (0.77).
- **Rerankers.** `ms-marco-MiniLM-L-6-v2` gave no gain (it's trained on short web snippets).
  `bge-reranker-base` was the biggest single improvement. Reranking 50 candidates instead of 30
  was slower and no better.
- **Company detection.** Without it, similar wording from other companies crowds out the answer.
  Plain rules (company names and tickers) took realistic recall from 0.77 to 0.88. Also detecting
  the year made things slightly worse.
- **Table-aware chunking.** I noticed about half of the 400-word chunks are longer than the 512
  tokens the embedder and reranker can read. I built a chunker that sizes chunks in tokens and
  keeps tables whole, repeating headers when a table has to be split:

  | Chunker | Company/year given | Realistic |
  |---|---|---|
  | 400 words (kept) | 0.89 | 0.88 |
  | 400 tokens | 0.92 | 0.88 |
  | 400 tokens, table-aware | 0.93 | 0.90 |

  Slightly ahead everywhere, but not significant (realistic: fixed 9, broke 7, p = 0.80), so I
  didn't switch. On my first, smaller question set it had looked like an 8-point win. That's why
  the eval now has more questions and `compare.py` exists. It's still available with
  `CHUNKER=tokens_tables`.
- **Fairer questions.** The first questions were generated from chunks, which favours whatever
  chunker produced them, and often didn't name a company. They're now generated from raw filing
  text, must name the company, and are paraphrased so they don't just reuse the filing's words.

## Limitations

- When a question names a year but not a filing, search can pick up the wrong year's table
  (e.g. Walmart's FY2026 advertising costs answered from the FY2025 filing).
- Section labels are approximate; some financial statements get tagged with the wrong Item.
- Questions that compare years or companies aren't handled specially yet.

## Next

- Hand-check the question set and add unanswerable questions.
- Check numeric answers in code instead of with the LLM grader.
- Add a short header (company, year, section) to each chunk before embedding.
- Split multi-year and multi-company questions into one search per filing.
- Deploy the demo (Cloud Run + Neon) with rate limits and a spending cap.

## Project layout

| Path | What it does |
|---|---|
| `src/filings/ingest.py` | Downloads 10-Ks from EDGAR, chunks, embeds and stores them |
| `src/filings/chunking.py` | HTML to text, table flattening, the chunkers |
| `src/filings/embed.py`, `rerank.py` | Local embedding model and cross-encoder reranker |
| `src/filings/retrieve.py` | Embedding, keyword, hybrid and reranked search |
| `src/filings/query_parse.py` | Finds the company named in a question |
| `src/filings/answer.py`, `llm.py` | Builds the prompt and calls Gemini with a JSON schema |
| `app/streamlit_app.py` | Demo page |
| `evals/` | Question set, question generator, eval runner, paired comparison |
| `tests/` | Unit tests (offline, no API key needed) |
