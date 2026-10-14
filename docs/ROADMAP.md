# Roadmap

What I plan to work on next, roughly in order. Effort: S = a few hours, M = about a day,
L = several days.

The eval comes first. With 60 questions a single score is uncertain by around ±11 points, and the
table-chunking experiment showed how easy it is to mistake noise for a win.

## 1. Make the eval trustworthy

| What | Why | Effort |
|---|---|---|
| Grow the question set to 200+ and hand-check it (121 so far in `evals/questions_v2.jsonl`) | Needed to detect gains of around 5 points ([Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640)) | S, plus review time |
| Check numeric answers in code: parse `$`, `%`, millions/billions and `(negative)` values and compare with a tolerance | Units and signs are where 10-K answers go wrong, and LLM judges are unreliable on numbers ([TAT-QA](https://arxiv.org/pdf/2105.07624)) | S |
| Hand-label ~60 judge verdicts, report agreement (Cohen's kappa), add a judge from a different model family | Right now the same model family writes, answers and grades the questions ([Preference Leakage](https://arxiv.org/abs/2502.01534)) | M |
| Add the 150 public FinanceBench questions as an outside test set | Human-written questions on real 10-Ks ([FinanceBench](https://arxiv.org/abs/2311.11944)) | M |
| Add unanswerable questions (wrong year, company not loaded, metric not disclosed, false premise) and track over-refusals too | A single accuracy number hides both failure modes ([UAEval4RAG](https://arxiv.org/abs/2412.12300)) | S |
| Run a 20-question retrieval check in CI and fail on a drop | Catches regressions without API cost | S |

## 2. Retrieval and answers

| What | Why | Effort |
|---|---|---|
| Prefix every chunk with `Company (TICKER) · 10-K FY2025 · Item 7` before embedding | Document-level headers raised accuracy from ~55% to ~73% in [Snowflake's 10-K study](https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/) | S |
| Use the LLM to pull out company and years when the rules find nothing; fall back to no filter when unsure | A wrong filter loses the answer completely | S |
| Split multi-year or multi-company questions into one search per company and year | Cross-year questions are much harder for standard retrieval ([Fin-RATE](https://arxiv.org/html/2602.07294v3)) | M |
| Answer exact line items from XBRL (SEC `companyfacts` API) | Exact numbers across years. Tag names vary between companies, so it needs a mapping | L |
| Try newer rerankers and embedders (bge-reranker-v2-m3, mxbai-rerank-v2, gte-modernbert-base, Gemini embeddings) | General benchmarks don't predict finance results well ([FinMTEB](https://arxiv.org/pdf/2502.10990)), so only our eval decides | M |
| Send 12 to 20 chunks instead of 8, and let Gemini run code for ratios and growth rates | More context is cheap with Flash; check accuracy against cost | S |
| Re-test table-aware chunking once the eval is bigger | Small, non-significant gain so far | S |

Not planned: HyDE (it invents numbers that pull search toward wrong figures), parent-document
retrieval (it did worse on FinanceBench), and putting whole 10-Ks in the prompt by default
(expensive, and it mixes up years). The whole-filing approach might work as a fallback when
retrieval comes back with "not answerable".

## 3. Demo and polish

| What | Notes | Effort |
|---|---|---|
| Compare against Gemini File Search (Google's managed RAG) on the same eval | Shows whether building the pipeline myself was worth it | S to M |
| Measure `thinking_level` low vs. medium, use the Batch API (50% cheaper) for eval runs | `gemini-3.8-flash` prices double on 1 Jan 2027 | S |
| Deploy: Cloud Run (scales to zero) + Neon Postgres (free tier) | Use Gemini embeddings or an ONNX reranker in the hosted version to stay within memory | M |
| Before a public link: rate limits, input length cap, daily spend cap, prompt-injection tests | Use a paid Gemini project with a budget cap | S |
| Tracing with Langfuse or Arize Phoenix | One trace per request: search, rerank, answer, with tokens and cost | S |
