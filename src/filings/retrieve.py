"""
retrieve.py - finds the chunks most relevant to a question ("the R in RAG").

Four search modes, so the eval can compare them:
  - "dense"  : meaning search. Embed the question, find chunks with the closest vectors.
               Good at synonyms ("revenue" ~ "net sales"); weaker on exact terms and numbers.
  - "sparse" : keyword search using Postgres full-text search (similar to BM25).
               Good at exact words like "iPhone" or "Item 1A"; misses synonyms.
  - "hybrid" : run both, then merge the two ranked lists with Reciprocal Rank Fusion (RRF).
               Usually beats either one alone.
  - "rerank" : hybrid search for ~30 candidates, then a cross-encoder re-sorts them (rerank.py).
               Slower, but usually the most accurate.
"""

from dataclasses import dataclass
from typing import Literal

from filings.config import settings
from filings.db import connect
from filings.embed import embed_query
from filings.query_parse import parse_query
from filings.rerank import rerank_scores

Mode = Literal["dense", "sparse", "hybrid", "rerank"]


@dataclass
class Hit:
    """One search result."""

    id: int
    ticker: str
    fiscal_year: int
    section: str | None
    text: str
    score: float  # higher = more relevant (the scale depends on the search mode)


_COLS = "id, ticker, fiscal_year, section, text"


def _filters(ticker: str | None, year: int | None) -> tuple[str, list]:
    """Build optional WHERE conditions, e.g. only search Apple's 2025 filing."""
    clauses, params = [], []
    if ticker:
        clauses.append("ticker = %s")
        params.append(ticker.upper())
    if year:
        clauses.append("fiscal_year = %s")
        params.append(year)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def dense(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    """Meaning search. `<=>` is pgvector's cosine-distance operator (0 = identical)."""
    where, params = _filters(ticker, year)
    q = embed_query(query)
    rows = conn.execute(
        f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks WHERE TRUE{where}"
        " ORDER BY embedding <=> %s LIMIT %s",
        [q, *params, q, k],  # values fill the %s placeholders in order (safe from SQL injection)
    ).fetchall()
    return [Hit(*r) for r in rows]


def sparse(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    """Keyword search.

    plainto_tsquery turns the question into stemmed keywords joined with AND ("&"):
    'rang' & 'matur' & 'swap' & ... That would only match chunks containing EVERY word,
    which almost never happens for a full-sentence question. So we swap each "&" for
    "|" (OR): a chunk matches if it has ANY keyword. ts_rank_cd then scores chunks
    higher the more keywords they contain and the closer together they are.
    `@@` means "matches".
    """
    where, params = _filters(ticker, year)
    rows = conn.execute(
        f"SELECT {_COLS}, ts_rank_cd(tsv, q) AS score"
        " FROM chunks, to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|')) q"
        f" WHERE tsv @@ q{where}"
        " ORDER BY score DESC LIMIT %s",
        [query, *params, k],
    ).fetchall()
    return [Hit(*r) for r in rows]


def rrf(result_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: merge several ranked lists into one.

    Each chunk gets 1 / (c + rank) points from every list it appears in. A chunk ranked
    highly by BOTH searches ends up on top. We use ranks, not raw scores, because dense
    and sparse scores are on different scales and can't be compared directly.
    c=60 is the standard value from the original RRF paper.
    """
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (c + rank + 1)
            by_id.setdefault(h.id, h)
    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(**{**by_id[i].__dict__, "score": scores[i]}) for i in ranked]


def _hybrid(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    """Fetch a bigger pool (4k) from each search, then fuse down to k with RRF."""
    pool = 4 * k
    return rrf([dense(conn, query, pool, ticker, year), sparse(conn, query, pool, ticker, year)], k)


def rerank(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    """Take the top `rerank_pool` hybrid results and re-sort them with the cross-encoder."""
    candidates = _hybrid(conn, query, max(k, settings.rerank_pool), ticker, year)
    scores = rerank_scores(query, [h.text for h in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [Hit(**{**h.__dict__, "score": s}) for h, s in ranked]


def _known_years(conn) -> set[int]:
    """Fiscal years we have filings for (used to decide whether a year in the question is usable)."""
    return {r[0] for r in conn.execute("SELECT DISTINCT fiscal_year FROM chunks").fetchall()}


def search(
    query: str,
    k: int = 8,
    mode: Mode = "rerank",
    ticker=None,
    year=None,
    auto_filter: Literal["off", "company", "company+year"] = "company",
) -> list[Hit]:
    """Main entry point: return the top `k` chunks for `query` using the chosen mode.

    If the caller doesn't pass a ticker/year, `auto_filter` tries to read them from the question
    itself (query_parse.py). "company" was the best setting on our eval; see README.
    """
    with connect() as conn:
        if auto_filter != "off" and ticker is None and year is None:
            parsed = parse_query(query, _known_years(conn))
            ticker = parsed.ticker
            if auto_filter == "company+year":
                year = parsed.year
        if mode == "dense":
            return dense(conn, query, k, ticker, year)
        if mode == "sparse":
            return sparse(conn, query, k, ticker, year)
        if mode == "rerank":
            return rerank(conn, query, k, ticker, year)
        return _hybrid(conn, query, k, ticker, year)
