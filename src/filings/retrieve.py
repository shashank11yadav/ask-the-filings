"""Chunk retrieval: dense (pgvector), sparse (Postgres full-text), hybrid (RRF) and reranked."""

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
    id: int
    ticker: str
    fiscal_year: int
    section: str | None
    text: str
    score: float  # higher is better; scale depends on the mode


_COLS = "id, ticker, fiscal_year, section, text"


def _filters(ticker: str | None, year: int | None) -> tuple[str, list]:
    clauses, params = [], []
    if ticker:
        clauses.append("ticker = %s")
        params.append(ticker.upper())
    if year:
        clauses.append("fiscal_year = %s")
        params.append(year)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def dense(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    where, params = _filters(ticker, year)
    q = embed_query(query)
    rows = conn.execute(
        f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks WHERE TRUE{where}"
        " ORDER BY embedding <=> %s LIMIT %s",
        [q, *params, q, k],
    ).fetchall()
    return [Hit(*r) for r in rows]


def sparse(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    """Full-text search with the question's terms OR-ed together.

    plainto_tsquery ANDs every term, so a full-sentence question matched almost nothing
    (recall 0.23). OR-ing the terms and ranking with ts_rank_cd fixed that.
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
    """Reciprocal Rank Fusion. Uses ranks because dense and sparse scores aren't comparable;
    c=60 is the value from the original paper."""
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (c + rank + 1)
            by_id.setdefault(h.id, h)
    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(**{**by_id[i].__dict__, "score": scores[i]}) for i in ranked]


def _hybrid(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    pool = 4 * k
    return rrf([dense(conn, query, pool, ticker, year), sparse(conn, query, pool, ticker, year)], k)


def rerank(conn, query: str, k: int, ticker=None, year=None) -> list[Hit]:
    candidates = _hybrid(conn, query, max(k, settings.rerank_pool), ticker, year)
    scores = rerank_scores(query, [h.text for h in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [Hit(**{**h.__dict__, "score": s}) for h, s in ranked]


def _known_years(conn) -> set[int]:
    return {r[0] for r in conn.execute("SELECT DISTINCT fiscal_year FROM chunks").fetchall()}


def search(
    query: str,
    k: int = 8,
    mode: Mode = "rerank",
    ticker=None,
    year=None,
    auto_filter: Literal["off", "company", "company+year"] = "company",
) -> list[Hit]:
    """Top `k` chunks for `query`.

    Without an explicit ticker or year, the company (and optionally the year) is read from the
    question. Company-only filtering scored best on the eval; adding the year made it worse.
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
