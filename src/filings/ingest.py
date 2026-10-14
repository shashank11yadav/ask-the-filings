"""Download 10-K filings from EDGAR, chunk and embed them, and load them into Postgres.

By default the filings listed in filings.json are loaded, so every checkout works with the same
documents the eval questions were written against. --latest looks up the newest 10-Ks instead.

    uv run python -m filings.ingest                      # pinned filings (default)
    uv run python -m filings.ingest AAPL --force         # rebuild chunks, e.g. after changing the chunker
    uv run python -m filings.ingest AAPL MSFT --latest --years 2 [--save-manifest]
"""

import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import requests

from filings.chunking import Chunk, chunk_by_tokens, chunk_text, html_to_text
from filings.config import settings
from filings.db import connect, init_db
from filings.embed import count_tokens, embed_passages

MANIFEST = Path("filings.json")
RAW_DIR = Path("data/raw")

# XOM now maps to a new holding company that hasn't filed a 10-K yet; the reports are still
# filed under the original Exxon Mobil Corp.
CIK_OVERRIDES = {"XOM": 34088}


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    if not settings.sec_user_agent:
        raise SystemExit("Set SEC_USER_AGENT in .env (SEC requires 'Name email@example.com').")
    s = requests.Session()
    s.headers["User-Agent"] = settings.sec_user_agent
    return s


def _get(url: str, timeout: int = 30) -> requests.Response:
    time.sleep(0.2)  # SEC fair-access limit is 10 requests per second
    resp = _session().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


def ticker_to_cik(ticker: str) -> int:
    if ticker in CIK_OVERRIDES:
        return CIK_OVERRIDES[ticker]
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return int(row["cik_str"])
    raise ValueError(f"unknown ticker {ticker}")


def latest_10ks(ticker: str, years: int) -> list[dict]:
    """The company's newest `years` 10-Ks, newest first."""
    cik = ticker_to_cik(ticker)
    data = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    # "recent" holds only the latest ~1000 filings. Frequent filers like JPM need the older pages.
    pages = [data["filings"]["recent"]]
    older = [f["name"] for f in data["filings"].get("files", [])]
    out = []
    while pages:
        page = pages.pop(0)
        for form, acc, doc, report_date in zip(
            page["form"], page["accessionNumber"], page["primaryDocument"], page["reportDate"]
        ):
            if form == "10-K" and report_date:
                out.append({
                    "ticker": ticker,
                    "fiscal_year": int(report_date[:4]),
                    "cik": cik,
                    "accession": acc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
                })
            if len(out) == years:
                return out
        if older:
            pages.append(_get(f"https://data.sec.gov/submissions/{older.pop(0)}").json())
    return out


def fetch(filing: dict) -> str:
    """The filing's HTML, downloaded once into data/raw/."""
    dest = RAW_DIR / f"{filing['ticker']}_{filing['fiscal_year']}.html"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_get(filing["url"], timeout=60).text)
    return dest.read_text()


def filing_text(filing: dict) -> str:
    """Extracted text of a filing, checked against the manifest.

    The hash covers the extracted text rather than the raw HTML, because EDGAR injects a
    script tag with a random path into every response.
    """
    text = html_to_text(fetch(filing))
    expected = filing.get("text_sha256")
    if expected and text_hash(text) != expected:
        raise SystemExit(
            f"{filing['ticker']} FY{filing['fiscal_year']}: extracted text doesn't match {MANIFEST}. "
            "Either the filing was replaced or html_to_text changed; if the change is intended, "
            "regenerate the manifest and re-check the eval questions."
        )
    return text


def make_chunks(text: str) -> list[Chunk]:
    if settings.chunker == "words":
        return chunk_text(text, settings.chunk_tokens, settings.chunk_overlap)
    if settings.chunker in ("tokens", "tokens_tables"):
        return chunk_by_tokens(
            text,
            settings.max_chunk_tokens,
            settings.chunk_overlap_tokens,
            count_tokens,
            tables=settings.chunker == "tokens_tables",
        )
    raise SystemExit(f"Unknown CHUNKER {settings.chunker!r}: use words, tokens or tokens_tables")


def already_loaded(conn, ticker: str, fiscal_year: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE ticker = %s AND fiscal_year = %s LIMIT 1", (ticker, fiscal_year)
    ).fetchone()
    return row is not None


def load(filings: list[dict], force: bool = False) -> None:
    init_db()
    with connect() as conn:
        for f in filings:
            ticker, fy = f["ticker"], f["fiscal_year"]
            if not force and already_loaded(conn, ticker, fy):
                print(f"{ticker} FY{fy}: already loaded")
                continue
            chunks = make_chunks(filing_text(f))
            vectors = embed_passages([c.text for c in chunks])
            conn.execute("DELETE FROM chunks WHERE ticker = %s AND fiscal_year = %s", (ticker, fy))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (ticker, fiscal_year, section, chunk_index, text, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    [(ticker, fy, c.section, c.chunk_index, c.text, v) for c, v in zip(chunks, vectors)],
                )
            print(f"{ticker} FY{fy}: {len(chunks)} chunks")


def save_manifest(filings: list[dict]) -> None:
    for f in filings:
        f["text_sha256"] = text_hash(html_to_text(fetch(f)))
    filings = sorted(filings, key=lambda f: (f["ticker"], f["fiscal_year"]))
    MANIFEST.write_text(json.dumps(filings, indent=2) + "\n")
    print(f"wrote {len(filings)} filings to {MANIFEST}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", help="default: every company in filings.json")
    p.add_argument("--force", action="store_true", help="re-chunk and re-embed loaded filings")
    p.add_argument("--latest", action="store_true", help="use the newest 10-Ks instead of filings.json")
    p.add_argument("--years", type=int, default=2, help="with --latest: 10-Ks per company")
    p.add_argument("--save-manifest", action="store_true", help="with --latest: overwrite filings.json")
    args = p.parse_args()
    tickers = [t.upper() for t in args.tickers]

    if args.latest:
        if not tickers:
            tickers = sorted({f["ticker"] for f in load_manifest()})
        filings = [f for t in tickers for f in latest_10ks(t, args.years)]
        if args.save_manifest:
            save_manifest(filings)
            print("The eval questions were written against the old filings: re-check them.")
    else:
        filings = [f for f in load_manifest() if not tickers or f["ticker"] in tickers]
        if not filings:
            raise SystemExit(f"No filings in {MANIFEST} for {tickers}")
    load(filings, force=args.force)
