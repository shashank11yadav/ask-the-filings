"""
ingest.py - downloads 10-K filings from the SEC and loads them into the database.

For each company ticker (e.g. AAPL) it:
  1. looks up the company's SEC id (called a CIK)
  2. finds its most recent 10-K filings
  3. downloads each filing's HTML
  4. chunks the text            (chunking.py)
  5. embeds every chunk         (embed.py)
  6. saves chunks + vectors into Postgres

Re-running is cheap: anything already done is skipped.
  - SEC lookups (ticker list, filing lists) are saved in data/raw/ and reused.
  - Downloaded filings are saved in data/raw/ and reused.
  - Filings already in the database are skipped (no re-chunking or re-embedding).

Usage:
  uv run python -m filings.ingest AAPL MSFT NVDA --years 2
  uv run python -m filings.ingest AAPL --refresh   # ask the SEC again (e.g. a new 10-K came out)
  uv run python -m filings.ingest AAPL --force     # rebuild the chunks (e.g. after changing chunk size)
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import requests

from filings.chunking import Chunk, chunk_by_tokens, chunk_text, html_to_text
from filings.config import settings
from filings.db import connect, init_db
from filings.embed import count_tokens, embed_passages

RAW_DIR = Path("data/raw")  # downloaded filings are saved here
INDEX_DIR = RAW_DIR / "index"  # saved SEC lookups (ticker list, each company's 10-K list)

# Tickers whose 10-Ks live under a different CIK than the SEC's ticker list says.
# XOM: the ticker now maps to a new holding company ("ExxonMobil Holdings Corp") that hasn't
# filed a 10-K yet; the annual reports are still under the original "Exxon Mobil Corp".
CIK_OVERRIDES = {"XOM": 34088}


@lru_cache(maxsize=1)  # create the session only once, and only if we actually need the network
def _session() -> requests.Session:
    """An HTTP session with the User-Agent header the SEC requires."""
    if not settings.sec_user_agent:
        raise SystemExit("Set SEC_USER_AGENT in .env (SEC requires 'Name email@example.com').")
    s = requests.Session()
    s.headers["User-Agent"] = settings.sec_user_agent
    return s


def _get_json(url: str) -> dict:
    """Download a JSON file from the SEC."""
    time.sleep(0.2)  # be polite: the SEC allows at most 10 requests per second
    resp = _session().get(url, timeout=30)
    resp.raise_for_status()  # stop with a clear error on 403/404 instead of parsing an error page
    return resp.json()


def ticker_to_cik(ticker: str, refresh: bool = False) -> int:
    """Convert a stock ticker like 'AAPL' into the SEC's company id (CIK).

    The SEC's full ticker list is saved to data/raw/index/company_tickers.json the first time,
    and read from there afterwards.
    """
    if ticker in CIK_OVERRIDES:
        return CIK_OVERRIDES[ticker]
    path = INDEX_DIR / "company_tickers.json"
    if path.exists() and not refresh:
        data = json.loads(path.read_text())
    else:
        data = _get_json("https://www.sec.gov/files/company_tickers.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return int(row["cik_str"])
    raise ValueError(f"unknown ticker {ticker}")


def _find_10ks_online(cik: int, years: int) -> list[dict]:
    """Ask the SEC for the company's latest `years` 10-K filings (newest first)."""
    data = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")

    # The SEC lists only a company's ~1000 most recent filings under "recent". Older ones are
    # split into extra pages listed in "files". Big banks like JPM file thousands of small
    # documents a year, so their older 10-Ks are only in those extra pages.
    pages = [data["filings"]["recent"]]
    extra_pages = [f["name"] for f in data["filings"].get("files", [])]

    out = []
    while pages:
        page = pages.pop(0)
        # Each page stores parallel lists (form types, accession numbers, ...); zip walks them together.
        for form, acc, doc, report_date in zip(
            page["form"], page["accessionNumber"], page["primaryDocument"], page["reportDate"]
        ):
            if form == "10-K" and report_date:
                out.append(
                    {
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
                        "fiscal_year": int(report_date[:4]),  # "2025-09-27" -> 2025
                    }
                )
            if len(out) == years:
                return out
        # Not enough 10-Ks yet: load the next older page, if there is one.
        if extra_pages:
            pages.append(_get_json(f"https://data.sec.gov/submissions/{extra_pages.pop(0)}"))
    return out


def list_10ks(ticker: str, years: int, refresh: bool = False) -> list[dict]:
    """Return the company's latest `years` 10-Ks, using the saved list when it has enough.

    The list is saved to data/raw/index/<TICKER>_10k.json. Use refresh=True to check the SEC
    for newer filings (a company files one 10-K per year, so this is rarely needed).
    """
    path = INDEX_DIR / f"{ticker}_10k.json"
    if path.exists() and not refresh:
        saved = json.loads(path.read_text())
        if len(saved) >= years:
            return saved[:years]
    filings = _find_10ks_online(ticker_to_cik(ticker, refresh), years)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(filings, indent=2))
    return filings


def fetch(url: str, dest: Path) -> str:
    """Download a filing, or read it from data/raw/ if we already downloaded it."""
    if dest.exists():
        return dest.read_text()
    time.sleep(0.2)
    resp = _session().get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text)
    return resp.text


def make_chunks(html: str) -> list[Chunk]:
    """Clean the filing and cut it into chunks using the chunker chosen in config.py."""
    text = html_to_text(html)
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
    """True if this filing's chunks are already in the database."""
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE ticker = %s AND fiscal_year = %s LIMIT 1", (ticker, fiscal_year)
    ).fetchone()
    return row is not None


def ingest(tickers: list[str], years: int, refresh: bool = False, force: bool = False) -> None:
    init_db()  # make sure the table exists
    with connect() as conn:
        for ticker in tickers:
            filings = list_10ks(ticker, years, refresh)
            # Warn loudly instead of silently skipping. This can happen when a ticker points to a
            # new legal entity (e.g. after a restructuring) that hasn't filed a 10-K yet.
            if len(filings) < years:
                print(f"WARNING: {ticker}: found only {len(filings)} of {years} 10-Ks")

            for filing in filings:
                fy = filing["fiscal_year"]

                # Skip the slow part (chunking + embedding) if this filing is already loaded.
                # Use --force after changing chunking settings, so old chunks get replaced.
                if not force and already_loaded(conn, ticker, fy):
                    print(f"{ticker} FY{fy}: already in database, skipping")
                    continue

                html = fetch(filing["url"], RAW_DIR / f"{ticker}_{fy}.html")
                chunks = make_chunks(html)
                vectors = embed_passages([c.text for c in chunks])

                # Delete any old rows for this filing first, so we never end up with duplicates.
                conn.execute("DELETE FROM chunks WHERE ticker = %s AND fiscal_year = %s", (ticker, fy))
                # executemany inserts all rows in one go (much faster than one INSERT per chunk).
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (ticker, fiscal_year, section, chunk_index, text, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (ticker, fy, c.section, c.chunk_index, c.text, v)
                            for c, v in zip(chunks, vectors)
                        ],
                    )
                print(f"{ticker} FY{fy}: {len(chunks)} chunks")


# This block runs only when you execute the file directly (not when another file imports it).
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="+")  # one or more tickers
    p.add_argument("--years", type=int, default=2)  # how many years of filings per company
    p.add_argument("--refresh", action="store_true", help="re-check the SEC for newer filings")
    p.add_argument("--force", action="store_true", help="re-chunk and re-embed even if already loaded")
    args = p.parse_args()
    ingest([t.upper() for t in args.tickers], args.years, args.refresh, args.force)
