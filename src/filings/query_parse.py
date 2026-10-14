"""
query_parse.py - works out which company (and fiscal year) a question is about.

Why: without filters, search looks through ALL companies' filings, and very similar
boilerplate ("we use interest rate swaps to hedge...") from the wrong company can push out
the right chunk. Our eval showed recall dropping from 0.92 to 0.77 without filters.

How: simple rules, no LLM call (so it's instant and free):
  - company: look for a known name or ticker in the question ("Apple's", "JPMorgan", "NVDA")
  - year:    only if the question mentions exactly ONE year we have a filing for

If nothing is detected we return None and search runs over everything, which is the safe default.
"""

import re
from dataclasses import dataclass

# Names people use for each company -> ticker. Add a line here when you ingest a new company.
# Matching is case-insensitive and on whole words, so "Apple" won't match "pineapple".
ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google", "youtube"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase", "chase"],
    "XOM": ["exxon", "exxonmobil", "exxon mobil"],
    "UNH": ["unitedhealth", "united health", "unitedhealthcare", "optum"],
    "WMT": ["walmart", "wal-mart", "sam's club"],
    "TSLA": ["tesla"],
}

YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass
class ParsedQuery:
    ticker: str | None
    year: int | None


def _pattern(alias: str) -> re.Pattern:
    # \b = word boundary. Allows a possessive "'s" after the name ("Apple's").
    return re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)


# Build the patterns once. Each ticker also matches its own symbol, case-sensitive ("NVDA").
_PATTERNS: dict[str, list[re.Pattern]] = {
    t: [_pattern(a) for a in names] + [re.compile(rf"\b{t}\b")] for t, names in ALIASES.items()
}


def parse_query(question: str, known_years: set[int] | None = None) -> ParsedQuery:
    """Detect the company and fiscal year a question refers to (None when unsure)."""
    # Company: keep it only if exactly one company is mentioned. A question naming two
    # companies ("Compare Apple and Microsoft...") must search both, so no filter.
    found = {t for t, pats in _PATTERNS.items() if any(p.search(question) for p in pats)}
    ticker = found.pop() if len(found) == 1 else None

    # Year: only filter if exactly one year is mentioned AND we have that fiscal year loaded.
    # "2024 and 2023" -> no filter: the 2024 10-K also reports 2023 numbers, so either could answer.
    years = {int(y) for y in YEAR_RE.findall(question)}
    year = None
    if len(years) == 1 and (known_years is None or next(iter(years)) in known_years):
        year = years.pop()
    return ParsedQuery(ticker, year)
