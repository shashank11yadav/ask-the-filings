"""Rule-based detection of the company and fiscal year a question is about.

Without a company filter, near-identical boilerplate from other companies crowds out the answer
(realistic recall 0.77 without it, 0.88 with it). When unsure, return None and search everything.
"""

import re
from dataclasses import dataclass

# Whole-word, case-insensitive. Add an entry when a new company is ingested.
ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft", "linkedin"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google", "youtube"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase", "jpmorganchase", "chase"],
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
    return re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)


# Tickers also match their own symbol, case-sensitively.
_PATTERNS: dict[str, list[re.Pattern]] = {
    t: [_pattern(a) for a in names] + [re.compile(rf"\b{t}\b")] for t, names in ALIASES.items()
}


def parse_query(question: str, known_years: set[int] | None = None) -> ParsedQuery:
    # More than one company mentioned means a comparison: don't filter.
    found = {t for t, pats in _PATTERNS.items() if any(p.search(question) for p in pats)}
    ticker = found.pop() if len(found) == 1 else None

    # Only a single loaded year is used: a 2024 10-K also reports 2023 figures.
    years = {int(y) for y in YEAR_RE.findall(question)}
    year = None
    if len(years) == 1 and (known_years is None or next(iter(years)) in known_years):
        year = years.pop()
    return ParsedQuery(ticker, year)
