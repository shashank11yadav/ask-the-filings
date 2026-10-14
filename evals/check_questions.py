"""
check_questions.py - automatic checks on the eval questions (no API calls).

    uv run python evals/check_questions.py

For each answerable question it checks:
  - evidence found   : the gold evidence really appears in that company's filing for that year
  - numbers backed   : every number in the gold answer appears in the gold evidence
  - names company    : the question names the company, so a real user could ask it
  - year plausible   : any year mentioned is one the filing covers (its year or 2 before)
  - unique evidence  : no other question uses the same evidence (near-duplicate)

Prints every question that fails a check. Fix or delete those questions in evals/questions.jsonl.
"""

import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

from filings.chunking import TABLE_END, TABLE_START, html_to_text
from filings.numbers import unsupported_numbers
from filings.query_parse import parse_query

QUESTIONS = Path("evals/questions.jsonl")
RAW_DIR = Path("data/raw")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


@cache
def filing_text(ticker: str, year: int) -> str:
    """Cleaned, normalised text of one filing (cached: each file is parsed once)."""
    path = RAW_DIR / f"{ticker}_{year}.html"
    if not path.exists():
        return ""
    lines = [ln for ln in html_to_text(path.read_text()).splitlines() if ln not in (TABLE_START, TABLE_END)]
    return norm("\n".join(lines))


def problems(q: dict, evidence_counts: Counter) -> list[str]:
    """Return a list of problems found in one answerable question (empty list = looks fine)."""
    found = []
    evidence = q["gold_evidence"][0]
    if norm(evidence) not in filing_text(q["ticker"], q["year"]):
        found.append("evidence not found in filing")
    if unsupported_numbers(q["gold_answer"], [evidence]):
        found.append("answer has numbers not in the evidence")
    if parse_query(q["question"]).ticker != q["ticker"]:
        found.append("question doesn't clearly name the company")
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", q["question"])}
    if any(y > q["year"] or y < q["year"] - 2 for y in years):
        found.append(f"mentions a year outside FY{q['year'] - 2}-FY{q['year']} (fine for projections)")
    if evidence_counts[norm(evidence)] > 1:
        found.append("same evidence as another question")
    return found


def main() -> None:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    counts = Counter(norm(q["gold_evidence"][0]) for q in answerable)
    flagged = 0
    for q in answerable:
        issues = problems(q, counts)
        if issues:
            flagged += 1
            print(f"\n[{q['id']}] {q['ticker']} FY{q['year']}: {q['question']}")
            for issue in issues:
                print(f"   - {issue}")
    print(f"\n{len(answerable)} answerable questions checked, {flagged} flagged.")
    print(f"{len(questions) - len(answerable)} unanswerable questions (not checked here).")


if __name__ == "__main__":
    main()
