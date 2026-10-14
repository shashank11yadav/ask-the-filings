"""
review.py - check the eval questions, automatically and by hand.

    uv run python evals/review.py check     # automatic checks, prints a report, saves "flags"
    uv run python evals/review.py review    # go through unverified questions one by one

Automatic checks (no API calls):
  - evidence_found  : the gold evidence really appears in that company's filing for that year
  - numbers_backed  : every number in the gold answer appears in the gold evidence
  - names_company   : the question names the company, so a real user could ask it
  - year_plausible  : any year mentioned is one the filing actually covers (its year or 2 before)
  - unique_evidence : no other question uses the same evidence (near-duplicate)

`review` shows each question with its evidence and any flags, and you press:
  y = correct, mark verified    n = wrong, mark rejected (dropped from evals)
  s = skip for now               q = save and quit
Answers are saved after every question, so you can stop and continue later.
"""

import argparse
import json
import re
import sys
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


def load() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def save(questions: list[dict]) -> None:
    QUESTIONS.write_text("".join(json.dumps(q) + "\n" for q in questions))


def auto_flags(q: dict, evidence_counts: Counter) -> list[str]:
    """Return a list of problems found in one answerable question (empty list = looks fine)."""
    flags = []
    evidence = q["gold_evidence"][0]
    if norm(evidence) not in filing_text(q["ticker"], q["year"]):
        flags.append("evidence not found in filing")
    if unsupported_numbers(q["gold_answer"], [evidence]):
        flags.append("answer has numbers not in the evidence")
    if parse_query(q["question"]).ticker != q["ticker"]:
        flags.append("question doesn't clearly name the company")
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", q["question"])}
    if any(y > q["year"] or y < q["year"] - 2 for y in years):
        flags.append(f"mentions a year outside FY{q['year'] - 2}-FY{q['year']}")
    if evidence_counts[norm(evidence)] > 1:
        flags.append("same evidence as another question")
    return flags


def check() -> None:
    questions = load()
    answerable = [q for q in questions if q["answerable"]]
    counts = Counter(norm(q["gold_evidence"][0]) for q in answerable)
    flagged = 0
    for q in answerable:
        q["flags"] = auto_flags(q, counts)
        if q["flags"]:
            flagged += 1
            print(f"\n[{q['id']}] {q['ticker']} FY{q['year']}: {q['question']}")
            for f in q["flags"]:
                print(f"   - {f}")
    save(questions)
    status = Counter(q.get("review", "unreviewed") for q in questions)
    print(f"\n{len(answerable)} answerable questions checked, {flagged} flagged.")
    print(f"{len(questions) - len(answerable)} unanswerable questions (not auto-checked).")
    print(f"Review status: {dict(status)}")


def review() -> None:
    questions = load()
    todo = [q for q in questions if q.get("review", "unreviewed") == "unreviewed"]
    print(f"{len(todo)} questions to review. y = correct, n = wrong, s = skip, q = quit\n")
    for i, q in enumerate(todo, start=1):
        print("=" * 80)
        print(f"({i}/{len(todo)}) {q['ticker']} FY{q['year']}  [{q.get('source_type', q.get('category', ''))}]")
        print(f"Q: {q['question']}")
        if q["answerable"]:
            print(f"A: {q['gold_answer']}")
            print(f"Evidence: {q['gold_evidence'][0]}")
        else:
            print(f"Expected: the system should say it can't answer ({q.get('category')})")
        for f in q.get("flags", []):
            print(f"  ! {f}")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice in ("y", "n"):
            q["review"] = "verified" if choice == "y" else "rejected"
            q["verified"] = choice == "y"
            save(questions)  # save after each answer
    done = Counter(q.get("review", "unreviewed") for q in questions)
    print(f"\nSaved. Review status: {dict(done)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "review"])
    mode = p.parse_args().mode
    if mode == "review" and not sys.stdin.isatty():
        raise SystemExit("review is interactive: run it in a terminal.")
    check() if mode == "check" else review()
