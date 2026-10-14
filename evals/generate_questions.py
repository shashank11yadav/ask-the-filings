"""Draft eval questions with Gemini from passages of the downloaded filings.

Passages are sampled from the cleaned filing text rather than from database chunks, so the
questions don't favour the current chunker. About half start at a table, and each question is
tagged with source_type. Gemini is asked to paraphrase, since questions that copy the filing's
wording flatter keyword search. Run `make check-questions` afterwards.

    uv run python evals/generate_questions.py --n 60 [--out evals/questions.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from filings import llm
from filings.chunking import TABLE_END, TABLE_START, html_to_text
from filings.config import settings

RAW_DIR = Path("data/raw")
PASSAGE_WORDS = 300

SYSTEM = """You write evaluation questions for a question-answering system over SEC 10-K filings.
Given one excerpt, write a question an analyst might realistically ask that this excerpt answers.
Prefer specific facts: figures, risks, segments, year-over-year changes. Avoid questions answerable from general knowledge.
The question must name the company (e.g. "Walmart's", not "the Company's" or "the Firm's") and must make sense
to someone who has NOT seen the excerpt, as if a user typed it into a search box.
Paraphrase: use the words a user would use, not the excerpt's exact phrasing.
`evidence` must be a short span copied VERBATIM from the excerpt (10-30 words) that proves the answer.
For a table, the evidence should include the row label and the value(s), exactly as written (including the " | " separators).
If the excerpt has nothing worth asking about (boilerplate, table of contents, signatures), set usable to false."""


class Generated(BaseModel):
    usable: bool
    question: str
    answer: str
    evidence: str = Field(description="verbatim span from the excerpt")


def load_filings() -> list[tuple[str, int, list[str]]]:
    """[(ticker, fiscal_year, lines), ...] for every downloaded filing."""
    out = []
    for path in sorted(RAW_DIR.glob("*_*.html")):
        ticker, year = path.stem.rsplit("_", 1)
        out.append((ticker, int(year), html_to_text(path.read_text()).splitlines()))
    return out


def passage(lines: list[str], start: int) -> tuple[str, bool]:
    """Whole lines from `start` up to about PASSAGE_WORDS words. Returns (text, has_table)."""
    taken, words, has_table = [], 0, False
    for line in lines[start:]:
        if line in (TABLE_START, TABLE_END):
            has_table = has_table or line == TABLE_START
            continue
        taken.append(line)
        words += len(line.split())
        if words >= PASSAGE_WORDS:
            break
    return "\n".join(taken), has_table


def sample_passages(filings, n: int, rng: random.Random) -> list[tuple[str, int, str, str]]:
    """About half start at a table (including its caption line), half at a random line."""
    picks = []
    for i in range(n):
        ticker, year, lines = rng.choice(filings)
        want_table = i % 2 == 0
        if want_table:
            table_starts = [j for j, ln in enumerate(lines) if ln == TABLE_START]
            if not table_starts:
                continue
            start = max(0, rng.choice(table_starts) - 1)
        else:
            start = rng.randrange(len(lines))
        text, has_table = passage(lines, start)
        if len(text.split()) >= 80:
            picks.append((ticker, year, text, "table" if has_table else "text"))
    return picks


def main(n: int, out: str, seed: int) -> None:
    rng = random.Random(seed)
    filings = load_filings()
    if not filings:
        raise SystemExit("No filings in data/raw/. Run `make ingest` first.")

    candidates = sample_passages(filings, n * 2, rng)

    written = 0
    with open(out, "a") as f:
        for ticker, year, text, source_type in candidates:
            if written >= n:
                break
            r = llm.parse(
                model=settings.answer_model,
                system=SYSTEM,
                user=f"Company: {ticker}, fiscal year {year}\n\n<excerpt>\n{text}\n</excerpt>",
                schema=Generated,
                temperature=0.7,
            )
            g = r.parsed
            # The eval matches evidence text exactly, so it must be a verbatim quote.
            norm = lambda s: " ".join(s.split())
            if not g or not g.usable or norm(g.evidence) not in norm(text):
                continue
            f.write(json.dumps({
                "id": f"gen-{ticker}-{year}-{rng.randrange(10**8):08d}",
                "question": g.question,
                "ticker": ticker,
                "year": year,
                "gold_answer": g.answer,
                "gold_evidence": [g.evidence],
                "answerable": True,
                "source_type": source_type,
            }) + "\n")
            f.flush()
            written += 1
    print(f"wrote {written} questions to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", default="evals/questions.jsonl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.n, a.out, a.seed)
