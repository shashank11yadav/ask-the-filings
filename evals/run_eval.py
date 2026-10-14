"""
run_eval.py - measures how good the system is, so you can prove improvements with numbers.

It runs every question in evals/questions.jsonl through the system once per search mode
(dense / sparse / hybrid / rerank) and prints a table like this for your README:

| config     | n  | recall@k | correct | faithful | abstain_acc | p50_s | p95_s | usd_per_q |

What each metric means:
- recall@k    : Did search find the right chunk? (Does a chunk among the top k contain the gold evidence text?)
- mrr         : How close to the top was it? 1.0 = always ranked first, 0.5 = typically second.
                Checked by matching text, not chunk ids, so it still works after you change chunk sizes.
- correct     : Does the answer match the gold answer? A second Gemini call (the "judge") decides.
- faithful    : Is every claim in the answer backed by the chunks it cited? (Catches made-up facts.)
- abstain_acc : For questions with NO answer in the filings, how often did the system correctly say so?
- p50_s/p95_s : Typical and worst-case (95th percentile) response time in seconds.
- usd_per_q   : Average API cost per question, including the judge.

Usage:
  uv run python evals/run_eval.py --retrieval-only     # free: only measures search (no Gemini calls)
  uv run python evals/run_eval.py --limit 20           # quick, cheap full run on 20 questions
  uv run python evals/run_eval.py                      # full run on all questions
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from pydantic import BaseModel

from filings import llm
from filings.answer import answer, format_context
from filings.config import settings
from filings.retrieve import search

# Price per 1 million tokens in USD: (input, output). Output includes "thinking" tokens.
# These are paid-tier rates as of Oct 2026 (the free tier costs $0 but has rate limits).
# Prices change; check https://ai.google.dev/gemini-api/docs/pricing and update them.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),  # rises to (1.50, 7.50) on 1 Jan 2027
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),  # prompts up to 200k tokens
    "gemini-2.5-flash": (0.30, 2.50),
}

# Models we've already warned about, so the warning prints once instead of per question.
_unknown_models: set[str] = set()

# Instructions for the judge model.
JUDGE_SYSTEM = """You grade answers from a question-answering system over SEC filings.
correct: the system answer states the same key facts as the reference answer (wording may differ; numbers must match).
faithful: every factual claim in the system answer is supported by the cited excerpts. An answer with no claims is faithful."""


class Verdict(BaseModel):
    """The judge's JSON output. Asking for `reasoning` first makes the verdict more careful."""

    reasoning: str
    correct: bool
    faithful: bool


def norm(s: str) -> str:
    """Lower-case and collapse whitespace so small formatting differences don't matter."""
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_hit(gold: list[str], texts: list[str]) -> bool:
    """True if any gold evidence span appears inside any retrieved chunk."""
    joined = [norm(t) for t in texts]
    return any(norm(g) in t for g in gold for t in joined)


def reciprocal_rank(gold: list[str], texts: list[str]) -> float:
    """1/rank of the first retrieved chunk containing gold evidence (1.0 if it's first, 0.5 if
    second, ... 0 if not found). Averaged over questions this is MRR: it rewards putting the
    right chunk near the TOP, which recall@k ignores."""
    for rank, t in enumerate(texts, start=1):
        if any(norm(g) in norm(t) for g in gold):
            return 1.0 / rank
    return 0.0


def cost(model: str, inp: int, out: int) -> float:
    """Dollar cost of one call. Matches on prefix, since Gemini may report e.g. 'gemini-3.8-flash-001'."""
    # Check longer names first, so "gemini-3.5-flash-lite" isn't matched as a shorter name.
    for name, (pin, pout) in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(name):
            return (inp * pin + out * pout) / 1e6
    # Unknown model: warn (once) instead of silently reporting $0, which would make the table wrong.
    if model not in _unknown_models:
        _unknown_models.add(model)
        print(f"WARNING: no price for {model!r}; add it to PRICES in run_eval.py. Counting it as $0.")
    return 0.0


def evidence_coverage(questions: list[dict]) -> float:
    """Share of answerable questions whose gold evidence still exists, intact, inside SOME chunk.

    Gold evidence was copied from the chunks that existed when the questions were generated.
    If you change the chunker, a quote can end up split across two chunks; then no search can
    "find" it and recall drops for reasons that have nothing to do with search quality.
    Check this number when comparing chunkers: it's the ceiling on recall@k.
    """
    from filings.db import connect

    ans = [q for q in questions if q["answerable"]]
    found = 0
    with connect() as conn:
        for q in ans:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE ticker = %s AND fiscal_year = %s"
                " AND position(%s in regexp_replace(lower(text), '\\s+', ' ', 'g')) > 0 LIMIT 1",
                [q["ticker"], q["year"], norm(q["gold_evidence"][0])],
            ).fetchone()
            found += row is not None
    return found / len(ans) if ans else float("nan")


def pct(xs: list[float], p: float) -> float:
    """Simple percentile, e.g. p=0.95 gives the value 95% of runs were faster than."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run(questions: list[dict], mode: str, k: int, retrieval_only: bool, out_dir: Path,
        filters: bool = True, auto_filter: str = "company") -> dict:
    """Run all questions with one search mode and return a summary row for the table."""
    rows = []
    for q in questions:
        # By default the question's company and year are passed to search as filters. With
        # --no-filters, search must work them out from the question (see --auto-filter), like a
        # real user's question.
        ticker, year = (q.get("ticker"), q.get("year")) if filters else (None, None)
        # --- Retrieval-only mode: just search, no LLM calls, costs nothing ---
        if retrieval_only:
            t0 = time.perf_counter()
            hits = search(q["question"], k=k, mode=mode, ticker=ticker, year=year, auto_filter=auto_filter)
            rows.append({"id": q["id"], "answerable": q["answerable"], "source_type": q.get("source_type"),
                         "hit": evidence_hit(q.get("gold_evidence", []), [h.text for h in hits]),
                         "rr": reciprocal_rank(q.get("gold_evidence", []), [h.text for h in hits]),
                         "latency": time.perf_counter() - t0, "cost": 0.0})
            continue

        # --- Full mode: retrieve + answer with Gemini ---
        a = answer(q["question"], k=k, mode=mode, ticker=ticker, year=year)
        row = {"id": q["id"], "answerable": q["answerable"], "source_type": q.get("source_type"),
               "pred_answerable": a.answerable,
               "answer": a.answer, "citations": a.citations, "latency": a.latency_s,
               "hit": evidence_hit(q.get("gold_evidence", []), [h.text for h in a.hits]),
               "rr": reciprocal_rank(q.get("gold_evidence", []), [h.text for h in a.hits]),
               "cost": cost(a.meta["model"], a.input_tokens, a.output_tokens)}

        # Only judge when there IS a gold answer and the system attempted an answer.
        if q["answerable"] and a.answerable:
            cited = [h for h in a.hits if h.id in a.citations]
            v = llm.parse(
                model=settings.judge_model,
                system=JUDGE_SYSTEM,
                user=f"Question: {q['question']}\n\nReference answer: {q['gold_answer']}\n\n"
                     f"System answer: {a.answer}\n\nCited excerpts:\n{format_context(cited)}",
                schema=Verdict,
            )
            if v.parsed:
                row.update(correct=v.parsed.correct, faithful=v.parsed.faithful,
                           judge_reasoning=v.parsed.reasoning)
            row["cost"] += cost(v.model, v.input_tokens, v.output_tokens)
        rows.append(row)

    # Save every question's result, so you can read the failures and figure out what to fix.
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if filters else f"_nofilters_{auto_filter.replace('+', '_')}"
    with open(out_dir / f"{mode}_k{k}{suffix}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    # Compute the summary numbers.
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    judged = [r for r in ans if "correct" in r]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    lat = [r["latency"] for r in rows]
    return {
        "config": f"{mode} k={k}" + ("" if filters else f" no-filters auto={auto_filter}"),
        "n": len(rows),
        "recall@k": mean([r["hit"] for r in ans]),
        "mrr": mean([r["rr"] for r in ans]),
        # Recall split by question type (only for question sets that have "source_type").
        "recall_table": mean([r["hit"] for r in ans if r.get("source_type") == "table"]),
        "recall_text": mean([r["hit"] for r in ans if r.get("source_type") == "text"]),
        # If the system wrongly said "can't answer" on an answerable question, that counts as incorrect.
        "correct": mean([r.get("correct", False) for r in ans]) if not retrieval_only else None,
        "faithful": mean([r["faithful"] for r in judged]) if judged else None,
        "abstain_acc": mean([not r.get("pred_answerable", True) for r in unans]) if unans and not retrieval_only else None,
        "p50_s": statistics.median(lat) if lat else 0.0,
        "p95_s": pct(lat, 0.95),
        "usd_per_q": mean([r["cost"] for r in rows]),
    }


def fmt(v) -> str:
    """Format a table cell: 3 decimals for floats, a dash for 'not measured'."""
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="evals/questions.jsonl")
    p.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid", "rerank"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--retrieval-only", action="store_true", help="skip generation + judge (free)")
    p.add_argument("--verified-only", action="store_true", help="only use hand-checked questions")
    p.add_argument("--no-filters", action="store_true",
                   help="don't tell search the company/year (realistic, harder)")
    p.add_argument("--auto-filter", default="company", choices=["off", "company", "company+year"],
                   help="with --no-filters: what search may detect from the question itself")
    a = p.parse_args()

    # Load questions (one JSON object per line).
    lines = Path(a.questions).read_text().splitlines()
    qs = [json.loads(line) for line in lines if line.strip()]
    if a.verified_only:
        qs = [q for q in qs if q.get("verified")]
    qs = qs[: a.limit] if a.limit else qs

    # Each run gets its own timestamped folder, so old results are never overwritten.
    out_dir = Path("evals/results") / time.strftime("%Y%m%d-%H%M%S")

    results = [run(qs, m, a.k, a.retrieval_only, out_dir, not a.no_filters, a.auto_filter) for m in a.modes]

    # Build a Markdown table you can paste straight into the README.
    cols = list(results[0])
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in results)
    cov = evidence_coverage(qs)
    table += f"\n\nevidence coverage (ceiling on recall): {cov:.3f}"
    print(table)
    (out_dir / "summary.md").write_text(table + "\n")
    print(f"\nper-question results in {out_dir}/")


if __name__ == "__main__":
    main()
