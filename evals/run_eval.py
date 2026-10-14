"""Run the eval set and print a results table.

Metrics (answerable questions unless noted):
  recall@k, mrr  gold evidence text found in the top k chunks / reciprocal rank of the first hit
  correct        LLM judge: answer agrees with the gold answer (refusals count as wrong)
  faithful       LLM judge: claims are supported by the cited chunks
  numeric_ok     the gold answer's main figure appears in the answer (checked in code)
  grounded       answers whose numbers all appear in their cited chunks
  abstain_acc    unanswerable questions that were refused
  over_refusal   answerable questions that were refused

    uv run python evals/run_eval.py --retrieval-only           # no API calls
    uv run python evals/run_eval.py --modes rerank --no-filters
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
from filings.numbers import numbers_match
from filings.retrieve import search

# USD per 1M tokens (input, output), paid tier, October 2026. Thinking tokens bill as output.
# https://ai.google.dev/gemini-api/docs/pricing
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),  # rises to (1.50, 7.50) on 1 Jan 2027
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),  # prompts up to 200k tokens
    "gemini-2.5-flash": (0.30, 2.50),
}

_unknown_models: set[str] = set()

JUDGE_SYSTEM = """You grade answers from a question-answering system over SEC filings.
correct: the system answer states the same key facts as the reference answer (wording may differ; numbers must match).
faithful: every factual claim in the system answer is supported by the cited excerpts. An answer with no claims is faithful."""


class Verdict(BaseModel):
    reasoning: str  # first, so the verdict is written after the reasoning
    correct: bool
    faithful: bool


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_hit(gold: list[str], texts: list[str]) -> bool:
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
    # Prefix match (Gemini may report a versioned name); longest name first.
    for name, (pin, pout) in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(name):
            return (inp * pin + out * pout) / 1e6
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
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run(questions: list[dict], mode: str, k: int, retrieval_only: bool, out_dir: Path,
        filters: bool = True, auto_filter: str = "company") -> dict:
    rows = []
    for q in questions:
        # --no-filters makes search work out the company from the question, as with real users.
        ticker, year = (q.get("ticker"), q.get("year")) if filters else (None, None)
        if retrieval_only:
            t0 = time.perf_counter()
            hits = search(q["question"], k=k, mode=mode, ticker=ticker, year=year, auto_filter=auto_filter)
            rows.append({"id": q["id"], "answerable": q["answerable"], "source_type": q.get("source_type"),
                         "hit": evidence_hit(q.get("gold_evidence", []), [h.text for h in hits]),
                         "rr": reciprocal_rank(q.get("gold_evidence", []), [h.text for h in hits]),
                         "latency": time.perf_counter() - t0, "cost": 0.0})
            continue

        a = answer(q["question"], k=k, mode=mode, ticker=ticker, year=year)
        row = {"id": q["id"], "answerable": q["answerable"], "source_type": q.get("source_type"),
               "category": q.get("category"), "request_type": a.request_type,
               "unsupported_numbers": a.unsupported_numbers,
               "pred_answerable": a.answerable,
               "answer": a.answer, "citations": a.citations, "latency": a.latency_s,
               "hit": evidence_hit(q.get("gold_evidence", []), [h.text for h in a.hits]),
               "rr": reciprocal_rank(q.get("gold_evidence", []), [h.text for h in a.hits]),
               "cost": cost(a.meta["model"], a.input_tokens, a.output_tokens)}

        if q["answerable"]:
            match = numbers_match(q["gold_answer"], a.answer)
            row["numeric_ok"] = None if match is None else (match and a.answerable)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if filters else f"_nofilters_{auto_filter.replace('+', '_')}"
    with open(out_dir / f"{mode}_k{k}{suffix}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

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
        "recall_table": mean([r["hit"] for r in ans if r.get("source_type") == "table"]),
        "recall_text": mean([r["hit"] for r in ans if r.get("source_type") == "text"]),
        "correct": mean([r.get("correct", False) for r in ans]) if not retrieval_only else None,
        "faithful": mean([r["faithful"] for r in judged]) if judged else None,
        "numeric_ok": mean([r["numeric_ok"] for r in ans if r.get("numeric_ok") is not None]) if not retrieval_only else None,
        "grounded": mean([not r["unsupported_numbers"] for r in rows if r.get("pred_answerable")]) if not retrieval_only else None,
        "abstain_acc": mean([not r.get("pred_answerable", True) for r in unans]) if unans and not retrieval_only else None,
        "over_refusal": mean([not r.get("pred_answerable", True) for r in ans]) if not retrieval_only else None,
        "p50_s": statistics.median(lat) if lat else 0.0,
        "p95_s": pct(lat, 0.95),
        "usd_per_q": mean([r["cost"] for r in rows]),
    }


def fmt(v) -> str:
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
    p.add_argument("--no-filters", action="store_true",
                   help="don't tell search the company/year (realistic, harder)")
    p.add_argument("--auto-filter", default="company", choices=["off", "company", "company+year"],
                   help="with --no-filters: what search may detect from the question itself")
    a = p.parse_args()

    lines = Path(a.questions).read_text().splitlines()
    qs = [json.loads(line) for line in lines if line.strip()]
    qs = qs[: a.limit] if a.limit else qs

    out_dir = Path("evals/results") / time.strftime("%Y%m%d-%H%M%S")

    results = [run(qs, m, a.k, a.retrieval_only, out_dir, not a.no_filters, a.auto_filter) for m in a.modes]

    cols = list(results[0])
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in results)
    cov = evidence_coverage(qs)
    table += f"\n\nevidence coverage (ceiling on recall): {cov:.3f}"

    if not a.retrieval_only and any(not q["answerable"] for q in qs):
        for path in sorted(out_dir.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            by_cat: dict[str, list[bool]] = {}
            for r in rows:
                if not r["answerable"]:
                    by_cat.setdefault(r["category"], []).append(not r["pred_answerable"])
            table += f"\n\nrefused unanswerable questions ({path.stem}):"
            for cat, refused in sorted(by_cat.items()):
                table += f"\n  {cat}: {sum(refused)}/{len(refused)}"
    print(table)
    (out_dir / "summary.md").write_text(table + "\n")
    print(f"\nper-question results in {out_dir}/")


if __name__ == "__main__":
    main()
