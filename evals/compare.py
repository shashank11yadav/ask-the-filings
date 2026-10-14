"""Paired comparison of two run_eval.py result files.

Reports the questions each config got that the other missed, an exact McNemar test on those
disagreements, and a bootstrap 95% CI for the difference in recall. With ~100 questions a single
score is uncertain by several points, so a raw difference on its own proves little.

    uv run python evals/compare.py <run A>.jsonl <run B>.jsonl [--type table|text]
"""

import argparse
import json
import random
from math import comb
from pathlib import Path


def load(path: str, source_type: str | None = None) -> dict[str, bool]:
    """Question id -> retrieval hit, for answerable questions (optionally one source_type)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {
        r["id"]: bool(r["hit"])
        for r in rows
        if r["answerable"] and (source_type is None or r.get("source_type") == source_type)
    }


def mcnemar_exact(b_wins: int, a_wins: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the disagreeing questions."""
    n = b_wins + a_wins
    if n == 0:
        return 1.0
    k = min(b_wins, a_wins)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def bootstrap_ci(a: list[bool], b: list[bool], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for mean(b) - mean(a), resampling questions with replacement."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", help="baseline per-question results (.jsonl)")
    p.add_argument("b", help="new config per-question results (.jsonl)")
    p.add_argument("--type", choices=["table", "text"], help="only compare this kind of question")
    args = p.parse_args()

    A, B = load(args.a, args.type), load(args.b, args.type)
    ids = sorted(A.keys() & B.keys())
    a = [A[i] for i in ids]
    b = [B[i] for i in ids]
    b_wins = sum(bi and not ai for ai, bi in zip(a, b))
    a_wins = sum(ai and not bi for ai, bi in zip(a, b))
    lo, hi = bootstrap_ci(a, b)

    print(f"questions compared: {len(ids)}")
    print(f"recall  A: {sum(a) / len(a):.3f}   B: {sum(b) / len(b):.3f}   diff: {(sum(b) - sum(a)) / len(a):+.3f}")
    print(f"B fixed {b_wins} question(s) A missed; B broke {a_wins} question(s) A got right")
    print(f"McNemar exact p = {mcnemar_exact(b_wins, a_wins):.3f}   (below 0.05 = unlikely to be luck)")
    print(f"95% bootstrap CI for the difference: [{lo:+.3f}, {hi:+.3f}]   (excludes 0 = probably real)")
    if b_wins + a_wins < 10:
        print("note: fewer than 10 disagreeing questions - this comparison is underpowered; add questions.")


if __name__ == "__main__":
    main()
