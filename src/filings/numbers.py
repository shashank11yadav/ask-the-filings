"""
numbers.py - finds numbers in text and checks whether two numbers "match".

Used in three places:
  - the eval: is the number in the system's answer the same as in the gold answer?
    (checked with code, because LLM graders are unreliable with numbers)
  - the guardrail: does every number in the answer actually appear in the cited passages?
  - the question audit: do the gold answer's numbers appear in the gold evidence?

Filings write the same value in different ways: "$11,504 million", "$11.5 billion",
"11,504", "(14)" for minus 14. So matching allows:
  - small rounding differences (0.5% by default)
  - a factor of 1,000 (millions vs. billions, thousands vs. millions)
Years (e.g. 2025), days in dates ("June 30") and tiny counts (0-10) are ignored: they're
dates or labels, not the figures we care about.
"""

import re

# Optional "(" or "-", optional "$", digits with commas, optional decimals, optional ")" or "%".
_MONTH_BEFORE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*$",
    re.IGNORECASE,
)
_REFERENCE_BEFORE = re.compile(r"\b(note|notes|item|page|pages|section|exhibit)\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?%?")


def extract_numbers(text: str) -> list[float]:
    """Return the meaningful numbers in `text` (years and tiny counts removed)."""
    out = []
    for m in _NUMBER_RE.finditer(text):
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        token = m.group(0)
        is_integer = m.group(2) is None
        if is_integer and 1990 <= value <= 2040 and "," not in token:
            continue  # a year, like 2025
        if is_integer and value <= 10:
            continue  # small counts like "2 segments" or footnote markers
        if is_integer and value <= 31 and _MONTH_BEFORE.search(text[: m.start()]):
            continue  # the day in a date, like "December 31"
        if is_integer and _REFERENCE_BEFORE.search(text[: m.start()]):
            continue  # a reference like "Note 19" or "page 106", not a figure
        if token.startswith("(") and token.endswith(")") or token.startswith("-"):
            value = -value  # accounting style: (14) means -14
        out.append(value)
    return out


def same_number(a: float, b: float, tolerance: float = 0.005) -> bool:
    """True if a and b are the same figure, allowing rounding and a change of units."""
    # 1,000x: millions vs. billions. 1,000,000x: thousands vs. billions (e.g. share counts).
    for scale in (1, 1_000, 1 / 1_000, 1_000_000, 1 / 1_000_000):
        target = abs(b) * scale
        if target == 0:
            if a == 0:
                return True
            continue
        if abs(abs(a) - target) / target <= tolerance:
            return True
    return False


def unsupported_numbers(answer: str, sources: list[str]) -> list[float]:
    """Numbers in `answer` that don't appear (in any unit) anywhere in `sources`."""
    source_numbers = [n for s in sources for n in extract_numbers(s)]
    return [a for a in extract_numbers(answer) if not any(same_number(a, s) for s in source_numbers)]


def numbers_match(gold: str, predicted: str, all_numbers: bool = False) -> bool | None:
    """True if the gold answer's main figure (its first number) appears in the predicted answer.

    Gold answers often add context ("24.1%, a provision of $4,829 million"), and a correct
    answer may give only the main figure, so by default only the first number is required.
    all_numbers=True requires every gold number instead.
    Returns None when the gold answer has no numbers (then there's nothing to check in code).
    """
    gold_numbers = extract_numbers(gold)
    if not gold_numbers:
        return None
    required = gold_numbers if all_numbers else gold_numbers[:1]
    predicted_numbers = extract_numbers(predicted)
    return all(any(same_number(g, p) for p in predicted_numbers) for g in required)
