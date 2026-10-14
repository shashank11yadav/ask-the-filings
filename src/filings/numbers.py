"""Number extraction and matching for 10-K figures.

The same value can appear as "$11,504 million", "$11.5 billion" or "(14)" for -14, so matching
allows 0.5% rounding and unit changes of 1,000x or 1,000,000x. Years, days in dates, section
references ("Note 19") and small integers up to 10 are ignored as they are rarely the figure asked for.
"""

import re

_MONTH_BEFORE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*$",
    re.IGNORECASE,
)
_REFERENCE_BEFORE = re.compile(r"\b(note|notes|item|page|pages|section|exhibit)\s*$", re.IGNORECASE)
# Optional "(" or "-", optional "$", digits with commas, optional decimals, optional ")" or "%".
_NUMBER_RE = re.compile(r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?%?")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER_RE.finditer(text):
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        token = m.group(0)
        is_integer = m.group(2) is None
        if is_integer and 1990 <= value <= 2040 and "," not in token:
            continue  # year
        if is_integer and value <= 10:
            continue  # small counts, footnote markers
        if is_integer and value <= 31 and _MONTH_BEFORE.search(text[: m.start()]):
            continue  # day of a date
        if is_integer and _REFERENCE_BEFORE.search(text[: m.start()]):
            continue  # "Note 19", "page 106"
        if token.startswith("(") and token.endswith(")") or token.startswith("-"):
            value = -value
        out.append(value)
    return out


def same_number(a: float, b: float, tolerance: float = 0.005) -> bool:
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
    """Numbers in `answer` that don't appear in any of `sources`."""
    source_numbers = [n for s in sources for n in extract_numbers(s)]
    return [a for a in extract_numbers(answer) if not any(same_number(a, s) for s in source_numbers)]


def numbers_match(gold: str, predicted: str, all_numbers: bool = False) -> bool | None:
    """Whether the gold answer's first number (its main figure) appears in `predicted`.

    Gold answers often add context figures a correct answer can leave out. None when the gold
    answer has no numbers.
    """
    gold_numbers = extract_numbers(gold)
    if not gold_numbers:
        return None
    required = gold_numbers if all_numbers else gold_numbers[:1]
    predicted_numbers = extract_numbers(predicted)
    return all(any(same_number(g, p) for p in predicted_numbers) for g in required)
