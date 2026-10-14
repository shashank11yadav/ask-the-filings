"""Input limits and the fixed refusal texts used by answer.py.

Refusals are written here rather than taken from the model, so injected instructions can't
change what a refusal says.
"""

from typing import Literal

MAX_QUESTION_CHARS = 500

RequestType = Literal["filing_question", "investment_advice", "prediction", "off_topic"]

REFUSALS: dict[str, str] = {
    "investment_advice": (
        "I can't give investment advice or opinions on whether a stock is a good buy. "
        "I can answer factual questions about what the companies report in their 10-K filings."
    ),
    "prediction": (
        "I can't predict future results or prices. 10-K filings report past results, "
        "so ask about a figure the filings cover."
    ),
    "off_topic": (
        "I only answer questions about the 10-K filings of the companies in this database."
    ),
}

NO_SUPPORT = "I couldn't find support for an answer in the retrieved passages."


class GuardrailError(ValueError):
    """Question rejected before reaching the model; the message is shown to the user."""


def check_question(question: str) -> str:
    q = " ".join(question.split())
    if not q:
        raise GuardrailError("Please type a question.")
    if len(q) > MAX_QUESTION_CHARS:
        raise GuardrailError(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    return q
