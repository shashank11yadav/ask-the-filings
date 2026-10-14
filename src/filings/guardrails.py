"""
guardrails.py - checks that run before and after Gemini, so the system stays on its job.

Before the model:
  - check_question(): reject empty or very long questions (long inputs cost money and are a
    common way to smuggle in instructions).

Inside the model call (see answer.py):
  - Gemini also labels each request: a real filing question, investment advice, a prediction,
    or off-topic. Anything that isn't a filing question gets a fixed, polite refusal written
    here, not by the model, so the wording can't be manipulated.
  - The prompt says text inside <excerpt> tags is data, never instructions.

After the model:
  - An "answer" with no valid citation is turned into "couldn't find support".
  - Numbers in the answer that don't appear in the cited passages are listed as warnings,
    so the user (and the eval) can see possibly unsupported figures.
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
    """Raised when a question is rejected before reaching the model. The message is user-facing."""


def check_question(question: str) -> str:
    """Return the cleaned question, or raise GuardrailError with a message for the user."""
    q = " ".join(question.split())  # collapse whitespace and newlines
    if not q:
        raise GuardrailError("Please type a question.")
    if len(q) > MAX_QUESTION_CHARS:
        raise GuardrailError(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    return q
