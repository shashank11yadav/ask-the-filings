"""
answer.py - the full RAG step: retrieve chunks, then ask Gemini to answer using ONLY them.

Gemini returns JSON with three fields:
  - answerable : False if the excerpts don't contain the answer (we WANT it to say "I don't know"
                 instead of making something up, and the eval measures how well it does this)
  - answer     : the answer text
  - citations  : the ids of the excerpts it used, so the UI can show sources
"""

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from filings import llm
from filings.config import settings
from filings.retrieve import Hit, Mode, search

# The "system instruction": rules Gemini must follow for every question.
SYSTEM = """You answer questions about companies using ONLY the provided excerpts from their SEC 10-K filings.

Rules:
- Every factual claim must be supported by an excerpt; list the ids of the excerpts you used in `citations`.
- Quote numbers exactly as they appear, with units and the fiscal year they refer to.
- If the excerpts do not contain enough information, set `answerable` to false and say what is missing. Do not use outside knowledge."""


class AnswerOut(BaseModel):
    """The JSON shape Gemini must return (passed to llm.parse as the schema)."""

    answerable: bool
    answer: str
    citations: list[int] = Field(description="ids of the excerpts that support the answer")


@dataclass
class Answer:
    """Everything about one answered question: used by the API and by the eval."""

    question: str
    answer: str
    answerable: bool
    citations: list[int]
    hits: list[Hit]  # all retrieved chunks (not just the cited ones)
    latency_s: float  # total seconds, retrieval + generation
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


def format_context(hits: list[Hit]) -> str:
    """Wrap each chunk in <excerpt> tags with its id, so Gemini can cite it by id."""
    return "\n\n".join(
        f'<excerpt id="{h.id}" company="{h.ticker}" fiscal_year="{h.fiscal_year}"'
        f' section="{h.section or "unknown"}">\n{h.text}\n</excerpt>'
        for h in hits
    )


def answer(
    question: str, k: int = 8, mode: Mode = "rerank", ticker: str | None = None, year: int | None = None
) -> Answer:
    t0 = time.perf_counter()  # start the latency timer

    # 1. Retrieve the k most relevant chunks.
    hits = search(question, k=k, mode=mode, ticker=ticker, year=year)

    # 2. Ask Gemini, giving it the excerpts followed by the question.
    result = llm.parse(
        model=settings.answer_model,
        system=SYSTEM,
        user=f"{format_context(hits)}\n\nQuestion: {question}",
        schema=AnswerOut,
    )

    # 3. If Gemini was blocked or returned nothing usable, treat it as "can't answer".
    out = result.parsed or AnswerOut(answerable=False, answer="No answer was returned.", citations=[])

    # 4. Keep only citation ids that were actually in the excerpts. LLMs occasionally invent ids.
    valid_ids = {h.id for h in hits}
    return Answer(
        question=question,
        answer=out.answer,
        answerable=out.answerable,
        citations=[c for c in out.citations if c in valid_ids],
        hits=hits,
        latency_s=time.perf_counter() - t0,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        meta={"model": result.model, "blocked": result.blocked, "mode": mode, "k": k},
    )
