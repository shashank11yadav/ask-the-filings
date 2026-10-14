"""Tests for guardrails.py (input checks) and the output guardrails in answer.generate()."""

import pytest

from filings import answer as answer_mod
from filings.guardrails import NO_SUPPORT, REFUSALS, GuardrailError, check_question
from filings.llm import LLMResult
from filings.retrieve import Hit


def test_check_question_cleans_and_rejects():
    assert check_question("  What was\nApple's revenue?  ") == "What was Apple's revenue?"
    with pytest.raises(GuardrailError):
        check_question("   ")
    with pytest.raises(GuardrailError):
        check_question("x" * 501)


def _fake_gemini(monkeypatch, **fields):
    """Make generate() receive a fixed model output instead of calling Gemini."""
    out = answer_mod.AnswerOut(**fields)
    result = LLMResult(parsed=out, blocked=False, input_tokens=1, output_tokens=1, model="test")
    monkeypatch.setattr(answer_mod.llm, "parse", lambda **_: result)


HITS = [Hit(id=7, ticker="WMT", fiscal_year=2025, section="Item 8",
            text="Advertising costs were $ 5.1 billion, $ 4.4 billion and $ 4.1 billion", score=1.0)]


def test_advice_gets_fixed_refusal(monkeypatch):
    _fake_gemini(monkeypatch, request_type="investment_advice", answerable=True,
                 answer="Yes, buy it!", citations=[7])
    a = answer_mod.generate("Should I buy Walmart?", HITS)
    assert not a.answerable and a.answer == REFUSALS["investment_advice"] and a.citations == []


def test_answer_without_citations_is_withheld(monkeypatch):
    _fake_gemini(monkeypatch, request_type="filing_question", answerable=True,
                 answer="Advertising costs were $5.1 billion.", citations=[999])  # invented id
    a = answer_mod.generate("Walmart advertising costs?", HITS)
    assert not a.answerable and a.answer == NO_SUPPORT


def test_unsupported_numbers_are_flagged(monkeypatch):
    _fake_gemini(monkeypatch, request_type="filing_question", answerable=True,
                 answer="Advertising costs were $5.1 billion in 2025 and $6.3 billion in 2026.",
                 citations=[7])
    a = answer_mod.generate("Walmart advertising costs?", HITS)
    assert a.answerable and a.unsupported_numbers == [6.3]
