"""
Tests for the small helper functions used by the eval and by hybrid search.
These run offline: no database, no Gemini key needed.
"""

import sys
from pathlib import Path

import pytest

# run_eval.py lives in evals/, which isn't a package, so add that folder to Python's import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import run_eval

from filings.retrieve import Hit, rrf


def test_evidence_hit_ignores_whitespace_and_case():
    assert run_eval.evidence_hit(["Net  Sales increased"], ["... net sales\nincreased 2% ..."])
    assert not run_eval.evidence_hit(["operating margin"], ["net sales increased"])


def test_cost_matches_model_version_prefix():
    # 1M input tokens on gemini-3.8-flash = $0.75, even if Gemini reports a versioned name.
    assert run_eval.cost("gemini-3.8-flash-001", 1_000_000, 0) == pytest.approx(0.75)
    # 1M output tokens on gemini-3.5-flash-lite = $2.50
    assert run_eval.cost("gemini-3.5-flash-lite", 0, 1_000_000) == pytest.approx(2.50)


def _hit(i: int) -> Hit:
    return Hit(id=i, ticker="X", fiscal_year=2025, section=None, text="", score=0.0)


def test_rrf_rewards_agreement():
    dense = [_hit(1), _hit(2), _hit(3)]
    sparse = [_hit(3), _hit(4), _hit(1)]
    ids = [h.id for h in rrf([dense, sparse], k=4)]
    assert ids[:2] == [1, 3]  # chunks found by BOTH searches should rank first


def test_reciprocal_rank():
    texts = ["nothing here", "Net sales increased", "net sales increased again"]
    assert run_eval.reciprocal_rank(["net sales increased"], texts) == pytest.approx(0.5)
    assert run_eval.reciprocal_rank(["operating margin"], texts) == 0.0
