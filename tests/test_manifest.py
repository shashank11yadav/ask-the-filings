import json
from pathlib import Path

import pytest

from filings import ingest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "filings.json").read_text())


def test_manifest_entries_are_unique_and_complete():
    keys = [(f["ticker"], f["fiscal_year"]) for f in MANIFEST]
    assert len(keys) == len(set(keys))
    for f in MANIFEST:
        assert f["url"].startswith("https://www.sec.gov/Archives/edgar/data/")
        assert len(f["text_sha256"]) == 64


def test_every_eval_question_uses_a_pinned_filing():
    pinned = {(f["ticker"], f["fiscal_year"]) for f in MANIFEST}
    questions = [json.loads(line) for line in (ROOT / "evals/questions.jsonl").read_text().splitlines()]
    missing = {(q["ticker"], q["year"]) for q in questions if q["answerable"]} - pinned
    assert not missing


def _filing(tmp_path, monkeypatch, html: str, text_sha256: str) -> dict:
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    (tmp_path / "AAPL_2025.html").write_text(html)
    return {"ticker": "AAPL", "fiscal_year": 2025, "url": "unused", "text_sha256": text_sha256}


def test_changed_text_is_rejected(tmp_path, monkeypatch):
    filing = _filing(tmp_path, monkeypatch, "<p>Net sales | 391,035</p>", "0" * 64)
    with pytest.raises(SystemExit):
        ingest.filing_text(filing)


def test_random_script_tags_do_not_change_the_hash(tmp_path, monkeypatch):
    expected = ingest.text_hash("Net sales | 391,035")
    html = '<p>Net sales | 391,035</p><script src="/random/path/123"></script>'
    assert ingest.filing_text(_filing(tmp_path, monkeypatch, html, expected)) == "Net sales | 391,035"
