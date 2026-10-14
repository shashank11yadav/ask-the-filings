"""
Tests for chunking.py. Run with `make test`. Each test feeds in a tiny fake filing
and checks that the output looks the way we expect.
"""

from filings.chunking import chunk_text, html_to_text, split_sections


def test_html_to_text_flattens_tables():
    html = """<html><body><p>Item 7. Management's Discussion</p>
    <table><tr><td>Net sales</td><td>$</td><td>383,285</td></tr>
    <tr><td>Cost of sales</td><td>$</td><td>214,137</td></tr></table></body></html>"""
    text = html_to_text(html)
    assert "Net sales | 383,285" in text
    assert "Cost of sales | 214,137" in text


def test_split_sections_tags_items():
    text = "Cover page\nItem 1. Business\nWe sell phones.\nItem 1A. Risk Factors\nSupply chain risk."
    names = [name for name, _ in split_sections(text)]
    assert names == [None, "Item 1", "Item 1A"]


def test_chunks_overlap_and_respect_sections():
    body = " ".join(f"w{i}" for i in range(250))
    text = f"Item 1. Business\n{body}\nItem 2. Properties\nshort section"
    chunks = chunk_text(text, size=100, overlap=20)
    item1 = [c for c in chunks if c.section == "Item 1"]
    assert len(item1) == 3
    # last 20 words of chunk 0 are the first 20 words of chunk 1
    assert item1[0].text.split()[-20:] == item1[1].text.split()[:20]
    assert chunks[-1].section == "Item 2"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --- token-based / table-aware chunking (word count stands in for the tokenizer here) ---

from filings.chunking import TABLE_END, TABLE_START, chunk_by_tokens


def _words(s: str) -> int:
    return len(s.split())


def _table(n_rows: int) -> str:
    rows = "\n".join(f"Row{i} | {i},000 | {i},500" for i in range(n_rows))
    return f"The following table presents results:\n{TABLE_START}\n(in millions) | 2025 | 2024\n{rows}\n{TABLE_END}"


def test_token_chunks_respect_limit_and_drop_markers():
    text = "Item 7. MD&A\n" + "\n".join(f"sentence number {i} here" for i in range(100)) + "\n" + _table(5)
    chunks = chunk_by_tokens(text, max_tokens=40, overlap_tokens=8, count=_words, tables=True)
    assert all(_words(c.text) <= 40 for c in chunks)
    assert not any(TABLE_START in c.text or TABLE_END in c.text for c in chunks)


def test_small_table_kept_whole_with_caption():
    text = "Item 8. Financials\n" + "\n".join(f"filler line {i} words" for i in range(8)) + "\n" + _table(4)
    chunks = chunk_by_tokens(text, max_tokens=40, overlap_tokens=0, count=_words, tables=True)
    with_table = [c for c in chunks if "Row0 |" in c.text]
    assert len(with_table) == 1 and "Row3 |" in with_table[0].text
    assert "The following table presents results:" in with_table[0].text


def test_big_table_repeats_caption_and_header_on_every_piece():
    text = "Item 8. Financials\n" + _table(30)
    chunks = chunk_by_tokens(text, max_tokens=30, overlap_tokens=0, count=_words, tables=True)
    pieces = [c for c in chunks if "| 2025 | 2024" in c.text]
    assert len(pieces) > 2
    assert all("(in millions) | 2025 | 2024" in p.text and "following table" in p.text for p in pieces)
    # every data row appears somewhere
    assert all(any(f"Row{i} |" in p.text for p in pieces) for i in range(30))
