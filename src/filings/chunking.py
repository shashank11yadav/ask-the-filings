"""10-K HTML to plain text, split by Item, then cut into chunks.

chunk_text is the default (fixed word windows). chunk_by_tokens sizes chunks in model tokens and
can keep tables together; it was evaluated but not adopted (see README).
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline XBRL filings start with an XML declaration, which makes BeautifulSoup suggest an XML
# parser. The documents are HTML and parse correctly as HTML.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches 10-K section headings such as "Item 1A. Risk Factors" or "ITEM 7 - MD&A".
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)[\.\s:\-—–]", re.IGNORECASE)

# Lines that html_to_text puts around each table, so the table-aware chunker can find them.
TABLE_START = "⟦TABLE⟧"
TABLE_END = "⟦/TABLE⟧"


@dataclass
class Chunk:
    section: str | None  # None for the cover page, before the first Item
    chunk_index: int
    text: str


def html_to_text(html: str) -> str:
    """Plain text with one line per paragraph and one "label | value | ..." line per table row."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header"]):  # XBRL metadata, not document text
        tag.decompose()

    # Flatten each table row so values stay next to their row label after extraction.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            # "$" and "%" usually sit in their own cells.
            cells = [c for c in cells if c and c not in {"$", "%", ")"}]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(f"\n{TABLE_START}\n" + "\n".join(rows) + f"\n{TABLE_END}\n")

    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _strip_markers(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if ln not in (TABLE_START, TABLE_END))


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split at each Item heading. Returns [(section_name, section_text), ...]."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        m = ITEM_RE.match(line)
        if m and len(line) < 200:  # longer lines are sentences that happen to start with "Item"
            sections.append((f"Item {m.group(1).upper()}", [line]))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines)) for name, lines in sections if lines]


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[Chunk]:
    """Windows of `size` words overlapping by `overlap`, never crossing a section boundary."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for section, body in split_sections(_strip_markers(text)):
        words = body.split()
        start = 0
        while start < len(words):
            window = words[start : start + size]
            chunks.append(Chunk(section, len(chunks), " ".join(window)))
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


# Token-based chunking. The embedder and reranker truncate at 512 tokens, and number-heavy
# tables make about half of the 400-word chunks longer than that.

CountFn = Callable[[str], int]


def _blocks(body: str) -> list[tuple[str, str | None, list[str]]]:
    """Group lines into ("line", None, [line]) and ("table", caption, rows) blocks."""
    blocks: list[tuple[str, str | None, list[str]]] = []
    rows: list[str] | None = None
    for line in body.splitlines():
        if line == TABLE_START:
            rows = []
        elif line == TABLE_END:
            if rows:
                # Use the preceding line as the caption ("The following table presents...").
                prev = blocks[-1] if blocks and blocks[-1][0] == "line" else None
                caption = prev[2][0] if prev and len(prev[2][0].split()) <= 60 else None
                blocks.append(("table", caption, rows))
            rows = None
        elif rows is not None:
            rows.append(line)
        else:
            blocks.append(("line", None, [line]))
    if rows:  # table end marker fell into the next section
        blocks.append(("table", None, rows))
    return blocks


def _split_long(line: str, max_tokens: int, count: CountFn) -> list[str]:
    """Split one over-long line into word windows that fit."""
    n = count(line)
    if n <= max_tokens:
        return [line]
    words = line.split()
    per_window = max(1, int(len(words) * max_tokens / n * 0.9))
    return [" ".join(words[i : i + per_window]) for i in range(0, len(words), per_window)]


def chunk_by_tokens(
    text: str, max_tokens: int, overlap_tokens: int, count: CountFn, tables: bool = True
) -> list[Chunk]:
    """Pack whole lines into chunks of at most `max_tokens`.

    With tables=True, a table that fits is kept in one chunk, and a larger one is split by rows
    with its caption and header row repeated on every piece. With tables=False, table rows are
    packed like any other line.
    """
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in _pack_section(body, max_tokens, overlap_tokens, count, tables):
            chunks.append(Chunk(section, len(chunks), piece))
    return chunks


def _pack_section(
    body: str, max_tokens: int, overlap_tokens: int, count: CountFn, tables: bool
) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def emit() -> None:
        if cur:
            out.append("\n".join(cur))

    def start_new(keep_overlap: bool) -> None:
        nonlocal cur, cur_tokens
        emit()
        tail: list[str] = []
        if keep_overlap:
            total = 0
            for ln in reversed(cur):
                total += count(ln)
                if total > overlap_tokens:
                    break
                tail.insert(0, ln)
        cur, cur_tokens = tail, sum(count(ln) for ln in tail)

    def add_line(ln: str) -> None:
        nonlocal cur_tokens
        for piece in _split_long(ln, max_tokens, count):
            t = count(piece)
            if cur and cur_tokens + t > max_tokens:
                start_new(keep_overlap=True)
                if cur_tokens + t > max_tokens:
                    start_new(keep_overlap=False)
            cur.append(piece)
            cur_tokens += t

    for kind, caption, lines in _blocks(body):
        if kind == "line" or not tables:
            for ln in lines:
                add_line(ln)
            continue

        table_tokens = sum(count(r) for r in lines)
        if cur_tokens + table_tokens <= max_tokens:
            cur.extend(lines)
            cur_tokens += table_tokens
            continue

        start_new(keep_overlap=False)
        prefix = [caption] if caption else []
        if sum(count(p) for p in prefix) + table_tokens <= max_tokens:
            for ln in prefix + lines:
                add_line(ln)
            continue

        header = prefix + lines[:1]
        if sum(count(h) for h in header) > max_tokens // 2:
            header = lines[:1] if count(lines[0]) <= max_tokens // 2 else []
        header_tokens = sum(count(h) for h in header)
        cur, cur_tokens = list(header), header_tokens
        for row in lines[1:]:
            for piece in _split_long(row, max_tokens - header_tokens, count):
                t = count(piece)
                if cur_tokens + t > max_tokens:
                    emit()
                    cur, cur_tokens = list(header), header_tokens
                cur.append(piece)
                cur_tokens += t

    emit()
    return out
