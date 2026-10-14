"""
chunking.py - turns a raw 10-K web page (HTML) into small pieces of text ("chunks").

Why chunk? A 10-K is ~100 pages. We can't send all of it to the LLM for every question,
so we cut it into ~400-word pieces, find the few pieces relevant to the question, and
send only those.

The pipeline is:  HTML  --html_to_text-->  clean text  --split_sections-->  sections  --chunk_text-->  chunks

This "fixed 400-word windows" approach is a BASELINE. Trying other chunk sizes or smarter
splitting and measuring the difference with the eval is exactly what makes this project
stand out.
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern 10-Ks are "inline XBRL": an HTML page that starts with an XML header. BeautifulSoup
# sees that header and warns that we might want an XML parser. We don't: the content is
# still a web page, and the HTML parser reads it correctly. So we silence that warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# A 10-K is organised into "Items": Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A)...
# This regex spots a line that starts a new Item, e.g. "Item 1A. Risk Factors" or "ITEM 7 - ...".
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)[\.\s:\-—–]", re.IGNORECASE)

# Marker lines placed around each table by html_to_text (characters that never appear in filings).
TABLE_START = "⟦TABLE⟧"
TABLE_END = "⟦/TABLE⟧"


@dataclass
class Chunk:
    section: str | None  # e.g. "Item 7", or None for text before the first Item (cover page)
    chunk_index: int  # 0, 1, 2, ... in order through the document
    text: str


def html_to_text(html: str) -> str:
    """Strip HTML tags and return readable plain text, one line per paragraph/table row."""
    soup = BeautifulSoup(html, "lxml")

    # Remove things that aren't real content.
    for tag in soup(["script", "style"]):
        tag.decompose()
    # SEC "inline XBRL" filings hide machine-readable metadata in <ix:header>; drop it.
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()

    # Tables are where the numbers live. If we just took the text, a number like "383,285"
    # would get separated from its label "Net sales". So we rewrite each table row as
    # "Net sales | 383,285" to keep labels and numbers together.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            # Filings put "$" and "%" in their own cells; drop those noise cells.
            cells = [c for c in cells if c and c not in {"$", "%", ")"}]
            if cells:
                rows.append(" | ".join(cells))
        # Wrap the rows in marker lines so the table-aware chunker knows where a table starts/ends.
        # The other chunkers simply drop these marker lines.
        table.replace_with(f"\n{TABLE_START}\n" + "\n".join(rows) + f"\n{TABLE_END}\n")

    # Get the text, collapse repeated spaces, drop empty lines.
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _strip_markers(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if ln not in (TABLE_START, TABLE_END))


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split the document at each "Item X." heading. Returns [(section_name, section_text), ...]."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]  # start with the cover page
    for line in text.splitlines():
        m = ITEM_RE.match(line)
        # Headings are short; a long line starting with "Item" is probably a sentence, not a heading.
        if m and len(line) < 200:
            sections.append((f"Item {m.group(1).upper()}", [line]))
        else:
            sections[-1][1].append(line)  # add the line to the current section
    return [(name, "\n".join(lines)) for name, lines in sections if lines]


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[Chunk]:
    """Cut each section into windows of `size` words.

    Consecutive chunks share `overlap` words, so a sentence that falls on a boundary
    still appears whole in at least one chunk. A chunk never spans two sections.
    """
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
                break  # reached the end of this section
            start += size - overlap  # step forward, keeping `overlap` words
    return chunks


# ---------------------------------------------------------------------------------------------
# Token-based chunking (optionally table-aware)
#
# Why tokens instead of words: the embedding model and the reranker only read the first 512
# TOKENS of a chunk. Financial tables are token-heavy ("1,234.5" is several tokens), so a
# 400-word chunk is often 550-900 tokens and its tail was invisible to them. Sizing chunks
# in tokens guarantees the models see the whole chunk.
#
# Table-aware mode additionally:
#   - keeps a table in one chunk whenever it fits (instead of cutting it at a random row)
#   - when a table is too big, splits it by rows and repeats the table's caption (the
#     sentence just before it) and header row at the top of every piece, so a piece like
#     "Derivatives CVA | 2 | (14)" still says which years/units the columns are.
# ---------------------------------------------------------------------------------------------

CountFn = Callable[[str], int]


def _blocks(body: str) -> list[tuple[str, str | None, list[str]]]:
    """Group a section's lines into blocks: ("line", None, [line]) or ("table", caption, rows)."""
    blocks: list[tuple[str, str | None, list[str]]] = []
    rows: list[str] | None = None  # not None while we're inside a table
    for line in body.splitlines():
        if line == TABLE_START:
            rows = []
        elif line == TABLE_END:
            if rows:
                # The caption is the prose line right before the table, e.g. "The following table
                # presents net sales by category". Only used if it's short.
                prev = blocks[-1] if blocks and blocks[-1][0] == "line" else None
                caption = prev[2][0] if prev and len(prev[2][0].split()) <= 60 else None
                blocks.append(("table", caption, rows))
            rows = None
        elif rows is not None:
            rows.append(line)
        else:
            blocks.append(("line", None, [line]))
    if rows:  # a table whose end marker landed in the next section
        blocks.append(("table", None, rows))
    return blocks


def _split_long(line: str, max_tokens: int, count: CountFn) -> list[str]:
    """Split a single line that is longer than max_tokens into word windows that fit."""
    n = count(line)
    if n <= max_tokens:
        return [line]
    words = line.split()
    per_window = max(1, int(len(words) * max_tokens / n * 0.9))  # 0.9 = safety margin
    return [" ".join(words[i : i + per_window]) for i in range(0, len(words), per_window)]


def chunk_by_tokens(
    text: str, max_tokens: int, overlap_tokens: int, count: CountFn, tables: bool = True
) -> list[Chunk]:
    """Pack whole lines into chunks of at most `max_tokens` (as measured by `count`).

    Lines are never cut unless a single line is itself too long. Consecutive prose chunks
    share about `overlap_tokens` of lines. With tables=True, tables are handled as described
    above; with tables=False, table rows are treated like ordinary lines.
    """
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in _pack_section(body, max_tokens, overlap_tokens, count, tables):
            chunks.append(Chunk(section, len(chunks), piece))
    return chunks


def _pack_section(
    body: str, max_tokens: int, overlap_tokens: int, count: CountFn, tables: bool
) -> list[str]:
    """Do the packing for ONE section and return the chunk texts (see chunk_by_tokens)."""
    out: list[str] = []  # finished chunk texts
    cur: list[str] = []  # lines in the chunk being built
    cur_tokens = 0

    def emit() -> None:
        if cur:
            out.append("\n".join(cur))

    def start_new(keep_overlap: bool) -> None:
        """Close the current chunk and start a new one, optionally seeded with its last lines."""
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
                if cur_tokens + t > max_tokens:  # overlap + piece still too big: drop overlap
                    start_new(keep_overlap=False)
            cur.append(piece)
            cur_tokens += t

    for kind, caption, lines in _blocks(body):
        if kind == "line" or not tables:
            for ln in lines:
                add_line(ln)
            continue

        # --- a table ---
        table_tokens = sum(count(r) for r in lines)
        if cur_tokens + table_tokens <= max_tokens:
            cur.extend(lines)  # fits in the current chunk: keep it whole, right after its caption
            cur_tokens += table_tokens
            continue

        start_new(keep_overlap=False)
        prefix = [caption] if caption else []
        if sum(count(p) for p in prefix) + table_tokens <= max_tokens:
            for ln in prefix + lines:  # fits in a fresh chunk: keep it whole there
                add_line(ln)
            continue

        # Too big for one chunk: split by rows, repeating caption + header row on each piece.
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
        # Leave the last table piece open; following prose continues in it if it fits.

    emit()
    return out
