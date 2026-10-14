"""
streamlit_app.py - the demo web page.

Type a question about a company's 10-K, get an answer with clickable sources showing the
exact filing text it came from. It calls the same `answer()` function the eval uses,
so what you see here is exactly what the eval measured.

Run it with:  make demo      (then open http://localhost:8501)
"""

import json
from pathlib import Path

import streamlit as st

from filings.answer import answer
from filings.db import connect
from filings.embed import embed_query
from filings.rerank import rerank_scores

INDEX_DIR = Path("data/raw/index")  # saved by ingest.py: each company's 10-K download links

# Questions the system answers correctly in the eval (a mix of tables and text).
EXAMPLES = [
    "What were Microsoft's U.S. and foreign income before income taxes in fiscal year 2025?",
    "What was ExxonMobil's total oil-equivalent production per day in 2025?",
    "In which two segments does NVIDIA report its business results?",
    "Which industry standards does NVIDIA routinely assess its high-risk suppliers against?",
]

st.set_page_config(page_title="Ask the Filings", layout="centered")


@st.cache_resource(show_spinner="Loading search models (first time only)...")
def warm_up() -> list[str]:
    """Load the embedding + reranker models once, and list the companies in the database."""
    embed_query("warm up")
    rerank_scores("warm up", ["warm up"])
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM chunks ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def filing_url(ticker: str, year: int) -> str | None:
    """Link to the original 10-K on the SEC website, if ingest.py saved it."""
    path = INDEX_DIR / f"{ticker}_10k.json"
    if not path.exists():
        return None
    for f in json.loads(path.read_text()):
        if f["fiscal_year"] == year:
            return f["url"]
    return None


tickers = warm_up()

# ---------------- page header ----------------
st.title("Ask the Filings")
st.caption(
    f"Ask questions about the annual reports (10-K filings) of {len(tickers)} US companies: "
    f"{', '.join(tickers)}. Answers come only from the filings, with sources, "
    "and the system says so when the filings don't contain the answer."
)

# ---------------- inputs ----------------
def use_example(text: str) -> None:
    # Runs before the page redraws, so the question box shows the example straight away.
    st.session_state.question = text


st.write("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    cols[i % 2].button(ex, key=f"ex{i}", width="stretch", on_click=use_example, args=(ex,))

# A form groups the inputs: nothing runs until "Ask" is pressed.
with st.form("ask_form"):
    question = st.text_area("Your question", key="question", height=80,
                            placeholder="e.g. What was Microsoft's total revenue in fiscal 2025?")
    company = st.selectbox("Company", ["Auto-detect from question"] + tickers)
    ask = st.form_submit_button("Ask", type="primary")

if ask and not question.strip():
    st.info("Type a question first.")
    st.stop()

# ---------------- answer ----------------
if ask:
    if len(question) > 500:
        st.error("Please keep questions under 500 characters.")
        st.stop()
    ticker = None if company.startswith("Auto") else company
    with st.spinner("Searching the filings and asking Gemini..."):
        a = answer(question, ticker=ticker)

    if a.answerable:
        st.success(a.answer)
    else:
        # The system refused to guess: show that clearly, it's a feature, not an error.
        st.warning(f"**Not found in the filings.** {a.answer}")

    st.caption(
        f"{a.latency_s:.1f}s · {a.input_tokens:,} input / {a.output_tokens:,} output tokens · "
        f"model {a.meta['model']}"
    )

    # Sources the answer cited, in the order Gemini cited them.
    by_id = {h.id: h for h in a.hits}
    cited = [by_id[c] for c in a.citations if c in by_id]
    if cited:
        st.subheader("Sources")
        for n, h in enumerate(cited, start=1):
            label = f"[{n}] {h.ticker} · 10-K FY{h.fiscal_year} · {h.section or 'section unknown'}"
            with st.expander(label, expanded=(n == 1)):
                st.text(h.text)  # st.text keeps table rows ("label | value") on their own lines
                url = filing_url(h.ticker, h.fiscal_year)
                if url:
                    st.markdown(f"[Open the full filing on SEC EDGAR]({url})")

    # Everything retrieved, for transparency: shows what the model was given to read.
    others = [h for h in a.hits if h.id not in a.citations]
    if others:
        with st.expander(f"Other passages retrieved but not cited ({len(others)})"):
            for h in others:
                st.markdown(f"**{h.ticker} · FY{h.fiscal_year} · {h.section or '?'}**")
                st.text(h.text[:800] + ("…" if len(h.text) > 800 else ""))

st.divider()
st.caption(
    "Hybrid search (embeddings + keywords), cross-encoder reranking, then Gemini with a JSON schema. "
    "Evaluated on 121 questions: 95% correct, 0 confident wrong answers. "
    "Not financial advice."
)
