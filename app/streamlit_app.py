"""Streamlit demo. Uses the same answer() path as the eval.

Run with `make demo`.
"""

import json
from pathlib import Path

import streamlit as st

from filings.answer import answer
from filings.db import connect
from filings.embed import embed_query
from filings.guardrails import GuardrailError
from filings.rerank import rerank_scores

MANIFEST = Path("filings.json")
MAX_QUESTIONS_PER_SESSION = 20  # caps API spend if the page is shared

# Taken from the eval set, all answered correctly there.
EXAMPLES = [
    "What were Microsoft's U.S. and foreign income before income taxes in fiscal year 2025?",
    "What was ExxonMobil's total oil-equivalent production per day in 2025?",
    "In which two segments does NVIDIA report its business results?",
    "Which industry standards does NVIDIA routinely assess its high-risk suppliers against?",
]

st.set_page_config(page_title="Ask the Filings", layout="centered")


@st.cache_resource(show_spinner="Loading search models (first time only)...")
def warm_up() -> list[str]:
    """Load both models up front and list the companies in the database."""
    embed_query("warm up")
    rerank_scores("warm up", ["warm up"])
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM chunks ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def filing_url(ticker: str, year: int) -> str | None:
    for f in json.loads(MANIFEST.read_text()):
        if f["ticker"] == ticker and f["fiscal_year"] == year:
            return f["url"]
    return None


tickers = warm_up()

st.title("Ask the Filings")
st.caption(
    f"Ask questions about the annual reports (10-K filings) of {len(tickers)} US companies: "
    f"{', '.join(tickers)}. Answers come only from the filings, with sources, "
    "and the system says so when the filings don't contain the answer."
)

def use_example(text: str) -> None:
    # Callbacks run before the rerun, so the text area picks up the new value.
    st.session_state.question = text


st.write("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    cols[i % 2].button(ex, key=f"ex{i}", width="stretch", on_click=use_example, args=(ex,))

with st.form("ask_form"):
    question = st.text_area("Your question", key="question", height=80,
                            placeholder="e.g. What was Microsoft's total revenue in fiscal 2025?")
    company = st.selectbox("Company", ["Auto-detect from question"] + tickers)
    ask = st.form_submit_button("Ask", type="primary")

if ask and not question.strip():
    st.info("Type a question first.")
    st.stop()

st.session_state.setdefault("asked", 0)

if ask:
    if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
        st.error(f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per session.")
        st.stop()
    ticker = None if company.startswith("Auto") else company
    try:
        with st.spinner("Searching the filings and asking Gemini..."):
            a = answer(question, ticker=ticker)
    except GuardrailError as e:
        st.error(str(e))
        st.stop()
    st.session_state.asked += 1

    if a.answerable:
        st.success(a.answer)
        if a.unsupported_numbers:
            nums = ", ".join(f"{n:,g}" for n in a.unsupported_numbers)
            st.warning(f"Check these figures: {nums} could not be found in the cited passages "
                       "(they may be calculated or rounded).")
    elif a.request_type != "filing_question":
        st.info(a.answer)
    else:
        st.warning(f"**Not found in the filings.** {a.answer}")

    st.caption(
        f"{a.latency_s:.1f}s · {a.input_tokens:,} input / {a.output_tokens:,} output tokens · "
        f"model {a.meta['model']}"
    )

    by_id = {h.id: h for h in a.hits}
    cited = [by_id[c] for c in a.citations if c in by_id]
    if cited:
        st.subheader("Sources")
        for n, h in enumerate(cited, start=1):
            label = f"[{n}] {h.ticker} · 10-K FY{h.fiscal_year} · {h.section or 'section unknown'}"
            with st.expander(label, expanded=(n == 1)):
                st.text(h.text)  # keeps table rows on separate lines
                url = filing_url(h.ticker, h.fiscal_year)
                if url:
                    st.markdown(f"[Open the full filing on SEC EDGAR]({url})")

    others = [h for h in a.hits if h.id not in a.citations]
    if others and a.request_type == "filing_question":
        with st.expander(f"Other passages retrieved but not cited ({len(others)})"):
            for h in others:
                st.markdown(f"**{h.ticker} · FY{h.fiscal_year} · {h.section or '?'}**")
                st.text(h.text[:800] + ("…" if len(h.text) > 800 else ""))

st.divider()
st.caption(
    "Hybrid search (embeddings + keywords), cross-encoder reranking, then Gemini with a JSON schema. "
    "Evaluated on 144 questions: 95% correct, 0 wrong answers, 26 of 27 unanswerable questions refused. "
    "Not financial advice."
)
