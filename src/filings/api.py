"""
api.py - a small web API so a frontend (or curl) can ask questions.

Start it with:  make serve      (then open http://127.0.0.1:8000/docs for a clickable test page)

Endpoints:
  GET  /health  -> {"ok": true}       quick check that the server is running
  POST /ask     -> answer + sources   body: {"question": "...", "ticker": "AAPL"}
"""

from fastapi import FastAPI
from pydantic import BaseModel

from filings.answer import answer
from filings.retrieve import Mode

app = FastAPI(title="Ask the Filings")


class AskRequest(BaseModel):
    """The JSON body the client sends. FastAPI validates it and returns a 422 error if it's wrong."""

    question: str
    ticker: str | None = None  # optional: only search this company
    year: int | None = None  # optional: only search this fiscal year
    k: int = 8  # how many chunks to retrieve
    mode: Mode = "rerank"  # "dense", "sparse", "hybrid" or "rerank" (most accurate, ~1s slower)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest):
    a = answer(req.question, k=req.k, mode=req.mode, ticker=req.ticker, year=req.year)
    # Only return the chunks Gemini actually cited, as the "sources" for the answer.
    cited = {h.id: h for h in a.hits if h.id in a.citations}
    return {
        "answer": a.answer,
        "answerable": a.answerable,
        "sources": [
            {"id": h.id, "ticker": h.ticker, "fiscal_year": h.fiscal_year, "section": h.section,
             "text": h.text[:600]}  # trim long chunks for display
            for h in cited.values()
        ],
        "latency_s": round(a.latency_s, 2),
        "usage": {"input_tokens": a.input_tokens, "output_tokens": a.output_tokens},
    }
