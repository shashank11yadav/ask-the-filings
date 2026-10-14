"""
rerank.py - re-sorts search results with a "cross-encoder" model for better precision.

Why:
  The embedding model (embed.py) turns the question and each chunk into vectors SEPARATELY,
  then compares them. That's fast enough to search thousands of chunks, but it never sees
  the question and the chunk side by side, so it misses details.

  A cross-encoder reads the question and ONE chunk TOGETHER and outputs a relevance score.
  It's far more accurate, but too slow to run on every chunk in the database. So we use it
  as a second stage: hybrid search grabs ~30 candidates quickly, then the cross-encoder
  re-orders just those 30 and we keep the best k.

Like the embedding model, it runs locally for free. The first run downloads it.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from filings.config import settings


@lru_cache(maxsize=1)  # load the model once and reuse it
def _model() -> CrossEncoder:
    # max_length caps how many tokens of question + chunk the model reads at once.
    return CrossEncoder(settings.rerank_model, max_length=512)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Return one relevance score per text (higher = more relevant)."""
    if not texts:
        return []
    # The model takes (question, chunk) pairs and scores each pair.
    return _model().predict([(query, t) for t in texts], batch_size=16).tolist()
