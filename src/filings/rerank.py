"""Cross-encoder reranking of search candidates.

The bi-encoder in embed.py scores question and chunk separately, which is fast but coarse.
A cross-encoder reads them together and is much more accurate, so it re-sorts the top hybrid
candidates rather than searching the whole table.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from filings.config import settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model, max_length=512)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return _model().predict([(query, t) for t in texts], batch_size=16).tolist()
