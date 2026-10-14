"""Local sentence embeddings (bge-small) for chunks and queries."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from filings.config import settings

# bge models expect this instruction on queries only, not on the passages being searched.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    # Normalised vectors make cosine distance in pgvector equivalent to a dot product.
    return _model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)


def embed_query(text: str) -> np.ndarray:
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True)


def count_tokens(text: str) -> int:
    """Token count as seen by the embedding model, used to size token-based chunks."""
    return len(_model().tokenizer(text, add_special_tokens=False)["input_ids"])
