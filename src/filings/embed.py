"""
embed.py - turns text into "embeddings" (vectors of numbers that capture meaning).

Two texts with similar meaning get vectors that point in similar directions, even if they
use different words ("revenue" vs. "net sales"). That's what powers "dense"/meaning search.

We use a small open-source model (bge-small) that runs locally, so embeddings are free.
The first run downloads the model (~130 MB).
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from filings.config import settings

# bge models were trained to expect this prefix on search QUERIES (not on the stored passages).
# Adding it noticeably improves retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)  # load the model once and reuse it (loading takes a few seconds)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed many chunks at once (used during ingestion).

    normalize_embeddings=True scales every vector to length 1, so cosine similarity
    becomes a simple dot product.
    """
    return _model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)


def count_tokens(text: str) -> int:
    """How many tokens the embedding model sees for `text` (used to size chunks)."""
    return len(_model().tokenizer(text, add_special_tokens=False)["input_ids"])


def embed_query(text: str) -> np.ndarray:
    """Embed a single user question (used at search time)."""
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
