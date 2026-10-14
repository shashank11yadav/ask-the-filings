"""Settings, read from the environment or .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # google-genai only reads real environment variables, so llm.py passes this in explicitly.
    gemini_api_key: str = ""
    database_url: str = "postgresql://filings:filings@localhost:5432/filings"
    sec_user_agent: str = ""  # EDGAR rejects requests without a contact User-Agent

    answer_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.8-flash"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # bge-reranker-base beat ms-marco-MiniLM-L-6-v2 on the eval (recall@8 0.917 vs 0.883).
    # A pool of 50 was slower and no better than 30.
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_pool: int = 30

    # "words": fixed 400-word windows. "tokens" / "tokens_tables": see chunking.chunk_by_tokens.
    # Re-run `make ingest ARGS=--force` after changing any of these.
    chunker: str = "words"
    chunk_tokens: int = 400  # words per chunk for the "words" chunker
    chunk_overlap: int = 60
    # The embedder and reranker read at most 512 tokens, and the reranker also needs room for
    # the question, so token-based chunks stay at 400.
    max_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50


settings = Settings()
