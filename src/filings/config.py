"""
config.py - all the settings for the project in one place.

Values are read from the `.env` file (or real environment variables). If a value
isn't set, the default written here is used. Any file can do:

    from filings.config import settings
    print(settings.answer_model)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Tell pydantic-settings to also read the `.env` file and ignore unknown keys in it.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Your Gemini API key, read from GEMINI_API_KEY in `.env`.
    # (The Gemini library only looks at real environment variables, not the `.env` file,
    # so we read it here and hand it to the library ourselves in llm.py.)
    gemini_api_key: str = ""

    # Where Postgres is running (docker-compose.yml starts it with these credentials).
    database_url: str = "postgresql://filings:filings@localhost:5432/filings"

    # The SEC blocks requests without a contact User-Agent, e.g. "Jane Doe jane@example.com".
    sec_user_agent: str = ""

    # Which Gemini models to use. answer_model writes answers; judge_model grades them in evals.
    # See which models your key can use: uv run python -c "from filings.llm import _get_client; [print(m.name) for m in _get_client().models.list()]"
    answer_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.8-flash"

    # The embedding model turns text into vectors (lists of numbers) for similarity search.
    # It runs locally on your laptop, so it's free. bge-small produces 384 numbers per text.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # The reranker re-sorts the top `rerank_pool` hybrid results with a slower, more accurate model
    # (see rerank.py). Also local and free.
    # bge-reranker-base beat ms-marco-MiniLM-L-6-v2 on our eval (recall@8 0.917 vs 0.883);
    # MiniLM is trained on short web snippets and gave no gain on long 10-K chunks.
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_pool: int = 30  # 50 was slower and no better

    # How filings are cut into chunks (see chunking.py):
    #   "words"         - fixed 400-word windows (the original baseline)
    #   "tokens"        - whole lines packed into chunks of at most `max_chunk_tokens` tokens
    #   "tokens_tables" - like "tokens", but keeps tables whole / repeats table headers
    # After changing any of these, re-run `make ingest ARGS=--force` to rebuild the chunks.
    chunker: str = "words"
    chunk_tokens: int = 400  # "words" chunker: words per chunk (name kept for compatibility)
    chunk_overlap: int = 60  # "words" chunker: words shared between consecutive chunks
    # Token chunkers: the embedder and reranker read at most 512 tokens, and the reranker
    # also needs room for the question, so 400 keeps every chunk fully visible to both.
    max_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50


# A single shared instance that the rest of the code imports.
settings = Settings()
