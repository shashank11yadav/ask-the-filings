"""
db.py - sets up the Postgres database and gives other files a way to connect to it.

We store every chunk of every filing in ONE table called `chunks`. Each row has:
  - which company/year/section it came from
  - the chunk's text
  - `embedding`: a vector (384 numbers) for "meaning" search      -> uses the pgvector extension
  - `tsv`: a processed word list for "keyword" search              -> uses Postgres full-text search

Run it directly to create the table:  uv run python -m filings.db
"""

import psycopg
from pgvector.psycopg import register_vector

from filings.config import settings

# The SQL that creates the table and indexes. "IF NOT EXISTS" makes it safe to run many times.
SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;          -- turn on pgvector

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,           -- auto-incrementing id; used in citations
    ticker      TEXT NOT NULL,                   -- e.g. 'AAPL'
    fiscal_year INT  NOT NULL,                   -- e.g. 2025
    section     TEXT,                            -- e.g. 'Item 1A' (Risk Factors)
    chunk_index INT  NOT NULL,                   -- position of this chunk within the filing
    text        TEXT NOT NULL,
    embedding   vector({settings.embed_dim}),    -- the meaning-vector
    -- Postgres computes this automatically from `text`: lower-cased, stemmed words for keyword search
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (ticker, fiscal_year, chunk_index)
);

-- Indexes make searches fast instead of scanning every row.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);                          -- keyword
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);  -- vector
"""


def connect() -> psycopg.Connection:
    """Open a connection. autocommit=True means each statement is saved immediately."""
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Teach psycopg how to send/receive numpy arrays as pgvector values.
    register_vector(conn)
    return conn


def init_db() -> None:
    """Create the table and indexes if they don't exist yet."""
    with connect() as conn:
        conn.execute(SCHEMA)


if __name__ == "__main__":
    init_db()
    print("schema ready")
