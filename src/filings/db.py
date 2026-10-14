"""Postgres schema and connections. One table holds every chunk with its embedding and tsvector."""

import psycopg
from pgvector.psycopg import register_vector

from filings.config import settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    fiscal_year INT  NOT NULL,
    section     TEXT,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({settings.embed_dim}),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (ticker, fiscal_year, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


if __name__ == "__main__":
    init_db()
    print("schema ready")
