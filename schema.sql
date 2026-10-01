CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tariff_chunks (
  id        SERIAL PRIMARY KEY,
  org_name  TEXT NOT NULL,
  service   TEXT NOT NULL,
  rate      NUMERIC NOT NULL,
  content   TEXT NOT NULL,
  embedding vector(768),
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_embedding ON tariff_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_tsv ON tariff_chunks USING gin (tsv);
