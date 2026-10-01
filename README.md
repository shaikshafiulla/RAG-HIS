# Tariff RAG Assistant

Hospital staff ask questions like *"What is the brain MRI rate for Star Health?"* and get
answers that come only from a tariff table (organization, service, rate).

## Flowchart

```
                         ┌──────────────────────┐
                         │   data/tariffs.csv   │
                         └──────────┬───────────┘
                                    │  one row = one chunk
                                    ▼
                         ┌──────────────────────┐
                         │  embed chunks        │  nomic-embed-text
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  PostgreSQL          │  vector (HNSW index)
                         │  + pgvector          │  words  (GIN index)
                         └──────────┬───────────┘
                                    │
  ════════════════════════ SETUP ↑ │ ↓ QUERY ════════════════════════
                                    │
                         ┌──────────────────────┐
                         │  user question       │
                         └──────────┬───────────┘
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │  vector search      │       │  keyword search     │
          │  (same meaning)     │       │  (same words)       │
          └──────────┬──────────┘       └──────────┬──────────┘
                     └──────────────┬──────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  combine (RRF)       │  top 5 rows
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  llama3.2            │  "answer only from this data"
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  answer + sources    │
                         └──────────────────────┘
```

## Design

| Part | Choice | Role |
|---|---|---|
| Database | PostgreSQL + pgvector | Stores chunks, vectors and the keyword index in one place |
| Embeddings | `nomic-embed-text` (Ollama, 768 dimensions) | Turns text into vectors that capture meaning |
| LLM | `llama3.2` (Ollama, temperature 0) | Writes the answer from the retrieved rows only |

**Design decisions**
- **One row = one chunk:** each row is already one complete fact. Splitting a row loses meaning,
  and merging rows would mix up rates.
- **Hybrid search:** keyword search catches exact names and codes ("CBP", "HDFC"). Vector search
  catches synonyms ("chest x-ray" → "X-Ray Chest PA View"). Neither is enough on its own.
- **RRF to combine:** `score = Σ 1 / (60 + rank)`. It uses only positions in each list, so the two
  different score types never need to be compared. A row that ranks high in both lists wins.
- **Grounded prompt:** the LLM answers only from the retrieved rows, or replies "Rate not found".
- **Sources returned:** staff can check which rows the answer came from.

## Workflow

**Setup** (`ingest.js`, runs on every app start)
1. **Load:** read `data/tariffs.csv`.
2. **Chunk:** each row becomes
   `Organization: Star Health | Service: MRI Brain Plain | Rate: Rs 6500`.
3. **Embed:** `nomic-embed-text` turns each chunk into 768 numbers.
4. **Store:** save the text and vector in `tariff_chunks`. Postgres builds the keyword column (`tsvector`) automatically.

**Query** (`server.js`, `POST /api/ask`)
1. **Vector search:** embed the question, then take the 5 nearest rows by cosine distance (`<=>`).
2. **Keyword search:** remove stop words, then take the 5 best matches with `to_tsquery` + `ts_rank`.
3. **Combine:** merge both lists with RRF and keep the top 5 rows.
4. **Prompt:** top rows + "answer only from this data" + the question.
5. **Generate:** `llama3.2` writes the answer.
6. **Respond:** return `{ answer, sources }`.
