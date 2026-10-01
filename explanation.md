# Explanation: The Building Blocks

Simple explanations of each piece used in this project, and where it appears in the code.

1. [nomic-embed-text: the embedding model](#1-nomic-embed-text-the-embedding-model)
2. [How vector search works](#2-how-vector-search-works)
3. [HNSW index: fast vector search](#3-hnsw-index-fast-vector-search)
4. [GIN index: fast keyword search](#4-gin-index-fast-keyword-search)
5. [llama3.2: the answer model](#5-llama32-the-answer-model)

---

## 1. nomic-embed-text: the embedding model

### What is an embedding?

An **embedding** is a list of numbers that represents the **meaning** of a piece of text.
Texts with similar meanings get similar numbers.

```
"MRI Brain Plain"     → [0.021, -0.113, 0.087, ... 768 numbers]
"brain scan"          → [0.019, -0.108, 0.091, ...]   ← close: similar meaning
"General Ward Bed"    → [-0.204, 0.055, -0.012, ...]  ← far: different meaning
```

An **embedding model** is a neural network trained to produce these numbers.

### What is nomic-embed-text?

| | |
|---|---|
| Made by | Nomic AI |
| Size | ~137 million parameters (~274 MB download in Ollama) |
| Output | **768 numbers** per text (that's why our column is `vector(768)`) |
| Max input | up to 8192 tokens (we send one short row, so this is never an issue) |
| License | Open source (Apache 2.0), free, runs locally |
| Runs with | Ollama (`ollama pull nomic-embed-text`) |

**Task prefixes.** This model was trained to see a short label in front of the text:

```
search_document: Organization: Star Health | Service: MRI Brain Plain | Rate: Rs 6500   ← when storing
search_query: What is the brain MRI rate for Star Health?                              ← when asking
```

Using the prefixes gives better matches. In this project, `ingest.js` adds `search_document:` and
`server.js` adds `search_query:`.

**Why we chose it:** it's small, fast on a laptop CPU, free, works offline and has good quality for its size.

**Golden rule:** use the **same model** for documents and questions. Vectors from different
models live in different "spaces" and can't be compared. If you switch models, re-embed
everything and change `vector(768)` to the new model's dimension.

### Alternatives

**Free, local (via Ollama)**

| Model | Dimensions | Notes |
|---|---|---|
| `nomic-embed-text` | 768 | Our choice. Good balance of size and quality |
| `all-minilm` | 384 | Very small and fast. Lower quality |
| `mxbai-embed-large` | 1024 | Larger, often more accurate. Slower |
| `snowflake-arctic-embed` | 384–1024 (several sizes) | Strong for retrieval |
| `bge-m3` | 1024 | Multilingual (useful for Hindi, Telugu and other languages) |

**Paid, cloud APIs**

| Provider | Example model | Dimensions | Notes |
|---|---|---|---|
| OpenAI | `text-embedding-3-small` / `-large` | 1536 / 3072 | Very popular. Needs an API key |
| Cohere | `embed-english-v3.0` / `embed-multilingual-v3.0` | 1024 | Strong for search |
| Voyage AI | `voyage-3` family | 1024 (varies) | Recommended by Anthropic for use with Claude |
| Google | Gemini embedding models | 768–3072 | Part of Google Cloud / Gemini API |

**How to choose:** local and free for learning or private data (like hospital data). Cloud APIs
when you need top quality and don't mind cost and sending data out. Multilingual models when
users ask in more than one language.

---

## 2. How vector search works

### Step by step

```
1. SETUP:  every row → embedding → stored in the "embedding" column
2. QUERY:  question  → embedding (same model)
3. COMPARE: measure the distance from the question vector to every row vector
4. RETURN: the rows with the smallest distance (most similar meaning)
```

### Picture it in 2D

Real vectors have 768 dimensions, but the idea is the same with 2:

```
         ▲
         │      ● MRI Brain (Star)
         │    ● MRI Brain (HDFC)
         │       ★ question: "brain scan for Star Health"
         │
         │                          ● ICU Bed
         │                       ● General Ward Bed
         └──────────────────────────────────────►
```

The ★ question lands near the MRI rows, so those come back first.

### How "distance" is measured: cosine

**Cosine similarity** looks at the **angle** between two vectors, not their length.

```
cosine similarity = (A · B) / (|A| × |B|)

  1  → same direction    (same meaning)
  0  → 90° apart         (unrelated)
 -1  → opposite
```

pgvector's `<=>` operator returns **cosine distance = 1 − cosine similarity**, so **smaller = more similar**.
That's why our query sorts ascending:

```sql
SELECT id, content FROM tariff_chunks
ORDER BY embedding <=> $1     -- $1 = question vector
LIMIT 5;
```

Other pgvector operators:

| Operator | Measures | Used when |
|---|---|---|
| `<=>` | cosine distance | Text embeddings (most common, our choice) |
| `<->` | Euclidean (L2) distance | Straight-line distance between points |
| `<#>` | negative inner product | Vectors that are already normalized (fast) |

### Exact vs approximate

- **Exact search:** compare the question with **every** row. It's always correct, but slow for millions of rows.
- **Approximate search (ANN):** use an index such as **HNSW** to check only a small set of likely
  candidates. Very fast, and almost always finds the same results.

### Strengths and weaknesses

- ✅ Understands synonyms and paraphrases: "chest x-ray" ≈ "X-Ray Chest PA View".
- ❌ Weak with exact codes and names: it can rank "HDFC Ergo" rows near "Star Health" rows
  because they look alike. That's why we **also** run keyword search and combine the two with RRF.

---

## 3. HNSW index: fast vector search

**HNSW = Hierarchical Navigable Small World.** It's a **graph** index for fast approximate
nearest-neighbor search.

### The idea: an airport network

To fly from a small town to another small town far away, you don't visit every city.
You go **small airport → big hub → big hub → small airport**.

HNSW builds the same thing out of vectors:

```
Layer 2 (few points, long jumps)     ●───────────────●
                                     │               │
Layer 1 (more points)            ●───●─────●─────●───●
                                 │   │     │     │   │
Layer 0 (all points, short hops) ●─●─●─●─●─●─●─●─●─●─●─●
```

**Search:**
1. Start at the top layer and jump toward the question vector.
2. Drop down a layer and refine with shorter hops.
3. At the bottom layer, look at the nearby neighbors and return the closest ones.

You check a few hundred points instead of millions.

### In this project

```sql
CREATE INDEX idx_embedding ON tariff_chunks USING hnsw (embedding vector_cosine_ops);
```

- `hnsw`: the index type
- `vector_cosine_ops`: build it for **cosine** distance, to match our `<=>` query

Tuning settings (the defaults are fine for learning):

| Setting | Default | Meaning |
|---|---|---|
| `m` | 16 | Connections per point. More = better accuracy, more memory |
| `ef_construction` | 64 | Effort while building. More = better graph, slower to build |
| `hnsw.ef_search` | 40 | Effort while searching. More = better accuracy, slower queries |

**Alternative: IVFFlat.** It groups vectors into clusters and searches only the nearest clusters.
It builds faster and uses less memory, but it's usually less accurate than HNSW and must be built after the data is loaded.

**Note:** with only 50 rows, Postgres may simply scan every row, because that's already instant.
The index matters when you have thousands or millions of rows.

---

## 4. GIN index: fast keyword search

**GIN = Generalized Inverted Index.** It works like the **index at the back of a book**:
for each word, it lists the rows that contain it.

### Step 1: tsvector (the words in each row)

Postgres turns each chunk into a **tsvector**, a sorted list of words with their positions:

```
content: "Organization: Star Health | Service: MRI Brain Plain | Rate: Rs 6500"
tsv:     '6500':10 'brain':6 'health':3 'mri':5 'organization':1 'plain':7 'rate':8 'rs':9 'service':4 'star':2
```

In `schema.sql` this column is built automatically:

```sql
tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
```

`'simple'` means: lowercase the words and keep them as they are (no stemming, no language rules),
which suits names and codes like "CBP" and "HDFC".

### Step 2: the GIN index (word → rows)

```
'brain'  → the 5 MRI Brain rows (one per organization)
'cbp'    → the 5 CBP rows
'star'   → the 10 Star Health rows
'hdfc'   → the 10 HDFC Ergo rows
...
```

```sql
CREATE INDEX idx_tsv ON tariff_chunks USING gin (tsv);
```

### Step 3: the query

```sql
WHERE tsv @@ to_tsquery('simple', 'cbp | cghs')   -- @@ = "matches"; | = OR
ORDER BY ts_rank(tsv, to_tsquery('simple', 'cbp | cghs')) DESC
```

Postgres looks up `cbp` and `cghs` in the index, gets their row lists, and ranks rows that match
**more** words higher.

### HNSW vs GIN

| | HNSW | GIN |
|---|---|---|
| Used for | Vector search (meaning) | Keyword search (exact words) |
| Column | `embedding vector(768)` | `tsv tsvector` |
| Structure | Graph of nearby vectors | Word → list of rows |
| Result | Approximate nearest rows | Exact rows containing the words |
| Also used for | — | JSONB and array columns |

---

## 5. llama3.2: the answer model

### What is it?

| | |
|---|---|
| Made by | Meta (part of the Llama family) |
| Released | September 2024 |
| Text sizes | **1B** and **3B** parameters (there are also larger vision models) |
| `llama3.2` in Ollama | The **3B** model, compressed (quantized) to about **2.0 GB** |
| License | Llama Community License (free to use, with some conditions) |
| Runs with | Ollama, fully offline |

A **parameter** is one learned number inside the model. More parameters usually means smarter but
slower, and it needs more memory. 3B is small enough to run on a laptop CPU.

**Quantization** stores each number with fewer bits (about 4 instead of 16). The file gets about 4×
smaller and faster, with a small loss in quality.

### Its job in this project

llama3.2 does **not** search and does **not** know our rates. It only **reads** the 5 rows we found
and writes a sentence:

```
Prompt: "Answer only from this data ... Data: <top 5 rows> ... Question: <question>"
Output: "The brain MRI rate for Star Health is Rs 6500."
```

- `temperature: 0` → the same answer every time (no randomness). That's what you want for exact rates.
- The rule "if the answer isn't in the data, reply 'Rate not found'" → prevents made-up rates.

**A real limitation we saw:** for "heart test for Cash", retrieval correctly found the ECG row, but
the small 3B model didn't connect "heart test" with "ECG" and said "not found". Bigger models handle
this kind of reasoning better.

### Alternatives

**Free, local (via Ollama)**

| Model | Size | Notes |
|---|---|---|
| `llama3.2:1b` | 1B | Faster, weaker |
| `llama3.2` | 3B | Our choice. Good for a laptop |
| `qwen2.5` | 0.5B–72B | Strong at following instructions and producing structured output |
| `mistral` | 7B | Good quality. Needs more memory |
| `gemma2` / `gemma3` | several sizes | Google's open models |
| `phi3` / `phi4` | several sizes | Microsoft's small, efficient models |
| `llama3.1:8b` | 8B | Smarter. Needs about 5 GB+ memory |

**Paid, cloud APIs**

| Provider | Models | Notes |
|---|---|---|
| Anthropic | Claude (Haiku, Sonnet, Opus) | Strong reasoning and instruction following |
| OpenAI | GPT models | Very popular |
| Google | Gemini models | Large context windows |

**How to switch (local):** change `CHAT_MODEL` in `lib.js` and the `ollama pull` line in
`docker-compose.yml`, then run `docker compose up --build`. The rest of the RAG flow stays the same.
