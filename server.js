import express from 'express';
import { pool, embed, chat, toVector } from './lib.js';

const app = express();
app.use(express.json());
app.use(express.static('public'));

const STOP_WORDS = new Set(['what', 'is', 'the', 'rate', 'for', 'of', 'a', 'an', 'how', 'much',
  'price', 'cost', 'tell', 'me', 'in', 'and', 'or', 'to', 'charges', 'charge', 'does', 'are']);

// Vector search: rows with similar meaning
// (questions get the "search_query: " prefix that nomic-embed-text expects)
async function vectorSearch(question) {
  const [qVector] = await embed([`search_query: ${question}`]);
  const { rows } = await pool.query(
    'SELECT id, content FROM tariff_chunks ORDER BY embedding <=> $1 LIMIT 5',
    [toVector(qVector)]
  );
  return rows;
}

// Keyword search: rows containing the exact words (e.g. "Star", "CBP", "ECG")
async function keywordSearch(question) {
  const words = question.toLowerCase().match(/[a-z0-9]+/g) || [];
  const keywords = words.filter((w) => !STOP_WORDS.has(w));
  if (keywords.length === 0) return [];
  const { rows } = await pool.query(
    `SELECT id, content FROM tariff_chunks
     WHERE tsv @@ to_tsquery('simple', $1)
     ORDER BY ts_rank(tsv, to_tsquery('simple', $1)) DESC
     LIMIT 5`,
    [keywords.join(' | ')]
  );
  return rows;
}

// Combine both lists with Reciprocal Rank Fusion: rows ranked high in both lists come first
// score = sum of 1 / (60 + rank), where rank starts at 1 (the standard RRF formula)
function combine(vectorRows, keywordRows) {
  const scores = new Map();
  [vectorRows, keywordRows].forEach((list) => {
    list.forEach((row, index) => {
      const rank = index + 1;
      const prev = scores.get(row.id) || { content: row.content, score: 0 };
      prev.score += 1 / (60 + rank);
      scores.set(row.id, prev);
    });
  });
  return [...scores.values()].sort((a, b) => b.score - a.score).slice(0, 5);
}

app.post('/api/ask', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: 'Question is required' });

    const [vectorRows, keywordRows] = await Promise.all([
      vectorSearch(question),
      keywordSearch(question),
    ]);
    const topRows = combine(vectorRows, keywordRows);

    const prompt = `Answer the question using only the rate data below.
If the answer is not in the data, reply "Rate not found".

Data:
${topRows.map((r) => r.content).join('\n')}

Question: ${question}`;

    const answer = await chat(prompt);
    res.json({ answer, sources: topRows.map((r) => r.content) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log('Open http://localhost:3000'));
