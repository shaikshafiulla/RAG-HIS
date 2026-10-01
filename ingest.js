import fs from 'fs';
import { pool, embed, toVector } from './lib.js';

// Step 1: read the tariff table (CSV here; in the hospital this comes from Oracle)
// - split on \r?\n so files saved on Windows/Excel also work
// - skip blank lines
// - organization = text before the first comma, rate = text after the last comma,
//   service = everything in between (so a service name may contain a comma)
const lines = fs.readFileSync('./data/tariffs.csv', 'utf8')
  .split(/\r?\n/)
  .slice(1)
  .filter((line) => line.trim() !== '');
const rows = lines.map((line) => {
  const first = line.indexOf(',');
  const last = line.lastIndexOf(',');
  return {
    organization: line.slice(0, first).trim(),
    service: line.slice(first + 1, last).trim().replace(/^"|"$/g, ''),
    rate: Number(line.slice(last + 1).trim()),
  };
});

// Stop early with a clear message if any row is broken
const bad = rows.find((r) => !r.organization || !r.service || Number.isNaN(r.rate));
if (bad) throw new Error(`Bad CSV row: ${JSON.stringify(bad)}`);

// Step 2: chunking - one row becomes one chunk
const chunks = rows.map((r) => ({
  ...r,
  content: `Organization: ${r.organization} | Service: ${r.service} | Rate: Rs ${r.rate}`,
}));

// Step 3: embeddings - convert every chunk into a vector
// nomic-embed-text works best when documents start with "search_document: "
// (questions get "search_query: " in server.js). The prefix is only for embedding;
// we store the plain content.
console.log(`Embedding ${chunks.length} chunks...`);
const vectors = await embed(chunks.map((c) => `search_document: ${c.content}`));

// Step 4: store chunks + vectors in PostgreSQL (pgvector)
await pool.query('TRUNCATE tariff_chunks');
for (let i = 0; i < chunks.length; i++) {
  const c = chunks[i];
  await pool.query(
    'INSERT INTO tariff_chunks (org_name, service, rate, content, embedding) VALUES ($1, $2, $3, $4, $5)',
    [c.organization, c.service, c.rate, c.content, toVector(vectors[i])]
  );
}

console.log('Done. Chunks stored in PostgreSQL.');
await pool.end();
