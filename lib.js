import pg from 'pg';

export const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL || 'postgres://rag:rag@localhost:5432/tariffs',
});

const OLLAMA = process.env.OLLAMA_URL || 'http://localhost:11434';
export const EMBED_MODEL = 'nomic-embed-text';
export const CHAT_MODEL = 'llama3.2';

// Turn text(s) into vectors using the local Ollama embedding model
export async function embed(texts) {
  const res = await fetch(`${OLLAMA}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, input: texts }),
  });
  if (!res.ok) throw new Error(`Embedding failed: ${await res.text()}`);
  const data = await res.json();
  return data.embeddings;
}

// Ask the local chat model a question
export async function chat(prompt) {
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: CHAT_MODEL,
      stream: false,
      options: { temperature: 0 }, // no creativity: we want the exact rate every time
      messages: [{ role: 'user', content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${await res.text()}`);
  const data = await res.json();
  return data.message.content;
}

// pgvector accepts vectors as a string like "[0.1,0.2,...]"
export const toVector = (arr) => `[${arr.join(',')}]`;
