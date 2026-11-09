# Plan

## 1. First real measurement

- Start PostgreSQL with `docker compose up -d` and load an embedding model in LM Studio
  (nomic-embed-text v1.5 by default).
- `rag-lab ingest`, then `rag-lab eval`.
- Fill in the results table in the README: retrieval metrics, agent metrics and cost per question.

## 2. Real documentation

- Replace or extend the sample manual with real documentation under a free licence
  (for example, part of the PostgreSQL or Kubernetes docs).
- Write 20–30 golden questions, including several that the docs cannot answer.

## 3. Experiments

- Chunk size and overlap.
- Hybrid search: pgvector plus PostgreSQL full-text search (`tsvector`), merged with reciprocal rank fusion.
- Re-ranking the top results before giving them to the agent.
- Record each experiment in the README with its numbers.

## 4. Answer quality

- Add Ragas to measure faithfulness (is every claim supported by the cited passages?) and answer relevance.
