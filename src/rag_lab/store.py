"""Vector stores: PostgreSQL + pgvector (HNSW, cosine) and an in-memory store for tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class MemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0))

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        new = np.asarray(vectors, dtype=float)
        self.matrix = new if self.matrix.size == 0 else np.vstack([self.matrix, new])

    def search(self, query_vec: list[float], k: int) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.matrix @ np.asarray(query_vec, dtype=float)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class PgVectorStore:
    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def reset(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.conn.execute(
            f"""
            CREATE TABLE chunks (
                id        text PRIMARY KEY,
                source    text NOT NULL,
                section   text NOT NULL,
                path      text NOT NULL,
                content   text NOT NULL,
                embedding vector({self.dim}) NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, source, section, path, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c.id, c.source, c.section, c.path, c.text, np.asarray(v, dtype=np.float32))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def search(self, query_vec: list[float], k: int) -> list[Hit]:
        rows = self.conn.execute(
            "SELECT id, source, section, path, content, 1 - (embedding <=> %s) AS score "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (np.asarray(query_vec, dtype=np.float32), np.asarray(query_vec, dtype=np.float32), k),
        ).fetchall()
        return [Hit(Chunk(r[0], r[1], r[2], r[3], r[4]), float(r[5])) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
