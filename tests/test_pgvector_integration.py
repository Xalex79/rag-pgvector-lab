"""Integration test against a real PostgreSQL + pgvector. Skipped unless RAG_TEST_DSN is set.

docker compose up -d
RAG_TEST_DSN=postgresql://rag:rag@localhost:5432/rag pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_lab.chunking import load_corpus
from rag_lab.embeddings import HashingEmbedder
from rag_lab.store import MemoryStore, PgVectorStore

DSN = os.environ.get("RAG_TEST_DSN")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="RAG_TEST_DSN not set")
def test_pgvector_matches_in_memory_search():
    emb = HashingEmbedder()
    chunks = load_corpus(ROOT / "corpus")
    vectors = emb.embed_documents([c.text for c in chunks])

    pg = PgVectorStore(DSN, emb.dim)
    pg.reset()
    pg.add(chunks, vectors)
    mem = MemoryStore()
    mem.add(chunks, vectors)

    assert pg.count() == len(chunks)
    query = emb.embed_query("how often replace the silicone sealing rubber")
    assert pg.search(query, 1)[0].chunk.id == mem.search(query, 1)[0].chunk.id
