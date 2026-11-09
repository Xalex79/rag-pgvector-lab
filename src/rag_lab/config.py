"""Settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Approximate: check the official pricing page before trusting the cost."""

    input: float = float(_env("RAG_PRICE_INPUT", "4.0"))
    output: float = float(_env("RAG_PRICE_OUTPUT", "20.0"))
    cache_write: float = float(_env("RAG_PRICE_CACHE_WRITE", "5.0"))
    cache_read: float = float(_env("RAG_PRICE_CACHE_READ", "0.2"))


@dataclass(frozen=True)
class Settings:
    dsn: str = _env("RAG_DSN", "postgresql://rag:rag@localhost:5432/rag")
    embed_base_url: str = _env("RAG_EMBED_BASE_URL", "http://localhost:1234/v1")
    embed_model: str = _env("RAG_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
    embed_dim: int = int(_env("RAG_EMBED_DIM", "768"))
    # nomic-embed-text expects task prefixes; set both to "" for models that don't use them.
    embed_doc_prefix: str = _env("RAG_EMBED_DOC_PREFIX", "search_document: ")
    embed_query_prefix: str = _env("RAG_EMBED_QUERY_PREFIX", "search_query: ")
    model: str = _env("RAG_MODEL", "claude-opus-5-5")
    effort: str = _env("RAG_EFFORT", "medium")
    max_tokens: int = int(_env("RAG_MAX_TOKENS", "2048"))
    top_k: int = int(_env("RAG_TOP_K", "5"))
    max_steps: int = int(_env("RAG_MAX_STEPS", "4"))
    chunk_chars: int = int(_env("RAG_CHUNK_CHARS", "800"))
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "150"))
    prices: Prices = field(default_factory=Prices)
