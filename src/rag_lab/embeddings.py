"""Embedding back ends.

- OpenAICompatibleEmbedder: any server exposing POST /v1/embeddings (LM Studio, llama.cpp, Ollama...).
  Claude does not offer an embeddings endpoint, so embeddings come from a local open-weight model.
- HashingEmbedder: deterministic bag-of-words baseline with no model and no network. Used by the
  tests and by the offline retrieval evaluation in CI.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are at be before by can do does for from how i if in is it its my of on or "
    "should the to what when which why with".split()
)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [_stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]
        for token in tokens:
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return _normalise(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        doc_prefix: str = "",
        query_prefix: str = "",
        client: httpx.Client | None = None,
        batch_size: int = 32,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.client = client or httpx.Client(timeout=60)
        self.batch_size = batch_size

    def _post(self, inputs: list[str]) -> list[list[float]]:
        response = self.client.post(
            f"{self.base_url}/embeddings", json={"model": self.model, "input": inputs}
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [_normalise(item["embedding"]) for item in data]
        for vec in vectors:
            if len(vec) != self.dim:
                raise ValueError(f"Model returned {len(vec)} dimensions but RAG_EMBED_DIM is {self.dim}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post([self.doc_prefix + t for t in texts[i : i + self.batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([self.query_prefix + text])[0]
