"""Unit tests: no network, no database, no API cost. The agent runs against a fake client."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from rag_lab.agent import Agent
from rag_lab.chunking import Chunk, chunk_markdown, load_corpus
from rag_lab.config import Prices
from rag_lab.embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from rag_lab.evaluate import GoldenItem, agent_metrics, load_golden, retrieval_metrics
from rag_lab.store import Hit, MemoryStore

ROOT = Path(__file__).resolve().parents[1]


# --- chunking -------------------------------------------------------------------------------


def test_chunks_keep_file_and_heading_path():
    md = "# Manual\n\nIntro text.\n\n## Daily checks\n\nCheck the air.\n\n### Gauge\n\nIt reads 6 bar."
    chunks = chunk_markdown("guide.md", md)
    assert [c.section for c in chunks] == ["Manual", "Daily checks", "Gauge"]
    assert chunks[2].path == "Manual > Daily checks > Gauge"
    assert chunks[2].id == "guide.md#Gauge#0"
    assert chunks[2].text.startswith("Manual > Daily checks > Gauge")


def test_long_sections_are_split_with_overlap():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_markdown("long.md", f"# Long\n\n{words}", size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + len("Long\n\n") for c in chunks)
    first_tail = chunks[0].text.split()[-3:]
    assert any(w in chunks[1].text.split() for w in first_tail)  # overlap carries words over


# --- embeddings and store -------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed_query("sealing temperature"), emb.embed_query("sealing temperature")
    assert a == b
    assert sum(v * v for v in a) == pytest.approx(1.0)


def test_openai_compatible_embedder_parses_response_in_order():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert body["input"] == ["doc: a", "doc: b"]
        data = [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]}]
        return httpx.Response(200, json={"data": data})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OpenAICompatibleEmbedder("http://lm/v1", "m", 2, doc_prefix="doc: ", client=client)
    assert emb.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_memory_store_ranks_by_similarity():
    emb = HashingEmbedder()
    chunks = load_corpus(ROOT / "corpus")
    store = MemoryStore()
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    top = store.search(emb.embed_query("alarm E35 film break thread the film"), 3)
    assert top[0].chunk.section == "Alarm E35: film break"
    assert top[0].score >= top[1].score >= top[2].score


# --- agent with a fake Claude client ----------------------------------------------------------


def _usage(i=100, o=50, cw=0, cr=0):
    return SimpleNamespace(
        input_tokens=i, output_tokens=o, cache_creation_input_tokens=cw, cache_read_input_tokens=cr
    )


def _tool_call(query):
    block = SimpleNamespace(type="tool_use", id="tu_1", name="search_docs", input={"query": query})
    return SimpleNamespace(stop_reason="tool_use", content=[block], usage=_usage(cw=1000))


def _final(payload):
    block = SimpleNamespace(type="text", text=json.dumps(payload))
    return SimpleNamespace(stop_reason="end_turn", content=[block], usage=_usage(cr=1000))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


CHUNK = Chunk(
    "02-maintenance.md#Lubrication#0",
    "02-maintenance.md",
    "Lubrication",
    "Maintenance > Lubrication",
    "Use NSF H1 grease every 500 hours.",
)


def test_agent_calls_search_tool_then_returns_structured_answer():
    answer = {
        "answer": "NSF H1 grease every 500 h.",
        "citations": [CHUNK.id],
        "answerable": True,
        "confidence": "high",
    }
    client = FakeClient([_tool_call("grease linear guides"), _final(answer)])
    agent = Agent(client, lambda q: [Hit(CHUNK, 0.9)], model="test-model")

    result = agent.ask("What grease do the guides need?")

    assert result.answer == answer["answer"] and result.answerable
    assert result.searches == ["grease linear guides"]
    assert result.retrieved_ids == [CHUNK.id]
    sent = client.calls[0]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["tools"][0]["name"] == "search_docs"
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and CHUNK.id in tool_result["content"]


def test_agent_accumulates_tokens_and_cost_including_cache():
    answer = {"answer": "x", "citations": [], "answerable": False, "confidence": "low"}
    client = FakeClient([_tool_call("q"), _final(answer)])
    prices = Prices(input=1.0, output=10.0, cache_write=2.0, cache_read=0.5)
    result = Agent(client, lambda q: [], model="m", prices=prices).ask("q")

    u = result.usage
    assert (u.calls, u.input_tokens, u.output_tokens) == (2, 200, 100)
    assert (u.cache_write_tokens, u.cache_read_tokens) == (1000, 1000)
    expected = (200 * 1.0 + 100 * 10.0 + 1000 * 2.0 + 1000 * 0.5) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)


def test_agent_gives_up_after_max_steps():
    client = FakeClient([_tool_call("a"), _tool_call("b")])
    result = Agent(client, lambda q: [], model="m", max_steps=2).ask("loop forever")
    assert not result.answerable and result.confidence == "low"
    assert result.searches == ["a", "b"]


# --- evaluation -------------------------------------------------------------------------------


def test_retrieval_metrics_on_toy_index():
    a = Chunk("f.md#A#0", "f.md", "A", "A", "a")
    b = Chunk("f.md#B#0", "f.md", "B", "B", "b")
    items = [
        GoldenItem("1", "first", True, [("f.md", "A")]),  # relevant at rank 1
        GoldenItem("2", "second", True, [("f.md", "A")]),  # relevant at rank 2
        GoldenItem("3", "none", False, []),  # unanswerable: ignored by retrieval metrics
    ]
    results = {"first": [Hit(a, 0.9), Hit(b, 0.5)], "second": [Hit(b, 0.9), Hit(a, 0.5)]}
    m = retrieval_metrics(items, lambda q: results.get(q, []), k=2)
    assert m["hit@2"] == 1.0 and m["recall@2"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 0.5) / 2)
    assert m["n"] == 2


def test_agent_metrics_and_golden_set_shape():
    golden = load_golden(ROOT / "eval" / "golden.jsonl")
    assert len(golden) == 14 and sum(not g.answerable for g in golden) == 2

    from rag_lab.agent import AgentResult

    items = [GoldenItem("1", "q", True, [("f.md", "A")]), GoldenItem("2", "q", False, [])]
    results = [
        AgentResult("a", ["f.md#A#0", "f.md#B#0"], True, "high", retrieved_ids=["f.md#A#0"]),
        AgentResult("n/a", [], False, "low"),
    ]
    m = agent_metrics(items, results)
    assert m["abstention_accuracy"] == 1.0
    assert m["citations_grounded"] == 0.5  # f.md#B#0 was never retrieved
    assert m["citations_correct_section"] == 0.5
