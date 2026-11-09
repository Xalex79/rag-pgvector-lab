"""Agent that answers questions with Claude, calling a search tool over the vector store.

Claude decides when and how many times to call `search_docs`. The final answer is a JSON object
validated by the API through structured outputs: {answer, citations, answerable, confidence}.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Prices
from .store import Hit

SYSTEM_PROMPT = """You answer questions about the technical documentation of a machine.

Rules:
- Use the search_docs tool to find relevant passages before answering. You may call it several times
  with different queries if the first results are not enough.
- Answer only with information found in the passages. Do not use outside knowledge.
- Cite the ids of the passages you used in `citations`, exactly as returned by the tool.
- If the documentation does not contain the answer, set `answerable` to false, leave `citations`
  empty and say briefly that the manual does not cover it.
- `confidence` is high when a passage states the answer directly, medium when you had to combine
  passages, and low otherwise.
"""

SEARCH_TOOL = {
    "name": "search_docs",
    "description": (
        "Semantic search over the machine documentation. Returns the most similar passages, "
        "each with an id, its section and its text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
        },
        "required": ["query"],
    },
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "citations", "answerable", "confidence"],
    "additionalProperties": False,
}


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def cost_usd(self, prices: Prices) -> float:
        return (
            self.input_tokens * prices.input
            + self.output_tokens * prices.output
            + self.cache_write_tokens * prices.cache_write
            + self.cache_read_tokens * prices.cache_read
        ) / 1_000_000


@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    answerable: bool
    confidence: str
    retrieved_ids: list[str] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    usage: UsageTotals = field(default_factory=UsageTotals)
    cost_usd: float = 0.0


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[id: {h.chunk.id}] (section: {h.chunk.path}, score: {h.score:.2f})\n{h.chunk.text}" for h in hits
    )


class Agent:
    def __init__(
        self,
        client: Any,
        search: Callable[[str], list[Hit]],
        model: str,
        effort: str = "medium",
        max_tokens: int = 2048,
        max_steps: int = 4,
        prices: Prices | None = None,
    ) -> None:
        self.client = client
        self.search = search
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.prices = prices or Prices()

    def ask(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        usage = UsageTotals()
        retrieved: list[str] = []
        searches: list[str] = []

        for _ in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # The system prompt and tool definition are identical on every call: cache them.
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=[SEARCH_TOOL],
                messages=messages,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                },
            )
            usage.add(response.usage)

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    query = block.input.get("query", "")
                    searches.append(query)
                    hits = self.search(query)
                    retrieved.extend(h.chunk.id for h in hits if h.chunk.id not in retrieved)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": format_hits(hits)}
                    )
                messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            return AgentResult(
                answer=data["answer"],
                citations=list(data["citations"]),
                answerable=bool(data["answerable"]),
                confidence=data["confidence"],
                retrieved_ids=retrieved,
                searches=searches,
                usage=usage,
                cost_usd=usage.cost_usd(self.prices),
            )

        return AgentResult(
            answer=f"No final answer after {self.max_steps} steps.",
            citations=[],
            answerable=False,
            confidence="low",
            retrieved_ids=retrieved,
            searches=searches,
            usage=usage,
            cost_usd=usage.cost_usd(self.prices),
        )
