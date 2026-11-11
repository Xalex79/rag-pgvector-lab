"""Tests for the LangGraph version of the agent, with a fake chat model (no network, no API cost)."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from rag_lab.agent_langgraph import Answer, LangGraphAgent  # noqa: E402
from rag_lab.chunking import Chunk  # noqa: E402
from rag_lab.config import Prices  # noqa: E402
from rag_lab.store import Hit  # noqa: E402

CHUNK = Chunk(
    "02-maintenance.md#Lubrication#0",
    "02-maintenance.md",
    "Lubrication",
    "Maintenance > Lubrication",
    "Use NSF H1 grease every 500 hours.",
)


def _usage(inp=100, out=50, cache_read=0, cache_write=0):
    # LangChain reports cache tokens inside input_tokens.
    return {
        "input_tokens": inp + cache_read + cache_write,
        "output_tokens": out,
        "total_tokens": inp + cache_read + cache_write + out,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
    }


def _tool_call(query, n=1):
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_docs", "args": {"query": query}, "id": f"call_{n}"}],
        usage_metadata=_usage(cache_write=1000),
    )


class FakeChat(FakeMessagesListChatModel):
    """Returns the scripted messages in order; the structured call returns `final`."""

    final: Answer | None = None
    structured_prompts: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            self.structured_prompts.append(messages)
            raw = AIMessage(content="", usage_metadata=_usage(cache_read=1000))
            return {"raw": raw, "parsed": self.final, "parsing_error": None}

        return RunnableLambda(respond)


def test_langgraph_agent_searches_then_returns_structured_answer():
    final = Answer(
        answer="NSF H1 grease every 500 h.", citations=[CHUNK.id], answerable=True, confidence="high"
    )
    model = FakeChat(
        responses=[
            _tool_call("grease linear guides"),
            AIMessage(content="Found it.", usage_metadata=_usage()),
        ],
        final=final,
    )
    result = LangGraphAgent(model, lambda q: [Hit(CHUNK, 0.9)]).ask("What grease do the guides need?")

    assert result.answer == final.answer and result.answerable and result.confidence == "high"
    assert result.searches == ["grease linear guides"]
    assert result.retrieved_ids == [CHUNK.id]
    # The final structured call gets the question and the passage (with its id) as plain text.
    final_prompt = model.structured_prompts[0]
    assert not any(m.type == "tool" for m in final_prompt)
    assert "What grease do the guides need?" in final_prompt[-1].content
    assert CHUNK.id in final_prompt[-1].content


def test_langgraph_agent_tokens_cost_and_step_limit():
    final = Answer(answer="Not covered.", citations=[], answerable=False, confidence="low")
    # The model would keep searching forever; max_steps=2 cuts it and goes to the final answer.
    model = FakeChat(responses=[_tool_call("a", 1), _tool_call("b", 2), _tool_call("c", 3)], final=final)
    prices = Prices(input=1.0, output=10.0, cache_write=2.0, cache_read=0.5)
    result = LangGraphAgent(model, lambda q: [], max_steps=2, prices=prices).ask("loop")

    assert result.searches == ["a", "b"]
    u = result.usage
    assert (u.calls, u.input_tokens, u.output_tokens) == (3, 300, 150)
    assert (u.cache_write_tokens, u.cache_read_tokens) == (2000, 1000)
    expected = (300 * 1.0 + 150 * 10.0 + 2000 * 2.0 + 1000 * 0.5) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert not result.answerable
