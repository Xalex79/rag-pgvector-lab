"""The same agent built with LangGraph, to compare it with the direct-API version in agent.py.

Graph:

    START -> agent --(tool calls?)--> tools -> agent ... --(no more tool calls)--> respond -> END

- agent:   the chat model with the search_docs tool bound; decides whether to search again.
- tools:   LangGraph's ToolNode runs search_docs over the vector store.
- respond: the same model with structured output produces {answer, citations, answerable, confidence}
           from the question and the passages found.

It reuses the system prompt, the search tool description, the answer fields and the evaluation of
agent.py, so both versions can be measured on the same golden set.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from .agent import SEARCH_TOOL, SYSTEM_PROMPT, AgentResult, UsageTotals, format_hits
from .config import Prices
from .store import Hit

FINAL_INSTRUCTION = (
    "Give your final answer to the question using only these passages, following the rules in the "
    "system prompt. Cite passage ids exactly as they appear."
)


class Answer(BaseModel):
    answer: str
    citations: list[str] = Field(description="Ids of the passages used, exactly as returned by the tool.")
    answerable: bool
    confidence: Literal["high", "medium", "low"]


class State(MessagesState):
    answer: dict | None


def _usage_from_metadata(meta: dict | None) -> SimpleNamespace:
    """LangChain's usage_metadata counts cache tokens inside input_tokens; split them back out."""
    meta = meta or {}
    details = meta.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0) or 0
    cache_write = details.get("cache_creation", 0) or 0
    return SimpleNamespace(
        input_tokens=(meta.get("input_tokens", 0) or 0) - cache_read - cache_write,
        output_tokens=meta.get("output_tokens", 0) or 0,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


class LangGraphAgent:
    def __init__(
        self,
        chat_model: Any,
        search: Callable[[str], list[Hit]],
        max_steps: int = 4,
        prices: Prices | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.search = search
        self.max_steps = max_steps
        self.prices = prices or Prices()

    def _build(self, retrieved: list[str], searches: list[str], usage: UsageTotals):
        def search_docs(query: str) -> str:
            searches.append(query)
            hits = self.search(query)
            retrieved.extend(h.chunk.id for h in hits if h.chunk.id not in retrieved)
            return format_hits(hits)

        tool = StructuredTool.from_function(
            search_docs, name=SEARCH_TOOL["name"], description=SEARCH_TOOL["description"]
        )
        model_with_tools = self.chat_model.bind_tools([tool])
        structured = self.chat_model.with_structured_output(Answer, method="json_schema", include_raw=True)
        system = SystemMessage(
            content=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        )

        def agent_node(state: State) -> dict:
            steps = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
            if steps >= self.max_steps:
                # Stop searching: go straight to the final answer.
                return {"messages": [AIMessage(content="Search limit reached.")]}
            reply = model_with_tools.invoke([system, *state["messages"]])
            usage.add(_usage_from_metadata(reply.usage_metadata))
            return {"messages": [reply]}

        def respond_node(state: State) -> dict:
            # The final call gets the question and the passages as plain text, without tool blocks,
            # so it does not need the tool definitions and cannot start another search.
            question = state["messages"][0].content
            passages = "\n\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage))
            passages = passages or "(no passages)"
            prompt = f"Question: {question}\n\nPassages found:\n\n{passages}\n\n{FINAL_INSTRUCTION}"
            result = structured.invoke([system, HumanMessage(content=prompt)])
            usage.add(_usage_from_metadata(getattr(result["raw"], "usage_metadata", None)))
            parsed = result["parsed"]
            return {"answer": parsed.model_dump() if parsed is not None else None}

        graph = StateGraph(State)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode([tool]))
        graph.add_node("respond", respond_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "respond"})
        graph.add_edge("tools", "agent")
        graph.add_edge("respond", END)
        return graph.compile()

    def ask(self, question: str) -> AgentResult:
        retrieved: list[str] = []
        searches: list[str] = []
        usage = UsageTotals()
        graph = self._build(retrieved, searches, usage)
        state = graph.invoke(
            {"messages": [HumanMessage(content=question)], "answer": None},
            {"recursion_limit": 2 * self.max_steps + 5},
        )
        data = state["answer"] or {
            "answer": "The model did not return a valid answer.",
            "citations": [],
            "answerable": False,
            "confidence": "low",
        }
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


def make_chat_model(model: str, effort: str, max_tokens: int) -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=max_tokens, output_config={"effort": effort})
