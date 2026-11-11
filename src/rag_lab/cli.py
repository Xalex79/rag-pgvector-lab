"""Command line: ingest, ask and eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import load_corpus
from .config import Settings
from .embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from .evaluate import agent_metrics, load_golden, retrieval_metrics
from .store import MemoryStore, PgVectorStore

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ("direct", "langgraph")  # direct Anthropic API (agent.py) or LangGraph (agent_langgraph.py)


def build_embedder(settings: Settings, offline: bool) -> Embedder:
    if offline:
        return HashingEmbedder()
    return OpenAICompatibleEmbedder(
        settings.embed_base_url,
        settings.embed_model,
        settings.embed_dim,
        settings.embed_doc_prefix,
        settings.embed_query_prefix,
    )


def build_store(settings: Settings, embedder: Embedder, corpus: Path, offline: bool):
    """Offline: index the corpus in memory. Online: use the PostgreSQL index built by `ingest`."""
    if offline:
        store = MemoryStore()
        chunks = load_corpus(corpus, settings.chunk_chars, settings.chunk_overlap)
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
        return store
    return PgVectorStore(settings.dsn, embedder.dim)


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    embedder = build_embedder(settings, offline=False)
    chunks = load_corpus(args.corpus, settings.chunk_chars, settings.chunk_overlap)
    store = PgVectorStore(settings.dsn, embedder.dim)
    store.reset()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    print(f"Indexed {store.count()} chunks from {args.corpus}")


def _agent(settings: Settings, search, kind: str = "direct"):
    if kind == "langgraph":
        from .agent_langgraph import LangGraphAgent, make_chat_model

        return LangGraphAgent(
            make_chat_model(settings.model, settings.effort, settings.max_tokens),
            search,
            max_steps=settings.max_steps,
            prices=settings.prices,
        )

    import anthropic

    from .agent import Agent

    return Agent(
        anthropic.Anthropic(),
        search,
        model=settings.model,
        effort=settings.effort,
        max_tokens=settings.max_tokens,
        max_steps=settings.max_steps,
        prices=settings.prices,
    )


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    embedder = build_embedder(settings, args.offline)
    store = build_store(settings, embedder, Path(args.corpus), args.offline)
    agent = _agent(settings, lambda q: store.search(embedder.embed_query(q), settings.top_k), args.agent)
    result = agent.ask(args.question)
    print(
        json.dumps(
            {
                "answer": result.answer,
                "citations": result.citations,
                "answerable": result.answerable,
                "confidence": result.confidence,
                "searches": result.searches,
                "tokens": vars(result.usage),
                "cost_usd": round(result.cost_usd, 5),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_eval(args: argparse.Namespace, settings: Settings) -> None:
    embedder = build_embedder(settings, args.offline)
    store = build_store(settings, embedder, Path(args.corpus), args.offline)
    items = load_golden(args.golden)

    def search(q: str):
        return store.search(embedder.embed_query(q), settings.top_k)

    report = {"retrieval": retrieval_metrics(items, search, settings.top_k)}
    if not args.retrieval_only:
        agent = _agent(settings, search, args.agent)
        report["agent"] = {"implementation": args.agent}
        report["agent"].update(agent_metrics(items, [agent.ask(i.question) for i in items]))
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag-lab")
    parser.add_argument("--corpus", default=str(ROOT / "corpus"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="chunk, embed and index the corpus in PostgreSQL")

    ask = sub.add_parser("ask", help="ask a question to the agent")
    ask.add_argument("question")
    ask.add_argument("--offline", action="store_true", help="hashing embeddings, in-memory index")
    ask.add_argument("--agent", choices=AGENTS, default="direct", help="agent implementation")

    ev = sub.add_parser("eval", help="evaluate retrieval (and the agent) on the golden set")
    ev.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ev.add_argument("--retrieval-only", action="store_true", help="skip the agent (no API calls)")
    ev.add_argument("--offline", action="store_true", help="hashing embeddings, in-memory index")
    ev.add_argument("--agent", choices=AGENTS, default="direct", help="agent implementation")

    args = parser.parse_args(argv)
    settings = Settings()
    {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}[args.command](args, settings)


if __name__ == "__main__":
    main()
