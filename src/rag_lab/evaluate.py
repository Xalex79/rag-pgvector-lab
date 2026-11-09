"""Evaluation against a golden set.

Retrieval: hit@k, recall@k and MRR over the answerable questions.
Agent: whether it knows when there is no answer (abstention), whether its citations are among the
retrieved passages (grounding) and whether they point to a relevant section.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .agent import AgentResult
from .chunking import Chunk
from .store import Hit


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    answerable: bool
    relevant: list[tuple[str, str]]  # (source, section)


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rel = [(r["source"], r["section"]) for r in d["relevant"]]
            items.append(GoldenItem(d["id"], d["question"], d["answerable"], rel))
    return items


def is_relevant(chunk: Chunk, item: GoldenItem) -> bool:
    return (chunk.source, chunk.section) in item.relevant


def retrieval_metrics(
    items: list[GoldenItem], search: Callable[[str], list[Hit]], k: int
) -> dict[str, float]:
    hits, recalls, rrs = [], [], []
    for item in items:
        if not item.answerable:
            continue
        results = search(item.question)[:k]
        found = {(h.chunk.source, h.chunk.section) for h in results if is_relevant(h.chunk, item)}
        hits.append(1.0 if found else 0.0)
        recalls.append(len(found) / len(item.relevant))
        rank = next((i for i, h in enumerate(results, 1) if is_relevant(h.chunk, item)), None)
        rrs.append(1.0 / rank if rank else 0.0)
    return {f"hit@{k}": mean(hits), f"recall@{k}": mean(recalls), "mrr": mean(rrs), "n": len(hits)}


def _section_of(chunk_id: str) -> tuple[str, str]:
    source, section, _ = chunk_id.split("#")
    return source, section


def agent_metrics(items: list[GoldenItem], results: list[AgentResult]) -> dict[str, float]:
    abstention, grounded, correct = [], [], []
    for item, res in zip(items, results, strict=True):
        abstention.append(1.0 if res.answerable == item.answerable else 0.0)
        if item.answerable and res.citations:
            grounded.append(mean(1.0 if c in res.retrieved_ids else 0.0 for c in res.citations))
            correct.append(mean(1.0 if _section_of(c) in item.relevant else 0.0 for c in res.citations))
    return {
        "abstention_accuracy": mean(abstention),
        "citations_grounded": mean(grounded) if grounded else 0.0,
        "citations_correct_section": mean(correct) if correct else 0.0,
        "cost_usd": sum(r.cost_usd for r in results),
    }
