"""Split Markdown documents into chunks by section, with overlap inside long sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file>#<section>#<n>", stable across runs
    source: str  # file name, e.g. "02-maintenance.md"
    section: str  # title of the deepest heading, e.g. "Daily checks"
    path: str  # full heading path, e.g. "Maintenance > Daily checks"
    text: str


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into windows of at most `size` characters, breaking at whitespace."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # Do not start in the middle of a word.
        while start < len(text) and start > 0 and not text[start - 1].isspace():
            start += 1
    return [w for w in out if w]


def chunk_markdown(source: str, markdown: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    stack: list[str] = []
    sections: list[tuple[list[str], list[str]]] = []  # (heading path, body lines)
    for line in markdown.splitlines():
        m = HEADING.match(line)
        if m:
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
            sections.append((list(stack), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip():
            sections.append(([Path(source).stem], [line]))

    chunks: list[Chunk] = []
    for heads, body in sections:
        text = "\n".join(body).strip()
        if not text:
            continue
        path = " > ".join(heads)
        for n, window in enumerate(_windows(text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{source}#{heads[-1]}#{n}",
                    source=source,
                    section=heads[-1],
                    path=path,
                    # The heading path goes into the text so retrieval can use it.
                    text=f"{path}\n\n{window}",
                )
            )
    return chunks


def load_corpus(folder: str | Path, size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(Path(folder).glob("*.md")):
        chunks.extend(chunk_markdown(file.name, file.read_text(encoding="utf-8"), size, overlap))
    return chunks
