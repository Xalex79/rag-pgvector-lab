# rag-pgvector-lab

Question answering over technical documentation: a **RAG** pipeline on **PostgreSQL + pgvector**, with a
**Claude agent** that decides when to search, answers in **structured JSON** with citations, says when the
manual does not cover a question, and reports **tokens and cost**. Retrieval and agent quality are measured
against a **golden set**. The agent exists in two versions, one on the Anthropic API directly and one
built with **LangGraph**, measured on the same golden set.

> Learning project, built in public. Next steps are in [docs/PLAN.md](docs/PLAN.md).

## Flow

```
corpus/*.md ──► chunking by Markdown section ──► embeddings (local model, /v1/embeddings)
   (with overlap, keeps file + section)                │
                                                       ▼
                                   PostgreSQL 17 + pgvector (HNSW index, cosine)
                                                       ▲
question ──► Claude agent ──► search_docs tool ────────┘   (called as many times as it needs)
                  │
                  ▼
   {answer, citations, answerable, confidence}  + tokens (incl. cache) and approximate cost
```

| Step | How |
|---|---|
| Chunking | By Markdown sections with overlap inside long sections. Each chunk keeps its file and heading path |
| Embeddings | A local open-weight model through any OpenAI-compatible `/v1/embeddings` server (LM Studio by default). Claude does not offer embeddings |
| Vector database | PostgreSQL 17 + pgvector in Docker, HNSW index with cosine distance |
| Agent with tool calling | Claude decides when and how many times to call `search_docs`. Two implementations: direct API and LangGraph (see below) |
| Structured output | JSON validated by the API (structured outputs): `{answer, citations, answerable, confidence}` |
| Cost | Tokens per question, including prompt cache writes and reads, and an approximate cost |
| Evaluation | 14 golden questions, 2 of them not answerable from the manual. Retrieval: hit@k, recall@k, MRR. Agent: abstention, whether citations were actually retrieved, and whether they point to the right section |
| Tests and CI | 12 unit tests with no network or API cost (both agents run against fake models), a pgvector integration test, and a GitHub Actions workflow: ruff, pytest, offline retrieval evaluation and the integration test against a pgvector service container |

The sample corpus is the manual of a **fictional** tray sealing machine, written for this project:
installation, maintenance, troubleshooting and safety.

## Two agent implementations

I'm learning LangChain and LangGraph, so I added a second version of the same agent to compare it with the
one written directly on the Anthropic API. Both use the same system prompt, the same `search_docs` tool,
the same answer fields and the same evaluation.

| | Direct API (`agent.py`) | LangGraph (`agent_langgraph.py`) |
|---|---|---|
| Loop | A `for` loop over `messages.create`; runs the tool when `stop_reason` is `tool_use` | A `StateGraph`: `agent` → `tools` (`ToolNode`) → `agent` … → `respond` |
| Structured answer | Structured outputs on every call; the last reply is the JSON answer | A separate `respond` node with `with_structured_output(method="json_schema")`, which gets the question and the passages as plain text |
| Step limit | `RAG_MAX_STEPS` calls | `RAG_MAX_STEPS` searches, then straight to `respond` |
| Tokens and cost | From `response.usage` | From LangChain's `usage_metadata`, with cache tokens split back out |

```bash
rag-lab ask --agent langgraph "How often should I replace the sealing rubber?"
rag-lab eval --agent langgraph
```

The LangGraph version always makes one extra call (the `respond` node), so it should cost a little more per
question. Whether that buys better answers is what the evaluation will show.

## Results

| Setup | hit@5 | recall@5 | MRR | Agent |
|---|---|---|---|---|
| Offline baseline (hashing bag-of-words, no model) | 1.00 | 0.96 | 0.88 | not run |
| nomic-embed-text v1.5 (LM Studio) + Claude, direct API agent | _pending_ | _pending_ | _pending_ | _pending_ |
| nomic-embed-text v1.5 (LM Studio) + Claude, LangGraph agent | _pending_ | _pending_ | _pending_ | _pending_ |

The baseline is high because the manual is small and shares vocabulary with the questions. It is a floor
for the real embedding model, not a result to show off. The real measurement is step 1 of the plan.

## Run it

Requirements: Python 3.11+, Docker, LM Studio (or another OpenAI-compatible embeddings server) with an
embedding model loaded, and an Anthropic API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in ANTHROPIC_API_KEY and check the model names
set -a && source .env && set +a

docker compose up -d            # PostgreSQL 17 + pgvector on localhost:5432
rag-lab ingest                  # chunk, embed and index corpus/
rag-lab ask "How often should I replace the sealing rubber?"
rag-lab eval                    # retrieval + agent on the golden set (makes API calls)
```

Without any external service:

```bash
pytest -q                                   # unit tests
rag-lab eval --offline --retrieval-only     # retrieval baseline, in memory
```

Integration test against the database:

```bash
docker compose up -d
RAG_TEST_DSN=postgresql://rag:rag@localhost:5432/rag pytest -q -m integration
```

## Configuration

Everything is set with environment variables (see [.env.example](.env.example)). The most useful:

| Variable | Default | Notes |
|---|---|---|
| `RAG_MODEL` | `claude-opus-5-5` | Use a smaller model to spend less while testing |
| `RAG_EFFORT` | `medium` | `low`, `medium`, `high`... |
| `RAG_EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | As named in LM Studio |
| `RAG_EMBED_DIM` | `768` | Must match the model; changing it requires `rag-lab ingest` again |
| `RAG_TOP_K` | `5` | Passages returned per search |
| `RAG_PRICE_*` | see file | Approximate USD per million tokens, only used for the cost estimate |

## What has been verified

- Unit tests, lint and the offline retrieval evaluation pass.
- The integration test passes against PostgreSQL 16 with pgvector 0.6 (CI uses PostgreSQL 17).
- `ingest` and `eval` were run end to end through HTTP against a stand-in embeddings server.
- `ask` was run with both agents against a stand-in Messages API to check the exact requests they send
  (tools, prompt caching, effort and the JSON schema).

Not yet verified: a real run with LM Studio embeddings and the Claude API. That is step 1 of the plan.

## Layout

```
corpus/          sample manual (fictional machine)
eval/            golden set
src/rag_lab/     chunking, embeddings, store (pgvector + in-memory), agents (direct API and LangGraph),
                 evaluation, CLI
tests/           unit tests and the pgvector integration test
docs/PLAN.md     next steps
```
