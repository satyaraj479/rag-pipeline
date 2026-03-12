# RAG Pipeline

A full Retrieval-Augmented Generation (RAG) pipeline built on top of the Anthropic SDK.
Ingests local documents (PDF / TXT / Markdown), embeds them with HuggingFace, stores them in
ChromaDB, and answers questions using Claude.

---

## Architecture

```
Local Files (PDF / TXT / MD)
        │
   [Extractor]        extractor.py
        │
   [Chunker]          chunker.py
        │
   [Embedder]         embedder.py   ← all-MiniLM-L6-v2 (HuggingFace)
        │
   [ChromaDB]         indexer.py    ← persistent vector store
        │
   [Retriever]        query.py / server.py
        │
   [Claude Haiku]                   ← grounded answer generation
        │
      Answer
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Ingest documents
python ingest.py ./docs

# Ask a question (CLI)
python query.py "What are the benefits of RAG?"

# Or start the API server
python server.py
```

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | Liveness check                       |
| GET    | /stats    | Number of chunks in the index        |
| POST   | /ingest   | `{ "path": "./docs" }`               |
| POST   | /query    | `{ "question": "...", "top_k": 5 }`  |

Interactive docs at `http://localhost:8000/docs`

---

## Attribution — Skill vs Claude

This project was built using a **Claude Code RAG Pipeline skill** that provides
pre-vetted, interface-consistent worker code for standard pipeline stages,
combined with Claude-generated glue code for orchestration and the API layer.

### Per-file breakdown

| File | Lines | Est. Tokens | Origin | Notes |
|------|------:|------------:|--------|-------|
| `shared/models.py` | 24 | ~115 | **Skill (verbatim)** | Data contracts: Document, Chunk, EmbeddedChunk |
| `extractor.py` | 67 | ~463 | **Skill (adapted)** | Trimmed to PDF/TXT/MD; removed HTML & CSV extractors |
| `chunker.py` | 79 | ~689 | **Skill (verbatim)** | RecursiveChunker + ParentChildChunker |
| `embedder.py` | 94 | ~783 | **Skill (verbatim)** | HuggingFaceEmbedder + CachedEmbedder |
| `indexer.py` | 60 | ~489 | **Skill (verbatim)** | ChromaIndexer |
| `config.yaml` | 22 | ~122 | **Skill (verbatim)** | Config template from skill |
| `pipeline.py` | 72 | ~541 | **Claude** | Orchestrates all stages; `build_pipeline()` / `ingest()` |
| `ingest.py` | 15 | ~79 | **Claude** | CLI entry point |
| `query.py` | 91 | ~689 | **Claude** | Embed query → retrieve → Claude generation loop |
| `server.py` | 180 | ~1,289 | **Claude** | FastAPI server; lazy singletons; Pydantic schemas |
| `.claude/launch.json` | 11 | ~62 | **Claude** | Dev server config for `preview_start` |

### Summary

| Source | Files | Lines | Est. Output Tokens | Share |
|--------|------:|------:|-------------------:|------:|
| Skill file (verbatim / adapted) | 6 | 346 | ~2,661 | **50 %** |
| Claude Code (generated) | 5 | 369 | ~2,660 | **50 %** |
| **Total** | **11** | **715** | **~5,321** | |

> **Token estimates** are based on character count ÷ 4 (standard approximation for
> code). Actual LLM token counts depend on the tokenizer and were not captured
> during generation.

### What each source contributed

**Skill file** provided:
- Standardised data contracts (`Document`, `Chunk`, `EmbeddedChunk`)
- Correct ChromaDB, HuggingFace, and LangChain API usage
- Production patterns: embedding cache, cosine distance, HNSW config, batch encoding

**Claude Code** provided:
- Technology selection (ChromaDB over pgvector, PDF/TXT only)
- `pipeline.py` — wires the four stages into a single `ingest()` call
- `query.py` — end-to-end retrieval + Claude generation with system prompt
- `server.py` — FastAPI layer with lazy initialisation, Pydantic I/O, deduped sources
- Dev tooling: `launch.json`, `.gitignore`, `pyproject.toml` dependency additions

---

## Project Structure

```
.
├── shared/
│   ├── __init__.py
│   └── models.py          ← data contracts
├── extractor.py           ← PDF / TXT / MD extraction
├── chunker.py             ← recursive & parent-child chunking
├── embedder.py            ← HuggingFace embedder + disk cache
├── indexer.py             ← ChromaDB upsert & search
├── pipeline.py            ← ingest orchestrator
├── server.py              ← FastAPI server
├── ingest.py              ← CLI: python ingest.py <path>
├── query.py               ← CLI: python query.py "question"
├── config.yaml            ← all tuneable settings
├── docs/                  ← drop your documents here
└── .claude/
    └── launch.json        ← dev server config
```

## Environment

Copy `.env.example` to `.env` and set your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```
