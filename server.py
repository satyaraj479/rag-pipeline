"""FastAPI server for the RAG pipeline.

Endpoints:
    GET  /health          → liveness check
    POST /ingest          → ingest a file or directory into ChromaDB
    POST /query           → ask a question, get a Claude-grounded answer
    GET  /stats           → number of chunks in the index
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy singletons — built once on first request so startup is instant
# ---------------------------------------------------------------------------
_pipeline_cache: dict = {}


def _get_pipeline():
    if "pipeline" not in _pipeline_cache:
        from pipeline import build_pipeline
        chunker, embedder, indexer = build_pipeline()
        _pipeline_cache["chunker"] = chunker
        _pipeline_cache["embedder"] = embedder
        _pipeline_cache["indexer"] = indexer
    return (
        _pipeline_cache["chunker"],
        _pipeline_cache["embedder"],
        _pipeline_cache["indexer"],
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _pipeline_cache.clear()


app = FastAPI(
    title="RAG Pipeline API",
    description="Ingest documents and query them via Claude.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_in_store: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class StatsResponse(BaseModel):
    total_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    _, _, indexer = _get_pipeline()
    return StatsResponse(total_chunks=indexer.count())


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    import pathlib

    from extractor import ExtractorFactory, load_directory

    path = pathlib.Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    chunker, embedder, indexer = _get_pipeline()

    if path.is_dir():
        docs = load_directory(path)
    else:
        try:
            docs = ExtractorFactory.get(path).extract(path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    if not docs:
        raise HTTPException(status_code=422, detail="No text could be extracted.")

    chunks = chunker.chunk(docs)
    embedded = embedder.embed(chunks)
    count = indexer.upsert(embedded)

    return IngestResponse(chunks_indexed=count, total_in_store=indexer.count())


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    import anthropic
    import yaml

    from shared.models import Chunk

    _, embedder, indexer = _get_pipeline()

    if indexer.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="Index is empty. POST /ingest a document first.",
        )

    # Embed question and retrieve
    [embedded_query] = embedder.embed([Chunk(content=req.question)])
    chunks = indexer.search(embedded_query.embedding, top_k=req.top_k)

    # Build context
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.metadata.get("source", "unknown")
        sources.append(src)
        context_parts.append(f"[{i}] (source: {src})\n{chunk.content}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {req.question}"

    # Load generation config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. If the context does not contain enough information to "
        "answer, say so clearly. Do not make up information."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=cfg["generation"]["model"],
        max_tokens=cfg["generation"]["max_tokens"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return QueryResponse(
        answer=response.content[0].text,
        sources=list(dict.fromkeys(sources)),  # deduplicated, order preserved
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
