#!/usr/bin/env python
"""CLI: python query.py "your question here"

Retrieves relevant chunks from ChromaDB and asks Claude to answer
based solely on the retrieved context.
"""
import sys

import anthropic
import yaml
from dotenv import load_dotenv

from embedder import CachedEmbedder, HuggingFaceEmbedder
from indexer import ChromaIndexer

load_dotenv()

SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up information."""


def _load_config(config_path="config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def answer(question: str, config_path="config.yaml") -> str:
    cfg = _load_config(config_path)

    # Build embedder
    base_embedder = HuggingFaceEmbedder(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    embedder = (
        CachedEmbedder(base_embedder, cache_dir=cfg["embedding"]["cache_dir"])
        if cfg["embedding"]["cache"]
        else base_embedder
    )

    # Build indexer
    indexer = ChromaIndexer(
        collection_name=cfg["indexing"]["collection_name"],
        persist_directory=cfg["indexing"]["persist_directory"],
        distance_metric=cfg["indexing"]["distance_metric"],
    )

    if indexer.count() == 0:
        return "No documents have been indexed yet. Run: python ingest.py <path>"

    # Embed the query and retrieve top-k chunks
    from shared.models import Chunk

    query_chunk = Chunk(content=question)
    [embedded_query] = embedder.embed([query_chunk])

    top_k = cfg["generation"]["top_k_results"]
    chunks = indexer.search(embedded_query.embedding, top_k=top_k)

    # Build context block
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] (source: {source})\n{chunk.content}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"

    # Call Claude
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=cfg["generation"]["model"],
        max_tokens=cfg["generation"]["max_tokens"],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(answer(question))
