"""Orchestrates the ingest pipeline: extract → chunk → embed → index."""
from pathlib import Path

import yaml

from chunker import get_chunker
from embedder import CachedEmbedder, HuggingFaceEmbedder
from extractor import ExtractorFactory, load_directory
from indexer import ChromaIndexer


def _load_config(config_path="config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(config_path="config.yaml"):
    cfg = _load_config(config_path)

    chunker = get_chunker(
        cfg["chunking"]["strategy"],
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )

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

    indexer = ChromaIndexer(
        collection_name=cfg["indexing"]["collection_name"],
        persist_directory=cfg["indexing"]["persist_directory"],
        distance_metric=cfg["indexing"]["distance_metric"],
    )

    return chunker, embedder, indexer


def ingest(source: str, config_path="config.yaml") -> int:
    """Ingest a file or directory. Returns number of chunks indexed."""
    chunker, embedder, indexer = build_pipeline(config_path)

    path = Path(source)
    if path.is_dir():
        docs = load_directory(path)
    else:
        docs = ExtractorFactory.get(path).extract(path)

    if not docs:
        print(f"No documents extracted from {source}")
        return 0

    print(f"Extracted {len(docs)} document(s)")

    chunks = chunker.chunk(docs)
    print(f"Created {len(chunks)} chunk(s)")

    embedded = embedder.embed(chunks)
    print(f"Embedded {len(embedded)} chunk(s) with model "
          f"'{embedded[0].embedding_model}' (dim={embedder.dimension})")

    count = indexer.upsert(embedded)
    print(f"Indexed {count} chunk(s) → total in store: {indexer.count()}")

    return count
