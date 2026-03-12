import hashlib

import chromadb
from chromadb.config import Settings

from shared.models import Chunk, EmbeddedChunk


class ChromaIndexer:
    def __init__(
        self,
        collection_name="documents",
        persist_directory=".chroma_store",
        distance_metric="cosine",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

    def _id(self, ec: EmbeddedChunk, i: int) -> str:
        raw = f"{ec.chunk.metadata.get('source', '')}:{ec.chunk.chunk_index}:{i}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def upsert(self, items: list[EmbeddedChunk]) -> int:
        if not items:
            return 0
        ids, vecs, docs, metas = [], [], [], []
        for i, ec in enumerate(items):
            ids.append(self._id(ec, i))
            vecs.append(ec.embedding)
            docs.append(ec.chunk.content)
            meta = {k: str(v) for k, v in ec.chunk.metadata.items()}
            if ec.chunk.parent_id:
                meta["parent_id"] = ec.chunk.parent_id
            metas.append(meta)
        self.col.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
        return len(ids)

    def search(self, vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[Chunk]:
        kw: dict = dict(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        if filters:
            kw["where"] = filters
        r = self.col.query(**kw)
        return [
            Chunk(content=d, metadata=m)
            for d, m in zip(r["documents"][0], r["metadatas"][0])
        ]

    def count(self) -> int:
        return self.col.count()
