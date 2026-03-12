from abc import ABC, abstractmethod

from shared.models import Chunk, EmbeddedChunk


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None,
                 batch_size=32, normalize=True):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        vecs = self.model.encode(
            [c.content for c in chunks],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(chunks) > 100,
            convert_to_numpy=True,
        )
        return [
            EmbeddedChunk(chunk=c, embedding=v.tolist(), embedding_model=self.model_name)
            for c, v in zip(chunks, vecs)
        ]


class CachedEmbedder(BaseEmbedder):
    """Disk cache keyed on content+model hash. Avoids re-embedding unchanged docs."""

    def __init__(self, embedder: BaseEmbedder, cache_dir=".embedding_cache"):
        import pathlib

        pathlib.Path(cache_dir).mkdir(exist_ok=True)
        self.embedder = embedder
        self.cache_dir = cache_dir

    @property
    def dimension(self):
        return self.embedder.dimension

    def _key(self, text, model):
        import hashlib
        import os

        h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{h}.json")

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        import json
        import os

        model = getattr(self.embedder, "model_name", "unknown")
        results: list[EmbeddedChunk | None] = [None] * len(chunks)
        miss_idx = []
        for i, c in enumerate(chunks):
            k = self._key(c.content, model)
            if os.path.exists(k):
                with open(k) as f:
                    d = json.load(f)
                results[i] = EmbeddedChunk(
                    chunk=c, embedding=d["embedding"], embedding_model=d["model"]
                )
            else:
                miss_idx.append(i)
        if miss_idx:
            embedded = self.embedder.embed([chunks[i] for i in miss_idx])
            for i, ec in zip(miss_idx, embedded):
                with open(self._key(chunks[i].content, model), "w") as f:
                    json.dump({"embedding": ec.embedding, "model": ec.embedding_model}, f)
                results[i] = ec
        return results
