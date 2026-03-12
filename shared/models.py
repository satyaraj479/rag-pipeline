from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    chunk_index: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]
    embedding_model: str = ""
