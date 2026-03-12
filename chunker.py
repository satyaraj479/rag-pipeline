from abc import ABC, abstractmethod

from shared.models import Chunk, Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[Document]) -> list[Chunk]: ...


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=512, chunk_overlap=64, separators=None):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            for i, text in enumerate(self.splitter.split_text(doc.content)):
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_index=i,
                        metadata={**doc.metadata, "source": doc.source},
                    )
                )
        return chunks


class ParentChildChunker(BaseChunker):
    """Small child chunks indexed; large parent chunks returned at query time."""

    def __init__(self, parent_size=1500, child_size=300, overlap=30):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.p_split = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=100
        )
        self.c_split = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=overlap
        )
        self.parent_store: dict[str, Chunk] = {}

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        import uuid

        children = []
        for doc in docs:
            for pi, p_text in enumerate(self.p_split.split_text(doc.content)):
                pid = str(uuid.uuid4())
                self.parent_store[pid] = Chunk(
                    content=p_text,
                    chunk_index=pi,
                    metadata={**doc.metadata, "source": doc.source},
                )
                for ci, c_text in enumerate(self.c_split.split_text(p_text)):
                    children.append(
                        Chunk(
                            content=c_text,
                            chunk_index=ci,
                            parent_id=pid,
                            metadata={**doc.metadata, "source": doc.source},
                        )
                    )
        return children

    def get_parent(self, parent_id: str) -> Chunk | None:
        return self.parent_store.get(parent_id)


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    return {"recursive": RecursiveChunker, "parent_child": ParentChildChunker}[
        strategy
    ](**kwargs)
