from abc import ABC, abstractmethod
from pathlib import Path

from shared.models import Document


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, source) -> list[Document]: ...


class ExtractorFactory:
    _map = {
        ".pdf": "PDFExtractor",
        ".md": "MarkdownExtractor",
        ".txt": "TextExtractor",
    }

    @staticmethod
    def get(source) -> BaseExtractor:
        ext = Path(source).suffix.lower()
        cls = globals().get(ExtractorFactory._map.get(ext, ""))
        if not cls:
            raise ValueError(f"No extractor for: {ext}")
        return cls()


class PDFExtractor(BaseExtractor):
    def extract(self, source) -> list[Document]:
        import pdfplumber

        docs = []
        with pdfplumber.open(source) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            content=text,
                            source=str(source),
                            metadata={"page": i + 1, "file": Path(source).name},
                        )
                    )
        return docs


class TextExtractor(BaseExtractor):
    def extract(self, source) -> list[Document]:
        with open(source, encoding="utf-8") as f:
            text = f.read()
        return [Document(content=text, source=str(source),
                         metadata={"file": Path(source).name})]


class MarkdownExtractor(TextExtractor):
    pass


def load_directory(path, glob="**/*") -> list[Document]:
    docs = []
    for file in Path(path).glob(glob):
        if file.is_file():
            try:
                docs.extend(ExtractorFactory.get(file).extract(file))
            except ValueError:
                pass
    return docs
