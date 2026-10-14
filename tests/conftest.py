"""Gemeinsame Fixtures und Fakes für die Testsuite.

Kein Test lädt ein Embedding-Modell, öffnet eine Netzverbindung oder ruft eine
API auf: Embeddings und LLM-Antworten kommen aus deterministischen Fakes.
"""

import hashlib

import pytest

from core.embeddings.base import EmbeddingProvider
from core.llm.base import LLMProvider, LLMResponse
from core.parser.base import ParsedDocument, Section

_FAKE_DIMENSION = 16


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministische Bag-of-Words-Embeddings ohne Modell.

    Jedes Wort wird auf eine Dimension gehasht. Texte mit gemeinsamen Wörtern
    liegen dadurch nah beieinander — genug, um Suchverhalten zu testen.
    """

    def __init__(self, dimension: int = _FAKE_DIMENSION) -> None:
        self._dimension = dimension
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[digest[0] % self._dimension] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            # Leerer Text: gültiger Einheitsvektor statt Nullvektor
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return self._vector(query)

    def dimension(self) -> int:
        return self._dimension

    def model_name(self) -> str:
        return "fake-embeddings"


class FakeLLMProvider(LLMProvider):
    """LLM-Provider, der eine feste Antwort liefert und die Aufrufe mitschreibt."""

    def __init__(self, answer: str = "Feste Testantwort.") -> None:
        self._answer = answer
        self.calls: list[dict[str, str]] = []

    def generate(self, prompt: str, context: str = "", system: str = "") -> LLMResponse:
        self.calls.append({"prompt": prompt, "context": context, "system": system})
        return LLMResponse(
            text=self._answer,
            model="fake-llm",
            input_tokens=len(context.split()),
            output_tokens=len(self._answer.split()),
        )

    def stream(self, prompt: str, context: str = "", system: str = ""):
        self.calls.append({"prompt": prompt, "context": context, "system": system})
        yield from self._answer.split(" ")

    def model_name(self) -> str:
        return "fake-llm"

    def is_available(self) -> bool:
        return True


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def vector_store(tmp_path, fake_embeddings):
    """Ein echter LanceDB-VectorStore in einem temporären Verzeichnis."""
    from core.search.vector_store import VectorStore

    return VectorStore(
        fake_embeddings,
        db_path=tmp_path / "lancedb",
        table_name="test_chunks",
    )


def make_document(
    sections: list[tuple[str, str]],
    file_name: str = "VA-0815 Lenkung von Dokumenten.pdf",
) -> ParsedDocument:
    """Baut ein ParsedDocument aus (Titel, Inhalt)-Paaren."""
    parsed_sections = [
        Section(title=title, content=content, level=1) for title, content in sections
    ]
    return ParsedDocument(
        file_path=f"/docs/{file_name}",
        file_name=file_name,
        text="\n\n".join(content for _, content in sections),
        sections=parsed_sections,
    )
