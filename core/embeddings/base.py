"""Abstraktes Interface für Embedding-Provider.

Ermöglicht den Wechsel zwischen lokalem Modell und Cloud-APIs
durch Konfigurationsänderung.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstraktes Interface für Embedding-Erzeugung."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Erzeugt Embeddings für eine Liste von Texten.

        Args:
            texts: Liste von Texten, für die Embeddings erzeugt werden sollen.

        Returns:
            Liste von Embedding-Vektoren (gleiche Länge wie texts).
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Erzeugt ein Embedding für eine Suchanfrage.

        Manche Modelle (z.B. E5) unterscheiden zwischen Dokumenten-
        und Query-Embeddings.

        Args:
            query: Die Suchanfrage.

        Returns:
            Embedding-Vektor.
        """

    @abstractmethod
    def dimension(self) -> int:
        """Gibt die Dimension der Embedding-Vektoren zurück."""

    @abstractmethod
    def model_name(self) -> str:
        """Gibt den Namen des Embedding-Modells zurück."""
