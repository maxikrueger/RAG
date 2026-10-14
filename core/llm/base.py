"""Abstraktes Interface für LLM-Provider.

Ermöglicht den Wechsel zwischen Cloud (Claude) und lokal (Ollama)
durch Konfigurationsänderung.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Antwort eines LLM-Aufrufs."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstraktes Interface für LLM-Antwortgenerierung."""

    @abstractmethod
    def generate(self, prompt: str, context: str = "", system: str = "") -> LLMResponse:
        """Generiert eine Antwort.

        Args:
            prompt: Die Nutzerfrage.
            context: Relevante Dokumenten-Chunks als Kontext.
            system: System-Prompt.

        Returns:
            LLMResponse mit Text und Metadaten.
        """

    @abstractmethod
    def stream(
        self, prompt: str, context: str = "", system: str = ""
    ) -> Generator[str, None, None]:
        """Generiert eine Antwort als Stream (Token für Token).

        Args:
            prompt: Die Nutzerfrage.
            context: Relevante Dokumenten-Chunks als Kontext.
            system: System-Prompt.

        Yields:
            Einzelne Text-Tokens.
        """

    @abstractmethod
    def model_name(self) -> str:
        """Gibt den Namen des verwendeten Modells zurück."""

    @abstractmethod
    def is_available(self) -> bool:
        """Prüft ob der Provider erreichbar ist."""
