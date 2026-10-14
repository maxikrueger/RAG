"""Basis-Klassen und Registry für Document Parser.

Definiert das gemeinsame Interface für alle Dokumenten-Parser
und eine Registry zum automatischen Auswählen des richtigen Parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    """Ein struktureller Abschnitt eines Dokuments."""

    title: str
    content: str
    level: int = 0  # Überschriften-Level (1=H1, 2=H2 etc., 0=kein Heading)
    page_number: int | None = None


@dataclass
class ParsedDocument:
    """Ergebnis eines Parser-Durchlaufs.

    Attributes:
        file_path: Pfad zur Quelldatei.
        file_name: Dateiname ohne Pfad.
        text: Volltext des Dokuments.
        sections: Strukturierte Abschnitte (falls vom Parser erkannt).
        properties: Dokument-Metadaten (Autor, Datum etc. aus Dateieigenschaften).
    """

    file_path: str
    file_name: str
    text: str
    sections: list[Section] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)


class DocumentParser(ABC):
    """Abstraktes Interface für Dokumenten-Parser."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Extrahiert Text und Struktur aus einer Datei.

        Args:
            file_path: Pfad zur Datei.

        Returns:
            ParsedDocument mit extrahiertem Text, Abschnitten und Properties.

        Raises:
            FileNotFoundError: Datei existiert nicht.
            ValueError: Datei kann nicht gelesen werden (korrupt, passwortgeschützt etc.).
        """

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Gibt die unterstützten Dateiendungen zurück (z.B. {'.pdf'})."""


class ParserRegistry:
    """Registry zum automatischen Auswählen des richtigen Parsers für eine Datei."""

    def __init__(self) -> None:
        self._parsers: list[DocumentParser] = []

    def register(self, parser: DocumentParser) -> None:
        """Registriert einen Parser."""
        self._parsers.append(parser)

    def get_parser(self, file_path: Path) -> DocumentParser | None:
        """Gibt den passenden Parser für eine Datei zurück, oder None."""
        suffix = file_path.suffix.lower()
        for parser in self._parsers:
            if suffix in parser.supported_extensions():
                return parser
        return None

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parst eine Datei mit dem passenden Parser.

        Raises:
            ValueError: Kein Parser für dieses Dateiformat registriert.
        """
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(
                f"Kein Parser für Dateiformat '{file_path.suffix}' registriert"
            )
        return parser.parse(file_path)

    @property
    def supported_extensions(self) -> set[str]:
        """Alle unterstützten Dateiendungen."""
        extensions: set[str] = set()
        for parser in self._parsers:
            extensions.update(parser.supported_extensions())
        return extensions
