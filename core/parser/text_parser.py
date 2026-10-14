"""Text-Parser für einfache Formate (TXT, CSV, HTML).

Behandelt Encoding-Erkennung, CSV-Spaltenköpfe und HTML-Tag-Stripping.
"""

import csv
import io
import re
from html.parser import HTMLParser
from pathlib import Path

from core.logging_config import get_logger
from core.parser.base import DocumentParser, ParsedDocument, Section

logger = get_logger("parser.text")

# Encodings in Reihenfolge der Wahrscheinlichkeit für deutsche Dokumente.
# utf-8-sig steht vor utf-8, weil es beides liest und das BOM entfernt, das
# Windows-Exporte oft voranstellen — sonst landet U+FEFF im Chunk-Text.
# latin-1 steht am Ende, weil es jedes Byte akzeptiert und damit nie scheitert:
# es ist der Auffang-Fall, nicht die zweite Wahl.
_ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]


class _HTMLTextExtractor(HTMLParser):
    """Einfacher HTML-zu-Text-Konverter."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "head"}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._skip_tags:
            self._skip = True
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._skip_tags:
            self._skip = False
        if tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "table"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _read_file_with_encoding(file_path: Path) -> str:
    """Liest eine Textdatei mit automatischer Encoding-Erkennung."""
    for encoding in _ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Letzter Versuch: binär lesen und Fehler ignorieren
    return file_path.read_text(encoding="utf-8", errors="replace")


class TxtParser(DocumentParser):
    """Parser für reine Textdateien (.txt)."""

    def supported_extensions(self) -> set[str]:
        return {".txt"}

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        text = _read_file_with_encoding(file_path)

        logger.info("TXT geparst: %s (%d Zeichen)", file_path.name, len(text))

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            text=text,
            sections=[Section(title="Inhalt", content=text, level=0)],
        )


class CsvParser(DocumentParser):
    """Parser für CSV-Dateien.

    Spaltenköpfe werden jedem Abschnitt vorangestellt für Kontext.
    """

    def supported_extensions(self) -> set[str]:
        return {".csv"}

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        raw_text = _read_file_with_encoding(file_path)

        # CSV-Dialekt erkennen
        try:
            dialect = csv.Sniffer().sniff(raw_text[:4096])
        except csv.Error:
            dialect = csv.excel  # type: ignore[assignment]

        reader = csv.reader(io.StringIO(raw_text), dialect)
        rows = list(reader)

        if not rows:
            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                text="",
                sections=[],
            )

        headers = rows[0]
        lines: list[str] = []
        lines.append("Spalten: " + " | ".join(headers))
        lines.append("-" * 40)

        for row in rows[1:]:
            parts: list[str] = []
            for i, cell in enumerate(row):
                if cell.strip() and i < len(headers) and headers[i].strip():
                    parts.append(f"{headers[i]}: {cell.strip()}")
                elif cell.strip():
                    parts.append(cell.strip())
            if parts:
                lines.append(" | ".join(parts))

        text = "\n".join(lines)

        logger.info(
            "CSV geparst: %s (%d Zeilen, %d Zeichen)",
            file_path.name,
            len(rows),
            len(text),
        )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            text=text,
            sections=[Section(title="Tabelle", content=text, level=0)],
        )


class HtmlParser(DocumentParser):
    """Parser für HTML-Dateien. Entfernt Tags und extrahiert reinen Text."""

    def supported_extensions(self) -> set[str]:
        return {".html", ".htm"}

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        raw_html = _read_file_with_encoding(file_path)

        # HTML zu Text konvertieren
        extractor = _HTMLTextExtractor()
        extractor.feed(raw_html)
        text = extractor.get_text()

        # Mehrfache Leerzeilen reduzieren
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        logger.info("HTML geparst: %s (%d Zeichen)", file_path.name, len(text))

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            text=text,
            sections=[Section(title="Inhalt", content=text, level=0)],
        )
