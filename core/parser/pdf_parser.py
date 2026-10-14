"""PDF-Parser mit PyMuPDF.

Extrahiert Text seitenweise, erkennt Kopf-/Fußzeilen und
bietet optional OCR-Fallback für gescannte PDFs.
"""

import io
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from core.logging_config import get_logger
from core.parser.base import DocumentParser, ParsedDocument, Section

logger = get_logger("parser.pdf")


class PdfParser(DocumentParser):
    """Parser für PDF-Dateien basierend auf PyMuPDF."""

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            raise ValueError(f"PDF kann nicht geöffnet werden: {file_path} ({e})") from e

        if doc.is_encrypted:
            doc.close()
            raise ValueError(f"PDF ist passwortgeschützt: {file_path}")

        # Kopf-/Fußzeilen erkennen
        header_footer = self._detect_header_footer(doc)

        sections: list[Section] = []
        all_text_parts: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")

            if not page_text or not page_text.strip():
                # Möglicherweise Bild-PDF — OCR-Fallback versuchen
                page_text = self._try_ocr(page, file_path, page_num)

            if not page_text or not page_text.strip():
                continue

            # Kopf-/Fußzeilen entfernen
            cleaned_text = self._remove_header_footer(
                page_text, header_footer
            )

            if cleaned_text.strip():
                all_text_parts.append(cleaned_text)
                sections.append(
                    Section(
                        title=f"Seite {page_num + 1}",
                        content=cleaned_text,
                        level=0,
                        page_number=page_num + 1,
                    )
                )

        # Dokument-Properties auslesen
        properties = self._extract_properties(doc)
        doc.close()

        full_text = "\n\n".join(all_text_parts)

        # Versuche, echte Abschnitte aus dem Text zu erkennen
        structured_sections = self._detect_sections(sections)

        logger.info(
            "PDF geparst: %s (%d Seiten, %d Zeichen)",
            file_path.name,
            len(sections),
            len(full_text),
        )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            text=full_text,
            sections=structured_sections if structured_sections else sections,
            properties=properties,
        )

    def _detect_header_footer(self, doc: fitz.Document) -> dict[str, set[str]]:
        """Erkennt wiederkehrende Kopf- und Fußzeilen über alle Seiten.

        Eine Zeile gilt als Kopf-/Fußzeile wenn sie auf >50% der Seiten
        identisch oder fast identisch vorkommt.
        """
        if len(doc) < 3:
            return {"header": set(), "footer": set()}

        first_lines: list[str] = []
        last_lines: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text or not text.strip():
                continue
            lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
            if len(lines) >= 2:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        threshold = len(doc) * 0.5

        header_counter = Counter(first_lines)
        footer_counter = Counter(last_lines)

        headers = {line for line, count in header_counter.items() if count >= threshold}
        footers = {line for line, count in footer_counter.items() if count >= threshold}

        return {"header": headers, "footer": footers}

    def _remove_header_footer(
        self, text: str, header_footer: dict[str, set[str]]
    ) -> str:
        """Entfernt erkannte Kopf- und Fußzeilen aus dem Text."""
        if not header_footer["header"] and not header_footer["footer"]:
            return text

        lines = text.split("\n")
        result: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            # Erste nicht-leere Zeile prüfen
            if i < 3 and stripped in header_footer["header"]:
                continue
            # Letzte nicht-leere Zeile prüfen
            if i > len(lines) - 4 and stripped in header_footer["footer"]:
                continue
            result.append(line)

        return "\n".join(result)

    def _detect_sections(self, page_sections: list[Section]) -> list[Section]:
        """Bildet aus den Seiten-Abschnitten echte, überschriftenbasierte Abschnitte.

        Erkennt typische QMS-Überschriftsmuster:
        - Nummerierte Überschriften (1. Zweck, 2.1 Geltungsbereich)
        - Kurze, vollständig großgeschriebene Zeilen

        Ein Abschnitt behält die Seitennummer, auf der seine Überschrift steht —
        so zeigen die Quellenangaben auf die richtige Seite, auch wenn ein
        Abschnitt über mehrere Seiten läuft.
        """
        if not page_sections:
            return []

        sections: list[Section] = []
        current_title = ""
        current_level = 0
        current_page = page_sections[0].page_number
        current_content: list[str] = []

        def flush(fallback_title: str) -> None:
            content_text = "\n".join(current_content).strip()
            if content_text:
                sections.append(
                    Section(
                        title=current_title or fallback_title,
                        content=content_text,
                        level=current_level,
                        page_number=current_page,
                    )
                )

        for page_section in page_sections:
            for line in page_section.content.split("\n"):
                stripped = line.strip()

                if not stripped:
                    current_content.append("")
                    continue

                if self._is_heading(stripped):
                    flush("Einleitung")
                    current_title = stripped
                    current_level = self._heading_level(stripped)
                    current_page = page_section.page_number
                    current_content = []
                else:
                    if not current_content and not current_title:
                        # Erster Inhalt ohne vorangehende Überschrift
                        current_page = page_section.page_number
                    current_content.append(line)

        flush("Inhalt")

        return sections

    def _is_heading(self, line: str) -> bool:
        """Prüft ob eine Zeile eine Überschrift sein könnte."""
        if not line or len(line) > 120:
            return False

        # Nummerierte Überschriften: "1.", "1.1", "2.3.1" etc.
        if re.match(r"^\d+(\.\d+)*\.?\s+\S", line):
            return True

        # Kurze Zeile komplett in Großbuchstaben
        return line.isupper() and len(line) <= 80 and len(line.split()) <= 8

    def _heading_level(self, title: str) -> int:
        """Bestimmt das Überschriften-Level aus dem Nummerierungsmuster."""
        match = re.match(r"^(\d+(\.\d+)*)", title)
        if match:
            return match.group(1).count(".") + 1
        if title.isupper():
            return 1
        return 0

    def _extract_properties(self, doc: fitz.Document) -> dict[str, str]:
        """Liest Dokument-Metadaten aus der PDF."""
        properties: dict[str, str] = {}
        metadata = doc.metadata
        if metadata:
            mapping = {
                "title": "title",
                "author": "author",
                "subject": "subject",
                "creator": "creator",
                "producer": "producer",
                "creationDate": "creation_date",
                "modDate": "modification_date",
            }
            for pdf_key, prop_key in mapping.items():
                value = metadata.get(pdf_key, "")
                if value:
                    properties[prop_key] = str(value)
        return properties

    def _try_ocr(self, page: fitz.Page, file_path: Path, page_num: int) -> str:
        """Versucht OCR-Extraktion für eine Seite (falls pytesseract installiert)."""
        try:
            import pytesseract
            from PIL import Image

            # Seite als Bild rendern
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            text = pytesseract.image_to_string(img, lang="deu+eng")
            if text and text.strip():
                logger.info(
                    "OCR erfolgreich für %s, Seite %d", file_path.name, page_num + 1
                )
                return text
        except ImportError:
            logger.debug(
                "pytesseract nicht installiert — überspringe OCR für %s, Seite %d",
                file_path.name,
                page_num + 1,
            )
        except Exception as e:
            logger.warning(
                "OCR fehlgeschlagen für %s, Seite %d: %s",
                file_path.name,
                page_num + 1,
                e,
            )
        return ""
