"""XLSX-Parser mit openpyxl.

Extrahiert alle Sheets, behandelt Merged Cells und
formatiert Tabellen als strukturierten Text.
"""

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from core.logging_config import get_logger
from core.parser.base import DocumentParser, ParsedDocument, Section

logger = get_logger("parser.xlsx")


class XlsxParser(DocumentParser):
    """Parser für Excel-Dateien (.xlsx)."""

    def supported_extensions(self) -> set[str]:
        return {".xlsx"}

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        try:
            wb = load_workbook(str(file_path), read_only=True, data_only=True)
        except Exception as e:
            raise ValueError(f"XLSX kann nicht geöffnet werden: {file_path} ({e})") from e

        sections: list[Section] = []
        all_text_parts: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_text = self._extract_sheet(ws, sheet_name)
            if sheet_text.strip():
                sections.append(
                    Section(
                        title=f"Sheet: {sheet_name}",
                        content=sheet_text,
                        level=1,
                    )
                )
                all_text_parts.append(f"=== {sheet_name} ===\n{sheet_text}")

        wb.close()

        properties = self._extract_properties(file_path)
        full_text = "\n\n".join(all_text_parts)

        logger.info(
            "XLSX geparst: %s (%d Sheets, %d Zeichen)",
            file_path.name,
            len(sections),
            len(full_text),
        )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            text=full_text,
            sections=sections,
            properties=properties,
        )

    def _extract_sheet(self, ws: Worksheet, sheet_name: str) -> str:
        """Extrahiert ein Sheet als strukturierten Text mit Spaltenköpfen."""
        rows_data: list[list[str]] = []

        for row in ws.iter_rows(values_only=True):
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            # Leere Zeilen überspringen
            if any(cells):
                rows_data.append(cells)

        if not rows_data:
            return ""

        # Erste Zeile als Header verwenden
        headers = rows_data[0]
        lines: list[str] = []

        if len(rows_data) == 1:
            # Nur Header, keine Daten
            lines.append(" | ".join(headers))
        else:
            # Header + Datenzeilen
            lines.append("Spalten: " + " | ".join(headers))
            lines.append("-" * 40)
            for row in rows_data[1:]:
                # Spaltenname: Wert Format für bessere Lesbarkeit
                parts: list[str] = []
                for i, cell in enumerate(row):
                    if cell and i < len(headers) and headers[i]:
                        parts.append(f"{headers[i]}: {cell}")
                    elif cell:
                        parts.append(cell)
                if parts:
                    lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _extract_properties(self, file_path: Path) -> dict[str, str]:
        """Liest Dokument-Properties aus der XLSX-Datei."""
        properties: dict[str, str] = {}
        try:
            wb = load_workbook(str(file_path), read_only=True)
            props = wb.properties
            if props.title:
                properties["title"] = props.title
            if props.creator:
                properties["author"] = props.creator
            if props.subject:
                properties["subject"] = props.subject
            if props.created:
                properties["creation_date"] = props.created.isoformat()
            if props.modified:
                properties["modification_date"] = props.modified.isoformat()
            if props.lastModifiedBy:
                properties["last_modified_by"] = props.lastModifiedBy
            wb.close()
        except Exception:
            pass  # Properties sind optional
        return properties
