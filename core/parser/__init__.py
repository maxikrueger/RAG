"""Document Parser Package.

Stellt eine vorkonfigurierte ParserRegistry mit allen verfügbaren Parsern bereit.
Parser mit fehlenden Abhängigkeiten werden übersprungen.
"""

import logging

from core.parser.base import DocumentParser, ParsedDocument, ParserRegistry, Section

logger = logging.getLogger(__name__)


def create_parser_registry() -> ParserRegistry:
    """Erstellt eine ParserRegistry mit allen verfügbaren Parsern.

    Parser deren Abhängigkeiten nicht installiert sind werden übersprungen.
    """
    registry = ParserRegistry()

    try:
        from core.parser.pdf_parser import PdfParser
        registry.register(PdfParser())
    except ImportError as e:
        logger.warning("PDF-Parser nicht verfügbar (PyMuPDF fehlt): %s", e)

    try:
        from core.parser.docx_parser import DocxParser
        registry.register(DocxParser())
    except ImportError as e:
        logger.warning("DOCX-Parser nicht verfügbar (python-docx fehlt): %s", e)

    try:
        from core.parser.xlsx_parser import XlsxParser
        registry.register(XlsxParser())
    except ImportError as e:
        logger.warning("XLSX-Parser nicht verfügbar (openpyxl fehlt): %s", e)

    try:
        from core.parser.pptx_parser import PptxParser
        registry.register(PptxParser())
    except ImportError as e:
        logger.warning("PPTX-Parser nicht verfügbar (python-pptx fehlt): %s", e)

    # Text-Parser haben keine externen Abhängigkeiten
    from core.parser.text_parser import CsvParser, HtmlParser, TxtParser
    registry.register(TxtParser())
    registry.register(CsvParser())
    registry.register(HtmlParser())

    return registry


__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "ParserRegistry",
    "Section",
    "create_parser_registry",
]
