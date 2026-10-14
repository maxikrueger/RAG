"""Tests des PDF-Parsers.

Die Test-PDFs werden mit PyMuPDF zur Laufzeit erzeugt.
"""

import fitz
import pytest

from core.parser.pdf_parser import PdfParser


def _write_pdf(path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    document.save(str(path))
    document.close()


def test_pdf_parser_extracts_text_from_all_pages(tmp_path):
    path = tmp_path / "qmh.pdf"
    _write_pdf(path, ["Inhalt der ersten Seite", "Inhalt der zweiten Seite"])

    doc = PdfParser().parse(path)

    assert "ersten Seite" in doc.text
    assert "zweiten Seite" in doc.text


def test_pdf_parser_detects_numbered_headings(tmp_path):
    path = tmp_path / "va.pdf"
    _write_pdf(
        path,
        [
            "1. Zweck\nDiese Anweisung regelt die Lenkung.\n"
            "2. Geltungsbereich\nSie gilt fuer alle Bereiche."
        ],
    )

    doc = PdfParser().parse(path)

    titles = [section.title for section in doc.sections]
    assert "1. Zweck" in titles
    assert "2. Geltungsbereich" in titles


def test_section_page_number_follows_the_heading_page(tmp_path):
    """Regression: früher trugen alle Abschnitte die Seitenzahl der ersten Seite."""
    path = tmp_path / "mehrseitig.pdf"
    _write_pdf(
        path,
        [
            "1. Zweck\nInhalt zum Zweck.",
            "2. Ablauf\nInhalt zum Ablauf.",
            "3. Aufzeichnungen\nInhalt zu Aufzeichnungen.",
        ],
    )

    doc = PdfParser().parse(path)

    pages = {section.title: section.page_number for section in doc.sections}
    assert pages["1. Zweck"] == 1
    assert pages["2. Ablauf"] == 2
    assert pages["3. Aufzeichnungen"] == 3


def test_heading_level_follows_the_numbering_depth(tmp_path):
    path = tmp_path / "ebenen.pdf"
    _write_pdf(path, ["1. Zweck\nText.\n1.2.3 Unterpunkt\nMehr Text."])

    doc = PdfParser().parse(path)

    levels = {section.title: section.level for section in doc.sections}
    assert levels["1. Zweck"] == 1
    assert levels["1.2.3 Unterpunkt"] == 3


def test_repeated_header_lines_are_removed(tmp_path):
    path = tmp_path / "mit_kopfzeile.pdf"
    _write_pdf(
        path,
        [f"Qualitaetsmanagement-Handbuch\nAbschnitt {i} mit Inhalt." for i in range(1, 6)],
    )

    doc = PdfParser().parse(path)

    assert doc.text.count("Qualitaetsmanagement-Handbuch") == 0
    assert "Abschnitt 3 mit Inhalt." in doc.text


def test_pdf_parser_reads_document_properties(tmp_path):
    path = tmp_path / "mit_metadaten.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Inhalt", fontsize=11)
    document.set_metadata({"title": "Verfahrensanweisung", "author": "QMB"})
    document.save(str(path))
    document.close()

    doc = PdfParser().parse(path)

    assert doc.properties["title"] == "Verfahrensanweisung"
    assert doc.properties["author"] == "QMB"


def test_pdf_parser_rejects_encrypted_file(tmp_path):
    path = tmp_path / "geschuetzt.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Geheim", fontsize=11)
    document.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="pw", user_pw="pw")
    document.close()

    with pytest.raises(ValueError, match="passwortgeschützt"):
        PdfParser().parse(path)


def test_pdf_parser_rejects_broken_file(tmp_path):
    path = tmp_path / "kaputt.pdf"
    path.write_bytes(b"kein gueltiges pdf")

    with pytest.raises(ValueError, match="PDF"):
        PdfParser().parse(path)


def test_pdf_parser_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PdfParser().parse(tmp_path / "fehlt.pdf")


def test_empty_pdf_yields_no_sections(tmp_path):
    path = tmp_path / "leer.pdf"
    document = fitz.open()
    document.new_page()
    document.save(str(path))
    document.close()

    doc = PdfParser().parse(path)

    assert doc.text == ""
    assert doc.sections == []
