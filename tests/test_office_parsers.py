"""Tests der Parser für DOCX, XLSX und PPTX.

Die Testdateien werden zur Laufzeit erzeugt, es liegen keine Binärdateien
im Repository.
"""

import docx
import pytest
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches

from core.parser.docx_parser import DocxParser
from core.parser.pptx_parser import PptxParser
from core.parser.xlsx_parser import XlsxParser


@pytest.fixture
def docx_file(tmp_path):
    document = docx.Document()
    document.add_heading("1. Zweck", level=1)
    document.add_paragraph("Diese Anweisung regelt die Lenkung von Dokumenten.")
    document.add_heading("2. Geltungsbereich", level=1)
    document.add_paragraph("Sie gilt für alle Bereiche.")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Dokument"
    table.cell(0, 1).text = "Version"
    table.cell(1, 0).text = "VA-0815"
    table.cell(1, 1).text = "2.1"

    document.core_properties.title = "Lenkung von Dokumenten"
    document.core_properties.author = "QM-Beauftragter"

    path = tmp_path / "VA-0815.docx"
    document.save(str(path))
    return path


def test_docx_parser_splits_at_headings(docx_file):
    doc = DocxParser().parse(docx_file)

    titles = [section.title for section in doc.sections]
    assert "1. Zweck" in titles
    assert "2. Geltungsbereich" in titles


def test_docx_parser_extracts_tables_as_sections(docx_file):
    doc = DocxParser().parse(docx_file)

    table_sections = [s for s in doc.sections if s.title.startswith("Tabelle")]
    assert len(table_sections) == 1
    assert "VA-0815 | 2.1" in table_sections[0].content


def test_docx_parser_reads_document_properties(docx_file):
    doc = DocxParser().parse(docx_file)

    assert doc.properties["title"] == "Lenkung von Dokumenten"
    assert doc.properties["author"] == "QM-Beauftragter"


def test_docx_parser_recognises_german_heading_styles(tmp_path):
    document = docx.Document()
    heading = document.add_paragraph("1. Zweck")
    heading.style = document.styles["Heading 1"]
    document.add_paragraph("Inhalt.")
    path = tmp_path / "de.docx"
    document.save(str(path))

    doc = DocxParser().parse(path)

    assert doc.sections[0].title == "1. Zweck"
    assert doc.sections[0].level == 1


def test_docx_parser_rejects_broken_file(tmp_path):
    path = tmp_path / "kaputt.docx"
    path.write_bytes(b"kein gueltiges docx")

    with pytest.raises(ValueError, match="DOCX"):
        DocxParser().parse(path)


def test_docx_parser_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocxParser().parse(tmp_path / "fehlt.docx")


@pytest.fixture
def xlsx_file(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Dokumentenlenkung"
    sheet.append(["Dokument", "Version", "Status"])
    sheet.append(["VA-0815", "2.1", "freigegeben"])
    sheet.append([None, None, None])
    sheet.append(["AA-0042", "1.0", "in Prüfung"])

    second = workbook.create_sheet("Schulungen")
    second.append(["Thema", "Datum"])
    second.append(["Hygieneschulung", "2026-03-01"])

    path = tmp_path / "matrix.xlsx"
    workbook.save(str(path))
    return path


def test_xlsx_parser_creates_one_section_per_sheet(xlsx_file):
    doc = XlsxParser().parse(xlsx_file)

    assert [section.title for section in doc.sections] == [
        "Sheet: Dokumentenlenkung",
        "Sheet: Schulungen",
    ]


def test_xlsx_parser_labels_cells_with_their_column(xlsx_file):
    doc = XlsxParser().parse(xlsx_file)

    assert "Dokument: VA-0815" in doc.text
    assert "Status: in Prüfung" in doc.text


def test_xlsx_parser_skips_empty_rows(xlsx_file):
    doc = XlsxParser().parse(xlsx_file)

    assert "None" not in doc.text


def test_xlsx_parser_rejects_broken_file(tmp_path):
    path = tmp_path / "kaputt.xlsx"
    path.write_bytes(b"kein gueltiges xlsx")

    with pytest.raises(ValueError, match="XLSX"):
        XlsxParser().parse(path)


@pytest.fixture
def pptx_file(tmp_path):
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Schulung Dokumentenlenkung"
    slide.placeholders[1].text = "Freigabe durch die Leitung"
    slide.notes_slide.notes_text_frame.text = "Hinweis auf ISO 9001 Kapitel 7.5"

    table_slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    shape = table_slide.shapes.add_table(
        2, 2, Inches(1), Inches(1), Inches(4), Inches(1)
    )
    shape.table.cell(0, 0).text = "Dokument"
    shape.table.cell(1, 0).text = "VA-0815"

    path = tmp_path / "schulung.pptx"
    presentation.save(str(path))
    return path


def test_pptx_parser_uses_slide_title_and_numbers_pages(pptx_file):
    doc = PptxParser().parse(pptx_file)

    assert doc.sections[0].title == "Schulung Dokumentenlenkung"
    assert [section.page_number for section in doc.sections] == [1, 2]


def test_pptx_parser_includes_speaker_notes(pptx_file):
    doc = PptxParser().parse(pptx_file)

    assert "[Notizen] Hinweis auf ISO 9001 Kapitel 7.5" in doc.sections[0].content


def test_pptx_parser_extracts_tables(pptx_file):
    doc = PptxParser().parse(pptx_file)

    assert "VA-0815" in doc.sections[1].content


def test_pptx_parser_rejects_broken_file(tmp_path):
    path = tmp_path / "kaputt.pptx"
    path.write_bytes(b"kein gueltiges pptx")

    with pytest.raises(ValueError, match="PPTX"):
        PptxParser().parse(path)
