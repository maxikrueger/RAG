"""Tests der Parser für TXT, CSV und HTML."""

import pytest

from core.parser.text_parser import CsvParser, HtmlParser, TxtParser


def test_txt_parser_reads_utf8(tmp_path):
    path = tmp_path / "hinweis.txt"
    path.write_text("Prüfung der Maßnahmen gemäß Verfahren.", encoding="utf-8")

    doc = TxtParser().parse(path)

    assert "Prüfung der Maßnahmen" in doc.text
    assert doc.file_name == "hinweis.txt"
    assert doc.sections[0].content == doc.text


def test_txt_parser_reads_cp1252_umlauts(tmp_path):
    path = tmp_path / "alt.txt"
    path.write_bytes("Änderung der Prüfvorschrift".encode("cp1252"))

    doc = TxtParser().parse(path)

    assert "Änderung der Prüfvorschrift" in doc.text


def test_txt_parser_strips_utf8_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfGeltungsbereich")

    doc = TxtParser().parse(path)

    assert doc.text.startswith("Geltungsbereich")


def test_parsers_raise_for_missing_file(tmp_path):
    missing = tmp_path / "gibtsnicht.txt"

    for parser in (TxtParser(), CsvParser(), HtmlParser()):
        with pytest.raises(FileNotFoundError):
            parser.parse(missing)


def test_csv_parser_labels_cells_with_their_column(tmp_path):
    path = tmp_path / "matrix.csv"
    path.write_text(
        "Dokument,Version,Status\nVA-0815,2.1,freigegeben\n", encoding="utf-8"
    )

    doc = CsvParser().parse(path)

    assert "Spalten: Dokument | Version | Status" in doc.text
    assert "Dokument: VA-0815" in doc.text
    assert "Status: freigegeben" in doc.text


def test_csv_parser_handles_semicolon_dialect(tmp_path):
    path = tmp_path / "deutsch.csv"
    path.write_text("Dokument;Version\nVA-0815;2.1\n", encoding="utf-8")

    doc = CsvParser().parse(path)

    assert "Dokument: VA-0815" in doc.text


def test_csv_parser_handles_empty_file(tmp_path):
    path = tmp_path / "leer.csv"
    path.write_text("", encoding="utf-8")

    doc = CsvParser().parse(path)

    assert doc.text == ""
    assert doc.sections == []


def test_html_parser_strips_tags_scripts_and_styles(tmp_path):
    path = tmp_path / "seite.html"
    path.write_text(
        """<html><head><style>p { color: red; }</style></head>
        <body><h1>Geltungsbereich</h1><p>Gilt f&uuml;r alle Bereiche.</p>
        <script>alert('x');</script></body></html>""",
        encoding="utf-8",
    )

    doc = HtmlParser().parse(path)

    assert "Geltungsbereich" in doc.text
    assert "Gilt für alle Bereiche." in doc.text
    assert "color: red" not in doc.text
    assert "alert" not in doc.text


def test_html_parser_collapses_blank_lines(tmp_path):
    path = tmp_path / "viele_umbrueche.html"
    path.write_text("<p>Eins</p><div></div><div></div><p>Zwei</p>", encoding="utf-8")

    doc = HtmlParser().parse(path)

    assert "\n\n\n" not in doc.text
