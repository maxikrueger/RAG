"""Tests der Parser-Registry."""

from pathlib import Path

import pytest

from core.parser import create_parser_registry
from core.parser.base import DocumentParser, ParsedDocument, ParserRegistry


class _FakeParser(DocumentParser):
    def __init__(self, extensions: set[str]) -> None:
        self._extensions = extensions

    def supported_extensions(self) -> set[str]:
        return self._extensions

    def parse(self, file_path: Path) -> ParsedDocument:
        return ParsedDocument(
            file_path=str(file_path), file_name=file_path.name, text="geparst"
        )


def test_registry_picks_parser_by_extension():
    registry = ParserRegistry()
    registry.register(_FakeParser({".pdf"}))
    registry.register(_FakeParser({".txt", ".md"}))

    assert registry.get_parser(Path("a.pdf")) is registry._parsers[0]
    assert registry.get_parser(Path("a.md")) is registry._parsers[1]


def test_registry_is_case_insensitive():
    registry = ParserRegistry()
    registry.register(_FakeParser({".pdf"}))

    assert registry.parse(Path("GROSS.PDF")).text == "geparst"


def test_registry_returns_none_for_unknown_extension():
    registry = ParserRegistry()
    registry.register(_FakeParser({".pdf"}))

    assert registry.get_parser(Path("a.zip")) is None


def test_registry_raises_for_unknown_extension():
    registry = ParserRegistry()

    with pytest.raises(ValueError, match="Kein Parser"):
        registry.parse(Path("a.zip"))


def test_supported_extensions_is_the_union_of_all_parsers():
    registry = ParserRegistry()
    registry.register(_FakeParser({".pdf"}))
    registry.register(_FakeParser({".txt", ".csv"}))

    assert registry.supported_extensions == {".pdf", ".txt", ".csv"}


def test_first_registered_parser_wins_for_an_extension():
    registry = ParserRegistry()
    first = _FakeParser({".txt"})
    registry.register(first)
    registry.register(_FakeParser({".txt"}))

    assert registry.get_parser(Path("a.txt")) is first


def test_default_registry_covers_the_documented_formats():
    registry = create_parser_registry()

    assert {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".csv", ".html", ".htm"} <= (
        registry.supported_extensions
    )


def test_default_registry_matches_configured_extensions():
    import config

    assert create_parser_registry().supported_extensions == config.SUPPORTED_EXTENSIONS
