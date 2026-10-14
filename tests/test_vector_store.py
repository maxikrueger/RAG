"""Tests des LanceDB-Vector-Stores.

Läuft gegen eine echte LanceDB in tmp_path, aber mit Fake-Embeddings —
kein Modell-Download, keine Netzverbindung.
"""

import pytest

from core.chunking.chunker import Chunk
from core.search.vector_store import _build_where, _quote_literal


def _chunk(index: int, text: str, source: str = "/docs/VA-0815.pdf", **kwargs) -> Chunk:
    return Chunk(
        text=text,
        source_file=source,
        file_name=source.rsplit("/", 1)[-1],
        chunk_index=index,
        section_title=kwargs.get("section_title", "1. Zweck"),
        page_number=kwargs.get("page_number", 1),
        last_modified="2026-03-01T10:00:00",
        hash=kwargs.get("hash", "hash-0815"),
    )


# --------------------------------------------------------------------------
# WHERE-Klauseln
# --------------------------------------------------------------------------

def test_quote_literal_escapes_single_quotes():
    assert _quote_literal("O'Brien") == "'O''Brien'"


def test_quote_literal_handles_windows_paths():
    assert _quote_literal(r"Q:\QMS\Ablauf.pdf") == r"'Q:\QMS\Ablauf.pdf'"


def test_build_where_combines_filters_with_and():
    where = _build_where({"file_name": "VA-0815.pdf", "section_title": "1. Zweck"})

    assert "file_name = 'VA-0815.pdf'" in where
    assert "section_title = '1. Zweck'" in where
    assert " AND " in where


def test_build_where_is_empty_without_filters():
    assert _build_where(None) == ""
    assert _build_where({}) == ""


def test_build_where_rejects_unknown_columns():
    with pytest.raises(ValueError, match="Unbekannte Filter-Spalte"):
        _build_where({"doc_type": "VA"})


def test_build_where_rejects_the_vector_column():
    with pytest.raises(ValueError, match="Unbekannte Filter-Spalte"):
        _build_where({"vector": "egal"})


def test_build_where_neutralises_injection_attempts():
    where = _build_where({"file_name": "x' OR '1'='1"})

    assert where == "file_name = 'x'' OR ''1''=''1'"


# --------------------------------------------------------------------------
# Tabelle und Indizierung
# --------------------------------------------------------------------------

def test_table_is_created_lazily(vector_store):
    assert vector_store.table_exists() is False
    assert vector_store.count_chunks() == 0

    vector_store.add_chunks([_chunk(0, "Lenkung von Dokumenten")])

    assert vector_store.table_exists() is True
    assert vector_store.count_chunks() == 1


def test_adding_no_chunks_does_not_create_a_table(vector_store):
    assert vector_store.add_chunks([]) == 0
    assert vector_store.table_exists() is False


def test_opening_an_existing_table_does_not_need_the_embedding_dimension(
    tmp_path, fake_embeddings
):
    """Regression: dimension() lud früher das Modell auch beim reinen Öffnen."""
    from core.search.vector_store import VectorStore

    first = VectorStore(fake_embeddings, db_path=tmp_path / "db", table_name="chunks")
    first.add_chunks([_chunk(0, "Inhalt")])

    class _ExplodingEmbeddings:
        def dimension(self):
            raise AssertionError("dimension() darf beim Öffnen nicht aufgerufen werden")

        def embed_query(self, query):
            return [0.0] * 16

        def embed_texts(self, texts):
            raise AssertionError("embed_texts() wird hier nicht erwartet")

    second = VectorStore(
        _ExplodingEmbeddings(), db_path=tmp_path / "db", table_name="chunks"
    )

    assert second.count_chunks() == 1


def test_indexed_files_map_source_files_to_their_hash(vector_store):
    vector_store.add_chunks(
        [
            _chunk(0, "Inhalt A", hash="hash-a"),
            _chunk(1, "Inhalt B", hash="hash-a"),
            _chunk(0, "Inhalt C", source="/docs/QMH.pdf", hash="hash-b"),
        ]
    )

    assert vector_store.get_indexed_files() == {
        "/docs/VA-0815.pdf": "hash-a",
        "/docs/QMH.pdf": "hash-b",
    }
    assert vector_store.count_files() == 2


def test_indexed_files_is_empty_without_a_table(vector_store):
    assert vector_store.get_indexed_files() == {}


def test_delete_by_source_file_removes_only_that_file(vector_store):
    vector_store.add_chunks(
        [_chunk(0, "Inhalt A"), _chunk(0, "Inhalt C", source="/docs/QMH.pdf")]
    )

    vector_store.delete_by_source_file("/docs/VA-0815.pdf")

    assert vector_store.count_chunks() == 1
    assert list(vector_store.get_indexed_files()) == ["/docs/QMH.pdf"]


def test_delete_handles_paths_containing_quotes(vector_store):
    odd_path = "/docs/O'Brien Ablauf.pdf"
    vector_store.add_chunks([_chunk(0, "Inhalt", source=odd_path)])

    vector_store.delete_by_source_file(odd_path)

    assert vector_store.count_chunks() == 0


def test_clear_drops_the_table(vector_store):
    vector_store.add_chunks([_chunk(0, "Inhalt")])

    vector_store.clear()

    assert vector_store.table_exists() is False
    assert vector_store.count_chunks() == 0


# --------------------------------------------------------------------------
# Suche
# --------------------------------------------------------------------------

def test_vector_search_ranks_the_matching_chunk_first(vector_store):
    vector_store.add_chunks(
        [
            _chunk(0, "Freigabe und Lenkung von Dokumenten"),
            _chunk(1, "Kalibrierung von Messmitteln"),
        ]
    )

    results = vector_store.search_vector("Lenkung von Dokumenten", top_k=2)

    assert results[0]["text"] == "Freigabe und Lenkung von Dokumenten"


def test_vector_search_respects_top_k(vector_store):
    vector_store.add_chunks([_chunk(i, f"Abschnitt {i}") for i in range(5)])

    assert len(vector_store.search_vector("Abschnitt", top_k=2)) == 2


def test_vector_search_applies_metadata_filters(vector_store):
    vector_store.add_chunks(
        [
            _chunk(0, "Lenkung von Dokumenten"),
            _chunk(0, "Lenkung von Dokumenten", source="/docs/QMH.pdf"),
        ]
    )

    results = vector_store.search_vector(
        "Lenkung", top_k=5, filters={"file_name": "QMH.pdf"}
    )

    assert [result["file_name"] for result in results] == ["QMH.pdf"]


def test_fulltext_search_finds_an_exact_document_number(vector_store):
    vector_store.add_chunks(
        [
            _chunk(0, "Diese Anweisung VA-0815 regelt die Lenkung."),
            _chunk(1, "Kalibrierung von Messmitteln."),
        ]
    )

    results = vector_store.search_fts("VA-0815", top_k=5)

    assert len(results) == 1
    assert "VA-0815" in results[0]["text"]


def test_fulltext_search_applies_metadata_filters(vector_store):
    """Regression: filters wurden in der BM25-Suche früher ignoriert."""
    vector_store.add_chunks(
        [
            _chunk(0, "Anweisung VA-0815 zur Lenkung."),
            _chunk(0, "Anweisung VA-0815 zur Lenkung.", source="/docs/QMH.pdf"),
        ]
    )

    results = vector_store.search_fts(
        "VA-0815", top_k=5, filters={"file_name": "QMH.pdf"}
    )

    assert [result["file_name"] for result in results] == ["QMH.pdf"]


def test_searches_see_chunks_added_after_the_first_index_build(vector_store):
    vector_store.add_chunks([_chunk(0, "Erste Anweisung VA-0815.")])
    vector_store.search_fts("VA-0815")

    vector_store.add_chunks([_chunk(0, "Zweite Anweisung AA-0042.", source="/docs/AA.pdf")])

    assert vector_store.search_fts("AA-0042")
