"""Tests der hybriden Suchlogik und ihrer Gewichtung."""

from typing import Any

import pytest

from core.search.hybrid_search import HybridSearch


class _RecordingStore:
    """VectorStore-Double, das Aufrufe mitschreibt und feste Treffer liefert."""

    def __init__(
        self,
        vector_results: list[dict[str, Any]] | None = None,
        fts_results: list[dict[str, Any]] | None = None,
        hybrid_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self._vector_results = vector_results or []
        self._fts_results = fts_results or []
        self._hybrid_results = hybrid_results or []
        self.calls: list[str] = []

    def search_vector(self, query, top_k=None, filters=None):
        self.calls.append("vector")
        return list(self._vector_results)

    def search_fts(self, query, top_k=None, filters=None):
        self.calls.append("fts")
        return list(self._fts_results)

    def search_hybrid(self, query, top_k=None, filters=None):
        self.calls.append("hybrid")
        return list(self._hybrid_results)


def _hit(chunk_index: int, text: str = "Inhalt", **extra) -> dict[str, Any]:
    return {
        "text": text,
        "source_file": "/docs/VA-0815.pdf",
        "file_name": "VA-0815.pdf",
        "section_title": "1. Zweck",
        "chunk_index": chunk_index,
        "page_number": 3,
        **extra,
    }


def test_free_text_query_uses_the_configured_default_weight():
    search = HybridSearch(_RecordingStore(), default_vector_weight=0.6)

    assert search.detect_weights("Wie läuft die Freigabe von Dokumenten ab?") == (0.6, 0.4)


@pytest.mark.parametrize(
    "query",
    ["Was steht in VA-0815?", "Zeige mir SOP 12", "Inhalt von AA-0042", "QMH-3 Kapitel"],
)
def test_document_numbers_shift_the_weight_towards_bm25(query):
    search = HybridSearch(_RecordingStore())

    vector_weight, bm25_weight = search.detect_weights(query)

    assert bm25_weight > vector_weight


def test_quoted_terms_shift_the_weight_towards_bm25():
    search = HybridSearch(_RecordingStore())

    assert search.detect_weights('Wo wird "Lenkung von Aufzeichnungen" geregelt?') == (0.3, 0.7)


@pytest.mark.parametrize("query", ["ISO 9001 Kapitel 7.1.6", "Anforderungen aus IATF 16949"])
def test_norm_references_shift_the_weight_moderately(query):
    search = HybridSearch(_RecordingStore())

    assert search.detect_weights(query) == (0.4, 0.6)


def test_weights_always_sum_to_one():
    search = HybridSearch(_RecordingStore(), default_vector_weight=0.75)

    for query in ["freie Frage", "VA-0815", "ISO 9001", '"exakt"']:
        assert sum(search.detect_weights(query)) == pytest.approx(1.0)


def test_semantic_mode_only_hits_the_vector_index():
    store = _RecordingStore(vector_results=[_hit(0)])
    results = HybridSearch(store).search("Frage", mode="semantic")

    assert store.calls == ["vector"]
    assert len(results) == 1


def test_keyword_mode_only_hits_the_fulltext_index():
    store = _RecordingStore(fts_results=[_hit(0)])
    results = HybridSearch(store).search("VA-0815", mode="keyword")

    assert store.calls == ["fts"]
    assert len(results) == 1


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unbekannter Suchmodus"):
        HybridSearch(_RecordingStore()).search("Frage", mode="magie")


def test_native_hybrid_result_is_used_without_extra_queries():
    store = _RecordingStore(hybrid_results=[_hit(0), _hit(1)])

    results = HybridSearch(store).search("Frage")

    assert store.calls == ["hybrid"]
    assert len(results) == 2


def test_manual_fallback_combines_both_indexes():
    store = _RecordingStore(
        hybrid_results=[],
        vector_results=[_hit(0), _hit(1)],
        fts_results=[_hit(2)],
    )

    results = HybridSearch(store).search("Frage", top_k=5)

    assert store.calls == ["hybrid", "vector", "fts"]
    assert {result.chunk_index for result in results} == {0, 1, 2}


def test_chunks_found_by_both_indexes_rank_first():
    store = _RecordingStore(
        hybrid_results=[],
        # Chunk 1 steht in der Vektorsuche hinten, wird aber von BM25 bestätigt
        vector_results=[_hit(0), _hit(1)],
        fts_results=[_hit(1)],
    )

    results = HybridSearch(store, default_vector_weight=0.5).search("Frage", top_k=5)

    assert results[0].chunk_index == 1


def test_manual_fallback_without_fulltext_hits_returns_vector_hits():
    store = _RecordingStore(hybrid_results=[], vector_results=[_hit(0)], fts_results=[])

    results = HybridSearch(store).search("Frage")

    assert [result.chunk_index for result in results] == [0]


def test_top_k_is_respected_in_the_manual_fallback():
    store = _RecordingStore(
        hybrid_results=[],
        vector_results=[_hit(i) for i in range(10)],
        fts_results=[_hit(i) for i in range(10, 20)],
    )

    results = HybridSearch(store).search("Frage", top_k=3)

    assert len(results) == 3


def test_empty_index_yields_no_results():
    store = _RecordingStore(hybrid_results=[], vector_results=[], fts_results=[])

    assert HybridSearch(store).search("Frage") == []


def test_metadata_is_mapped_into_the_search_result():
    store = _RecordingStore(hybrid_results=[_hit(4, text="Freigabe durch die Leitung")])

    result = HybridSearch(store).search("Frage")[0]

    assert result.text == "Freigabe durch die Leitung"
    assert result.file_name == "VA-0815.pdf"
    assert result.section_title == "1. Zweck"
    assert result.page_number == 3
    assert result.chunk_index == 4


def test_vector_distance_is_converted_to_a_descending_score():
    store = _RecordingStore(vector_results=[_hit(0, _distance=0.0), _hit(1, _distance=3.0)])

    results = HybridSearch(store).search("Frage", mode="semantic")

    assert results[0].score > results[1].score
    assert 0.0 < results[1].score <= 1.0


def test_native_relevance_score_is_passed_through():
    store = _RecordingStore(hybrid_results=[_hit(0, _relevance_score=0.42)])

    assert HybridSearch(store).search("Frage")[0].score == pytest.approx(0.42)
