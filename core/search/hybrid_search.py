"""Hybride Suchlogik mit automatischer Gewichtungsanpassung.

Kombiniert Vektor- und BM25-Suche. Erkennt automatisch, ob die Anfrage
Dokumentnummern oder Normen enthält, und passt die Gewichtung an.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import config
from core.logging_config import get_search_logger
from core.search.vector_store import VectorStore

logger = get_search_logger()

# Regex-Pattern für QMS-spezifische Begriffe
_DOC_NUMBER_PATTERN = re.compile(
    r"\b(VA|AA|FB|SOP|QMH|PA|WA)[-\s]?\d+",
    re.IGNORECASE,
)
_NORM_PATTERN = re.compile(
    r"\b(ISO|DIN|EN|IEC|VDA|IATF)\s*\d+",
    re.IGNORECASE,
)
_EXACT_TERM_PATTERN = re.compile(
    r'"([^"]+)"',  # Begriffe in Anführungszeichen
)

# Gewichtung (Vektor, BM25) je erkanntem Anfragetyp
_WEIGHTS_DOC_NUMBER = (0.3, 0.7)
_WEIGHTS_NORM = (0.4, 0.6)


@dataclass
class SearchResult:
    """Ein einzelnes Suchergebnis mit Relevanz-Score."""

    text: str
    source_file: str
    file_name: str
    section_title: str
    chunk_index: int
    page_number: int
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridSearch:
    """Hybride Suche über die Vektordatenbank mit automatischer Gewichtung."""

    def __init__(
        self,
        vector_store: VectorStore,
        default_vector_weight: float | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._default_vector_weight = (
            default_vector_weight
            if default_vector_weight is not None
            else config.HYBRID_VECTOR_WEIGHT
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Führt eine Suche durch.

        Args:
            query: Suchanfrage in natürlicher Sprache.
            top_k: Anzahl der Ergebnisse.
            filters: Metadaten-Filter (z.B. {"file_name": "QMH.pdf"}).
            mode: "hybrid" (Standard), "semantic" oder "keyword".

        Returns:
            Liste von SearchResult, sortiert nach Relevanz.

        Raises:
            ValueError: Unbekannter Suchmodus.
        """
        top_k = top_k or config.SEARCH_TOP_K

        if mode == "semantic":
            raw_results = self._vector_store.search_vector(query, top_k, filters)
        elif mode == "keyword":
            raw_results = self._vector_store.search_fts(query, top_k, filters)
        elif mode == "hybrid":
            raw_results = self._hybrid_search(query, top_k, filters)
        else:
            raise ValueError(
                f"Unbekannter Suchmodus: '{mode}'. "
                f"Erlaubt: 'hybrid', 'semantic', 'keyword'"
            )

        results = [self._to_search_result(r) for r in raw_results]

        vector_weight, bm25_weight = self.detect_weights(query)
        logger.info(
            "Suche: '%s' | Modus: %s | %d Ergebnisse | Gewichtung: %.1f/%.1f",
            query[:80],
            mode,
            len(results),
            vector_weight,
            bm25_weight,
        )

        return results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        """Hybride Suche mit automatischer Gewichtungsanpassung."""
        # Zuerst die native hybride Suche von LanceDB versuchen.
        results = self._vector_store.search_hybrid(query, top_k, filters)

        if results:
            return results

        # Fallback: beide Suchen separat ausführen und manuell kombinieren.
        vector_weight, _ = self.detect_weights(query)
        return self._manual_hybrid(query, top_k, vector_weight, filters)

    def _manual_hybrid(
        self,
        query: str,
        top_k: int,
        vector_weight: float,
        filters: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        """Manuelle Kombination von Vektor- und BM25-Ergebnissen.

        Beide Trefferlisten werden über ihren Rang gewichtet (Reciprocal-Rank-
        ähnlich), weil Distanz- und BM25-Scores nicht vergleichbar sind.
        """
        fetch_k = top_k * 2  # Mehr holen zum Kombinieren

        vector_results = self._vector_store.search_vector(query, fetch_k, filters)
        fts_results = self._vector_store.search_fts(query, fetch_k, filters)

        if not fts_results:
            return vector_results[:top_k]
        if not vector_results:
            return fts_results[:top_k]

        bm25_weight = 1.0 - vector_weight
        scored: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(vector_results):
            key = self._result_key(result)
            score = (1.0 - rank / len(vector_results)) * vector_weight
            scored[key] = {**result, "_combined_score": score}

        for rank, result in enumerate(fts_results):
            key = self._result_key(result)
            score = (1.0 - rank / len(fts_results)) * bm25_weight
            if key in scored:
                scored[key]["_combined_score"] += score
            else:
                scored[key] = {**result, "_combined_score": score}

        combined = sorted(
            scored.values(), key=lambda r: r["_combined_score"], reverse=True
        )
        return combined[:top_k]

    @staticmethod
    def _result_key(result: dict[str, Any]) -> str:
        """Eindeutiger Schlüssel eines Chunks für die Ergebnis-Zusammenführung."""
        return f"{result.get('source_file', '')}:{result.get('chunk_index', 0)}"

    def detect_weights(self, query: str) -> tuple[float, float]:
        """Erkennt die Gewichtung (Vektor, BM25) anhand der Anfrage.

        Dokumentnummern ("VA-0815") und Begriffe in Anführungszeichen müssen
        exakt gefunden werden und gewichten BM25 höher; freie Fragen gewichten
        die semantische Suche höher.
        """
        if _DOC_NUMBER_PATTERN.search(query) or _EXACT_TERM_PATTERN.search(query):
            return _WEIGHTS_DOC_NUMBER
        if _NORM_PATTERN.search(query):
            return _WEIGHTS_NORM
        return self._default_vector_weight, 1.0 - self._default_vector_weight

    def _to_search_result(self, raw: dict[str, Any]) -> SearchResult:
        """Konvertiert ein rohes DB-Ergebnis in ein SearchResult."""
        return SearchResult(
            text=raw.get("text", ""),
            source_file=raw.get("source_file", ""),
            file_name=raw.get("file_name", ""),
            section_title=raw.get("section_title", ""),
            chunk_index=raw.get("chunk_index", 0),
            page_number=raw.get("page_number", 0),
            score=self._relevance_score(raw),
        )

    @staticmethod
    def _relevance_score(raw: dict[str, Any]) -> float:
        """Normalisiert die unterschiedlichen Score-Spalten auf "größer = besser".

        LanceDB liefert je Suchmodus eine andere Spalte: `_distance` (Vektor,
        kleiner = besser), `_relevance_score` (native hybride Suche) oder
        `_score` (BM25). `_combined_score` stammt aus der manuellen
        Zusammenführung in `_manual_hybrid`.
        """
        for key in ("_combined_score", "_relevance_score", "_score"):
            if key in raw:
                return float(raw[key])
        if "_distance" in raw:
            return 1.0 / (1.0 + float(raw["_distance"]))
        return 0.0
