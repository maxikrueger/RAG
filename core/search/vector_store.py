"""LanceDB Vector Store Integration.

Verwaltet die Vektordatenbank: Schema, Tabellen, CRUD-Operationen
und hybride Suche (Vektor + BM25).
"""

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS

import config
from core.chunking.chunker import Chunk
from core.embeddings.base import EmbeddingProvider
from core.logging_config import get_logger

logger = get_logger("search.vector_store")


def _build_schema(embedding_dim: int) -> pa.Schema:
    """Erstellt das PyArrow-Schema für die LanceDB-Tabelle."""
    return pa.schema(
        [
            pa.field("text", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
            pa.field("source_file", pa.utf8()),
            pa.field("file_name", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("section_title", pa.utf8()),
            pa.field("page_number", pa.int32()),
            pa.field("last_modified", pa.utf8()),
            pa.field("hash", pa.utf8()),
        ]
    )


def _quote_literal(value: str) -> str:
    """Escapt einen String für eine SQL-WHERE-Klausel.

    Windows-Pfade und Dateinamen können Anführungszeichen enthalten; ohne
    Escaping bricht die Klausel ab (oder lässt sich manipulieren).
    """
    return "'" + str(value).replace("'", "''") + "'"


def _build_where(filters: dict[str, str] | None) -> str:
    """Baut eine WHERE-Klausel aus einem Metadaten-Filter-Dict.

    Raises:
        ValueError: Unbekannte Filter-Spalte.
    """
    if not filters:
        return ""

    allowed = {field.name for field in _build_schema(1) if field.name != "vector"}
    clauses: list[str] = []
    for key, value in filters.items():
        if key not in allowed:
            raise ValueError(
                f"Unbekannte Filter-Spalte: '{key}'. "
                f"Erlaubt: {', '.join(sorted(allowed))}"
            )
        clauses.append(f"{key} = {_quote_literal(value)}")

    return " AND ".join(clauses)


class VectorStore:
    """LanceDB-basierter Vector Store für QMS-Chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_path: str | Path | None = None,
        table_name: str | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._db_path = str(db_path or config.LANCEDB_DIR)
        self._table_name = table_name or config.CHUNKS_TABLE
        self._db = lancedb.connect(self._db_path)
        self._table = None
        self._fts_index_exists = False

        logger.info("VectorStore initialisiert: %s", self._db_path)

    def table_exists(self) -> bool:
        """Prüft, ob die Chunk-Tabelle bereits angelegt ist."""
        return self._table_name in self._db.table_names()

    def _scan_columns(self, columns: list[str]) -> list[dict[str, Any]]:
        """Liest ausgewählte Spalten der gesamten Tabelle.

        Die Vektoren machen den Großteil der Tabelle aus; sie hier nicht zu
        laden hält den Speicherbedarf auch bei großen Indizes klein.
        """
        table = self._get_or_create_table()
        return table.to_arrow().select(columns).to_pylist()

    def _get_or_create_table(self) -> Any:
        """Gibt die Tabelle zurück, erstellt sie falls nötig."""
        if self._table is not None:
            return self._table

        if self.table_exists():
            self._table = self._db.open_table(self._table_name)
            logger.info(
                "Bestehende Tabelle geöffnet: %s (%d Einträge)",
                self._table_name,
                self._table.count_rows(),
            )
        else:
            # dimension() lädt bei lokalen Providern das Embedding-Modell —
            # deshalb erst hier abfragen, nicht beim Öffnen einer Tabelle.
            schema = _build_schema(self._embedding_provider.dimension())
            self._table = self._db.create_table(self._table_name, schema=schema)
            logger.info("Neue Tabelle erstellt: %s", self._table_name)

        return self._table

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Fügt Chunks mit ihren Embeddings zur Datenbank hinzu.

        Args:
            chunks: Liste von Chunk-Objekten.

        Returns:
            Anzahl der hinzugefügten Chunks.
        """
        if not chunks:
            return 0

        table = self._get_or_create_table()

        # Embeddings erzeugen
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_provider.embed_texts(texts)

        # Daten als Records vorbereiten
        records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            records.append(
                {
                    "text": chunk.text,
                    "vector": embedding,
                    "source_file": chunk.source_file,
                    "file_name": chunk.file_name,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number or 0,
                    "last_modified": chunk.last_modified,
                    "hash": chunk.hash,
                }
            )

        table.add(records)

        # Der FTS-Index deckt die neuen Zeilen noch nicht ab.
        self._fts_index_exists = False

        logger.info("%d Chunks zur Datenbank hinzugefügt", len(records))
        return len(records)

    def search_vector(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Reine Vektorsuche.

        Args:
            query: Suchanfrage in natürlicher Sprache.
            top_k: Anzahl der Ergebnisse.
            filters: Metadaten-Filter (z.B. {"file_name": "QMH.pdf"}).

        Returns:
            Liste von Ergebnis-Dicts mit text, Metadaten und _distance.
        """
        table = self._get_or_create_table()
        top_k = top_k or config.SEARCH_TOP_K

        query_embedding = self._embedding_provider.embed_query(query)

        search = table.search(query_embedding).limit(top_k)

        where = _build_where(filters)
        if where:
            search = search.where(where)

        return search.to_list()

    def search_fts(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25-Volltextsuche.

        Erstellt den FTS-Index bei Bedarf selbst.
        """
        table = self._get_or_create_table()
        top_k = top_k or config.SEARCH_TOP_K

        if not self._ensure_fts_index():
            return []

        try:
            search = table.search(query, query_type="fts").limit(top_k)

            where = _build_where(filters)
            if where:
                search = search.where(where)

            return search.to_list()
        except Exception as e:
            logger.warning("FTS-Suche fehlgeschlagen: %s", e)
            return []

    def search_hybrid(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybride Suche: Vektor + BM25 über die native LanceDB-Reranking-Stufe.

        Args:
            query: Suchanfrage.
            top_k: Anzahl der Ergebnisse.
            filters: Metadaten-Filter.

        Returns:
            Kombinierte Ergebnisliste. Leer, wenn die native hybride Suche
            nicht verfügbar ist — die Aufrufebene kombiniert dann selbst.
        """
        table = self._get_or_create_table()
        top_k = top_k or config.SEARCH_TOP_K

        if not self._ensure_fts_index():
            return []

        query_embedding = self._embedding_provider.embed_query(query)

        try:
            search = (
                table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(query)
                .limit(top_k)
            )

            where = _build_where(filters)
            if where:
                search = search.where(where)

            return search.to_list()
        except Exception as e:
            logger.warning("Native hybride Suche nicht verfügbar: %s", e)
            return []

    def _ensure_fts_index(self) -> bool:
        """Stellt sicher, dass ein FTS-Index existiert. Gibt den Erfolg zurück."""
        if self._fts_index_exists:
            return True
        try:
            self.create_fts_index()
            return True
        except Exception:
            logger.warning("FTS-Index konnte nicht erstellt werden")
            return False

    def create_fts_index(self) -> None:
        """Erstellt den BM25-Volltextindex für die Spalte 'text'."""
        table = self._get_or_create_table()
        try:
            table.create_index("text", config=FTS(), replace=True)
            self._fts_index_exists = True
            logger.info("FTS-Index erstellt für Tabelle '%s'", self._table_name)
        except Exception as e:
            logger.error("FTS-Index konnte nicht erstellt werden: %s", e)
            raise

    def delete_by_source_file(self, source_file: str) -> None:
        """Löscht alle Chunks einer Quelldatei."""
        table = self._get_or_create_table()
        table.delete(f"source_file = {_quote_literal(source_file)}")
        self._fts_index_exists = False
        logger.info("Chunks gelöscht für: %s", source_file)

    def get_indexed_files(self) -> dict[str, str]:
        """Gibt ein Dict aller indizierten Dateien mit ihrem Datei-Hash zurück.

        Returns:
            Dict mit {source_file: hash}.
        """
        if not self.table_exists():
            return {}

        table = self._get_or_create_table()

        if table.count_rows() == 0:
            return {}

        rows = self._scan_columns(["source_file", "hash"])

        # Alle Chunks einer Datei tragen denselben Datei-Hash.
        return {row["source_file"]: row["hash"] for row in rows}

    def count_chunks(self) -> int:
        """Gibt die Gesamtanzahl der Chunks in der Datenbank zurück."""
        if not self.table_exists():
            return 0
        return self._get_or_create_table().count_rows()

    def count_files(self) -> int:
        """Gibt die Anzahl der indizierten Dateien zurück."""
        return len(self.get_indexed_files())

    def clear(self) -> None:
        """Löscht alle Daten aus der Tabelle."""
        if self.table_exists():
            self._db.drop_table(self._table_name)
            self._table = None
            self._fts_index_exists = False
            logger.info("Tabelle '%s' gelöscht", self._table_name)
