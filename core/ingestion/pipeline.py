"""Ingestion Pipeline für QMS-Dokumente.

Orchestriert den gesamten Prozess: Dateiscan → Parser → Chunker → Embedder → DB.
Unterstützt inkrementelle Indizierung (nur geänderte Dateien).
"""

import contextlib
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from core.chunking.chunker import ChunkingEngine
from core.embeddings.base import EmbeddingProvider
from core.logging_config import get_ingestion_logger
from core.parser import ParserRegistry, create_parser_registry
from core.search.vector_store import VectorStore

logger = get_ingestion_logger()


@dataclass
class IngestionStats:
    """Statistiken eines Ingestion-Durchlaufs."""

    total_files: int = 0
    new_files: int = 0
    updated_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _file_content_hash(file_path: Path) -> str:
    """Berechnet einen Hash über Dateigröße und Änderungszeitpunkt.

    Schneller als den gesamten Dateiinhalt zu hashen, aber ausreichend
    um Änderungen zu erkennen.
    """
    stat = file_path.stat()
    content = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(content).hexdigest()


class IngestionPipeline:
    """Pipeline für die Indizierung von QMS-Dokumenten."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore | None = None,
        parser_registry: ParserRegistry | None = None,
        chunking_engine: ChunkingEngine | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store or VectorStore(embedding_provider)
        self._parser_registry = parser_registry or create_parser_registry()
        self._chunking_engine = chunking_engine or ChunkingEngine()

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def run(
        self,
        source_path: str | Path | None = None,
        force: bool = False,
    ) -> IngestionStats:
        """Führt die Indizierung durch.

        Args:
            source_path: Pfad zu den Dokumenten. Standard: config.DOCS_PATH.
            force: Wenn True, werden alle Dateien neu indiziert (kein inkrementelles Update).

        Returns:
            IngestionStats mit Zusammenfassung des Durchlaufs.
        """
        source_path = Path(source_path or config.DOCS_PATH)
        stats = IngestionStats()
        start_time = time.time()

        if not source_path.exists():
            error_msg = f"Quellverzeichnis nicht gefunden: {source_path}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
            stats.duration_seconds = time.time() - start_time
            return stats

        # Alle unterstützten Dateien finden
        files = self._scan_files(source_path)
        stats.total_files = len(files)
        logger.info(
            "Starte Indizierung: %d Dateien in %s", len(files), source_path
        )

        # Bereits indizierte Dateien laden
        indexed_hashes = {} if force else self._vector_store.get_indexed_files()

        for file_path in files:
            try:
                file_hash = _file_content_hash(file_path)
                source_key = str(file_path)

                # Prüfen ob Datei bereits indiziert und unverändert ist
                is_update = not force and source_key in indexed_hashes

                if is_update and indexed_hashes[source_key] == file_hash:
                    stats.skipped_files += 1
                    continue

                if is_update:
                    # Datei hat sich geändert — alte Chunks löschen
                    self._vector_store.delete_by_source_file(source_key)
                    logger.info("Datei geändert, re-indiziere: %s", file_path.name)
                else:
                    logger.info("Neue Datei: %s", file_path.name)

                # Parsen
                parsed_doc = self._parser_registry.parse(file_path)

                # Chunken
                chunks = self._chunking_engine.chunk_document(parsed_doc)

                # Datei-Hash und Zeitstempel in Chunks setzen
                mod_time = time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(file_path.stat().st_mtime),
                )
                for chunk in chunks:
                    chunk.last_modified = mod_time
                    chunk.hash = file_hash  # Datei-Hash für Change-Detection

                # In DB speichern
                added = self._vector_store.add_chunks(chunks)
                stats.total_chunks += added

                # Erst nach erfolgreicher Indizierung zählen, damit eine
                # fehlgeschlagene Datei nicht zusätzlich als neu erscheint.
                if is_update:
                    stats.updated_files += 1
                else:
                    stats.new_files += 1

            except FileNotFoundError as e:
                error_msg = f"Datei nicht gefunden: {file_path} ({e})"
                logger.warning(error_msg)
                stats.failed_files += 1
                stats.errors.append(error_msg)
            except ValueError as e:
                error_msg = f"Fehler beim Parsen: {file_path.name} ({e})"
                logger.warning(error_msg)
                stats.failed_files += 1
                stats.errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unerwarteter Fehler: {file_path.name} ({e})"
                logger.error(error_msg, exc_info=True)
                stats.failed_files += 1
                stats.errors.append(error_msg)

        # FTS-Index neu erstellen
        if stats.new_files > 0 or stats.updated_files > 0:
            try:
                self._vector_store.create_fts_index()
            except Exception as e:
                logger.warning("FTS-Index konnte nicht erstellt werden: %s", e)

        stats.duration_seconds = time.time() - start_time

        logger.info(
            "Indizierung abgeschlossen: %d neue, %d aktualisiert, "
            "%d übersprungen, %d fehlgeschlagen, %d Chunks in %.1fs",
            stats.new_files,
            stats.updated_files,
            stats.skipped_files,
            stats.failed_files,
            stats.total_chunks,
            stats.duration_seconds,
        )

        return stats

    def _scan_files(self, source_path: Path) -> list[Path]:
        """Scannt das Quellverzeichnis rekursiv nach unterstützten Dateien."""
        supported = self._parser_registry.supported_extensions
        files: list[Path] = []

        for root, _dirs, filenames in os.walk(source_path):
            for filename in sorted(filenames):
                file_path = Path(root) / filename
                # Versteckte Dateien und Temp-Dateien überspringen
                if filename.startswith(".") or filename.startswith("~$"):
                    continue
                if file_path.suffix.lower() in supported:
                    files.append(file_path)

        return files

    def run_single_file(self, file_path: str | Path) -> IngestionStats:
        """Indiziert eine einzelne Datei (für manuelle Re-Indizierung)."""
        file_path = Path(file_path)
        stats = IngestionStats(total_files=1)
        start_time = time.time()

        try:
            # Alte Chunks löschen
            self._vector_store.delete_by_source_file(str(file_path))

            # Parsen + Chunken + Speichern
            parsed_doc = self._parser_registry.parse(file_path)
            chunks = self._chunking_engine.chunk_document(parsed_doc)

            mod_time = time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(file_path.stat().st_mtime),
            )
            file_hash = _file_content_hash(file_path)
            for chunk in chunks:
                chunk.last_modified = mod_time
                chunk.hash = file_hash

            added = self._vector_store.add_chunks(chunks)
            stats.total_chunks = added
            stats.new_files = 1

            # FTS-Index aktualisieren
            with contextlib.suppress(Exception):
                self._vector_store.create_fts_index()

        except Exception as e:
            stats.failed_files = 1
            stats.errors.append(str(e))
            logger.error("Fehler bei Einzeldatei-Indizierung: %s", e)

        stats.duration_seconds = time.time() - start_time
        return stats
