"""Tests der Ingestion-Pipeline, inklusive inkrementeller Indizierung."""

import pytest

from core.chunking.chunker import ChunkingEngine
from core.ingestion.pipeline import IngestionPipeline, _file_content_hash


@pytest.fixture
def docs_dir(tmp_path):
    """Ein kleines Dokumentenverzeichnis mit Unterordner und Störfällen."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "VA-0815.txt").write_text(
        "1. Zweck\nDiese Anweisung regelt die Lenkung von Dokumenten.", encoding="utf-8"
    )
    (root / "matrix.csv").write_text("Dokument,Version\nVA-0815,2.1\n", encoding="utf-8")

    subdir = root / "Formblaetter"
    subdir.mkdir()
    (subdir / "FB-0001.txt").write_text("Formblatt zur Abweichungsmeldung.", encoding="utf-8")

    # Dateien, die ignoriert werden müssen
    (root / "notizen.md").write_text("Nicht unterstütztes Format.", encoding="utf-8")
    (root / ".versteckt.txt").write_text("Versteckte Datei.", encoding="utf-8")
    (root / "~$offen.docx").write_bytes(b"Word-Sperrdatei")
    return root


@pytest.fixture
def pipeline(fake_embeddings, vector_store):
    return IngestionPipeline(
        fake_embeddings,
        vector_store,
        chunking_engine=ChunkingEngine(max_chunk_size=200, min_chunk_size=0, overlap=0),
    )


def test_run_indexes_supported_files_recursively(pipeline, docs_dir):
    stats = pipeline.run(docs_dir)

    assert stats.total_files == 3
    assert stats.new_files == 3
    assert stats.failed_files == 0
    assert stats.total_chunks >= 3
    assert stats.duration_seconds >= 0


def test_run_skips_hidden_lock_and_unsupported_files(pipeline, docs_dir):
    pipeline.run(docs_dir)

    indexed = {path.rsplit("/", 1)[-1] for path in pipeline.vector_store.get_indexed_files()}
    assert indexed == {"VA-0815.txt", "matrix.csv", "FB-0001.txt"}


def test_second_run_skips_unchanged_files(pipeline, docs_dir):
    pipeline.run(docs_dir)

    stats = pipeline.run(docs_dir)

    assert stats.skipped_files == 3
    assert stats.new_files == 0
    assert stats.updated_files == 0


def test_changed_file_is_reindexed_without_duplicating_chunks(pipeline, docs_dir):
    pipeline.run(docs_dir)
    chunks_before = pipeline.vector_store.count_chunks()

    changed = docs_dir / "VA-0815.txt"
    changed.write_text("1. Zweck\nVollstaendig neuer Inhalt nach der Revision.", encoding="utf-8")

    stats = pipeline.run(docs_dir)

    assert stats.updated_files == 1
    assert stats.skipped_files == 2
    assert pipeline.vector_store.count_chunks() == chunks_before
    assert pipeline.vector_store.count_files() == 3


def test_new_file_is_picked_up_on_the_next_run(pipeline, docs_dir):
    pipeline.run(docs_dir)

    (docs_dir / "AA-0042.txt").write_text("Arbeitsanweisung Kalibrierung.", encoding="utf-8")
    stats = pipeline.run(docs_dir)

    assert stats.new_files == 1
    assert pipeline.vector_store.count_files() == 4


def test_force_reindexes_everything(pipeline, docs_dir):
    pipeline.run(docs_dir)

    stats = pipeline.run(docs_dir, force=True)

    assert stats.skipped_files == 0
    assert stats.new_files == 3


def test_missing_source_directory_is_reported_not_raised(pipeline, tmp_path):
    stats = pipeline.run(tmp_path / "gibtsnicht")

    assert stats.total_files == 0
    assert stats.errors
    assert "nicht gefunden" in stats.errors[0]


def test_a_broken_file_does_not_abort_the_run(pipeline, docs_dir):
    (docs_dir / "kaputt.docx").write_bytes(b"kein gueltiges docx")

    stats = pipeline.run(docs_dir)

    assert stats.failed_files == 1
    assert stats.new_files == 3
    assert any("kaputt.docx" in error for error in stats.errors)


def test_chunks_carry_file_hash_and_timestamp(pipeline, docs_dir):
    pipeline.run(docs_dir)

    indexed = pipeline.vector_store.get_indexed_files()
    expected_hash = _file_content_hash(docs_dir / "VA-0815.txt")
    assert indexed[str(docs_dir / "VA-0815.txt")] == expected_hash


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("eins", encoding="utf-8")
    first = _file_content_hash(path)

    path.write_text("eins und zwei", encoding="utf-8")

    assert _file_content_hash(path) != first


def test_single_file_run_replaces_only_that_file(pipeline, docs_dir):
    pipeline.run(docs_dir)
    target = docs_dir / "VA-0815.txt"
    target.write_text("Revidierter Inhalt der Anweisung.", encoding="utf-8")

    stats = pipeline.run_single_file(target)

    assert stats.new_files == 1
    assert stats.failed_files == 0
    assert pipeline.vector_store.count_files() == 3


def test_single_file_run_reports_failure_for_missing_file(pipeline, tmp_path):
    stats = pipeline.run_single_file(tmp_path / "fehlt.txt")

    assert stats.failed_files == 1
    assert stats.errors


def test_indexed_content_is_searchable_end_to_end(pipeline, docs_dir):
    from core.search.hybrid_search import HybridSearch

    pipeline.run(docs_dir)
    search = HybridSearch(pipeline.vector_store)

    hits = search.search("Lenkung von Dokumenten", top_k=3)

    assert hits
    assert any("Lenkung von Dokumenten" in hit.text for hit in hits)


def test_exact_document_number_is_found_end_to_end(pipeline, docs_dir):
    from core.search.hybrid_search import HybridSearch

    pipeline.run(docs_dir)
    search = HybridSearch(pipeline.vector_store)

    hits = search.search("VA-0815", top_k=3, mode="keyword")

    assert hits
    assert any("VA-0815" in hit.text for hit in hits)
