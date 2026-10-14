"""Tests der Chunking-Engine."""

from conftest import make_document

from core.chunking.chunker import Chunk, ChunkingEngine, _estimate_tokens
from core.parser.base import ParsedDocument


def _engine(**kwargs) -> ChunkingEngine:
    defaults = {"max_chunk_size": 50, "min_chunk_size": 5, "overlap": 0}
    return ChunkingEngine(**{**defaults, **kwargs})


def test_small_sections_become_one_chunk_each():
    doc = make_document(
        [
            ("1. Zweck", "Diese Verfahrensanweisung regelt die Lenkung von Dokumenten."),
            ("2. Geltungsbereich", "Sie gilt für alle Standorte und alle Bereiche."),
        ]
    )

    chunks = _engine().chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].section_title == "1. Zweck"
    assert chunks[1].section_title == "2. Geltungsbereich"


def test_chunk_indexes_are_contiguous():
    doc = make_document([(f"{i}. Abschnitt", f"Inhalt Nummer {i}. " * 10) for i in range(5)])

    chunks = _engine().chunk_document(doc)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_oversized_section_is_split_at_paragraph_boundaries():
    paragraphs = "\n\n".join(f"Absatz {i} mit etwas Text darin." for i in range(20))
    doc = make_document([("3. Ablauf", paragraphs)])

    chunks = _engine(max_chunk_size=30).chunk_document(doc)

    assert len(chunks) > 1
    assert all(chunk.section_title == "3. Ablauf" for chunk in chunks)
    # Kein Chunk überschreitet die Grenze deutlich
    assert all(_estimate_tokens(chunk.text) <= 30 * 1.5 for chunk in chunks)


def test_oversized_paragraph_is_split_at_sentence_boundaries():
    single_paragraph = " ".join(f"Dies ist Satz Nummer {i}." for i in range(40))
    doc = make_document([("4. Durchführung", single_paragraph)])

    chunks = _engine(max_chunk_size=25).chunk_document(doc)

    assert len(chunks) > 1
    # Sätze bleiben ganz: kein Chunk endet mitten in einem Wort
    assert all(chunk.text.strip().endswith(".") for chunk in chunks)


def test_overlap_repeats_tail_of_previous_chunk():
    doc = make_document(
        [
            ("1. Zweck", "Der Zweck dieser Anweisung ist die Lenkung von Dokumenten."),
            ("2. Ablauf", "Freigaben erfolgen durch die Leitung."),
        ]
    )

    without_overlap = _engine(overlap=0, min_chunk_size=0).chunk_document(doc)
    with_overlap = _engine(overlap=10, min_chunk_size=0).chunk_document(doc)

    assert len(without_overlap) == len(with_overlap) == 2
    # Der erste Chunk bleibt unverändert, der zweite bekommt Kontext davor
    assert with_overlap[0].text == without_overlap[0].text
    assert with_overlap[1].text.endswith(without_overlap[1].text)
    assert len(with_overlap[1].text) > len(without_overlap[1].text)


def test_overlap_zero_is_respected_and_not_replaced_by_default():
    doc = make_document(
        [
            ("1. Zweck", "Der Zweck dieser Anweisung ist die Lenkung von Dokumenten."),
            ("2. Ablauf", "Freigaben erfolgen durch die Leitung."),
        ]
    )

    chunks = _engine(overlap=0, min_chunk_size=0).chunk_document(doc)

    assert chunks[1].text == "Freigaben erfolgen durch die Leitung."


def test_small_chunks_are_merged_into_predecessor():
    doc = make_document(
        [
            ("1. Zweck", "Ein ausreichend langer erster Abschnitt mit genügend Text. " * 3),
            ("2. Hinweis", "Kurz."),
        ]
    )

    chunks = _engine(min_chunk_size=20).chunk_document(doc)

    assert len(chunks) == 1
    assert "Kurz." in chunks[0].text


def test_document_without_sections_falls_back_to_full_text():
    doc = ParsedDocument(
        file_path="/docs/notiz.txt",
        file_name="notiz.txt",
        text="\n\n".join(f"Absatz {i}." for i in range(10)),
        sections=[],
    )

    chunks = _engine().chunk_document(doc)

    assert chunks
    assert all(chunk.file_name == "notiz.txt" for chunk in chunks)


def test_empty_document_yields_no_chunks():
    doc = ParsedDocument(file_path="/docs/leer.txt", file_name="leer.txt", text="", sections=[])

    assert _engine().chunk_document(doc) == []


def test_chunk_metadata_is_carried_over():
    doc = make_document([("1. Zweck", "Inhalt des Abschnitts.")])

    chunk = _engine().chunk_document(doc)[0]

    assert chunk.source_file == "/docs/VA-0815 Lenkung von Dokumenten.pdf"
    assert chunk.file_name == "VA-0815 Lenkung von Dokumenten.pdf"
    assert chunk.hash


def test_chunk_hash_is_content_addressed():
    first = Chunk(
        text="Gleicher Text",
        source_file="/a.pdf",
        file_name="a.pdf",
        chunk_index=0,
        section_title="1",
    )
    second = Chunk(
        text="Gleicher Text",
        source_file="/b.pdf",
        file_name="b.pdf",
        chunk_index=7,
        section_title="2",
    )
    other = Chunk(
        text="Anderer Text",
        source_file="/a.pdf",
        file_name="a.pdf",
        chunk_index=0,
        section_title="1",
    )

    assert first.hash == second.hash
    assert first.hash != other.hash


def test_page_numbers_survive_chunking():
    from core.parser.base import Section

    doc = ParsedDocument(
        file_path="/docs/qmh.pdf",
        file_name="qmh.pdf",
        text="",
        sections=[
            Section(title="Seite 1", content="Inhalt der ersten Seite.", page_number=1),
            Section(title="Seite 7", content="Inhalt der siebten Seite.", page_number=7),
        ],
    )

    chunks = _engine(min_chunk_size=0).chunk_document(doc)

    assert [chunk.page_number for chunk in chunks] == [1, 7]
