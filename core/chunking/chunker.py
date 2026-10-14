"""Chunking-Engine für QMS-Dokumente.

Zerlegt ParsedDocuments in semantisch sinnvolle Chunks mit Overlap.
Strategie: Abschnittsbasiert → Absatz → Satz.
"""

import hashlib
import re
from dataclasses import dataclass, field

import config
from core.logging_config import get_logger
from core.parser.base import ParsedDocument, Section

logger = get_logger("chunking")


@dataclass
class Chunk:
    """Ein einzelner Chunk eines Dokuments mit Metadaten."""

    text: str
    source_file: str
    file_name: str
    chunk_index: int
    section_title: str
    page_number: int | None = None
    last_modified: str = ""
    hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Grobe Token-Schätzung: ~4 Zeichen pro Token für deutschen Text."""
    return len(text) // 4


def _split_into_sentences(text: str) -> list[str]:
    """Teilt Text in Sätze. Berücksichtigt deutsche Abkürzungen."""
    # Satz-Grenzen erkennen, aber nicht bei Abkürzungen wie "z.B.", "Nr.", "Abs."
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])",
        text,
    )
    return [s.strip() for s in sentences if s.strip()]


class ChunkingEngine:
    """Zerlegt Dokumente in Chunks mit konfigurierbarer Größe und Overlap."""

    def __init__(
        self,
        max_chunk_size: int | None = None,
        min_chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        # Bewusst "is not None": 0 ist ein gültiger Wert für min_chunk_size
        # und overlap und darf nicht auf den Default zurückfallen.
        self.max_chunk_size = (
            max_chunk_size if max_chunk_size is not None else config.MAX_CHUNK_SIZE
        )
        self.min_chunk_size = (
            min_chunk_size if min_chunk_size is not None else config.MIN_CHUNK_SIZE
        )
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Zerlegt ein ParsedDocument in Chunks.

        Strategie:
        1. Wenn Sections vorhanden: Jede Section wird einzeln verarbeitet
        2. Zu große Sections werden an Absatzgrenzen gesplittet
        3. Overlap wird zwischen aufeinanderfolgenden Chunks eingefügt
        """
        raw_chunks = (
            self._chunk_from_sections(doc) if doc.sections else self._chunk_from_text(doc)
        )

        # Overlap hinzufügen
        chunks_with_overlap = self._add_overlap(raw_chunks)

        # Zu kleine Chunks mit Vorgänger zusammenführen
        merged_chunks = self._merge_small_chunks(chunks_with_overlap)

        # Chunk-Index neu vergeben und Hash berechnen
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i
            chunk.hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

        logger.info(
            "Dokument chunked: %s -> %d Chunks",
            doc.file_name,
            len(merged_chunks),
        )

        return merged_chunks

    def _chunk_from_sections(self, doc: ParsedDocument) -> list[Chunk]:
        """Erstellt Chunks basierend auf den Dokument-Abschnitten."""
        chunks: list[Chunk] = []

        for section in doc.sections:
            section_tokens = _estimate_tokens(section.content)

            if section_tokens <= self.max_chunk_size:
                # Section passt in einen Chunk
                chunks.append(
                    self._make_chunk(
                        text=section.content,
                        doc=doc,
                        section_title=section.title,
                        page_number=section.page_number,
                    )
                )
            else:
                # Section zu groß — an Absätzen splitten
                sub_chunks = self._split_large_section(section, doc)
                chunks.extend(sub_chunks)

        return chunks

    def _split_large_section(
        self, section: Section, doc: ParsedDocument
    ) -> list[Chunk]:
        """Splittet eine zu große Section an Absatz- oder Satzgrenzen."""
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [p.strip() for p in section.content.split("\n") if p.strip()]

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _estimate_tokens(para)

            if para_tokens > self.max_chunk_size:
                # Absatz selbst zu groß — auf Satzebene splitten
                if current_parts:
                    chunks.append(
                        self._make_chunk(
                            text="\n\n".join(current_parts),
                            doc=doc,
                            section_title=section.title,
                            page_number=section.page_number,
                        )
                    )
                    current_parts = []
                    current_tokens = 0

                sentence_chunks = self._split_by_sentences(
                    para, section, doc
                )
                chunks.extend(sentence_chunks)
                continue

            if current_tokens + para_tokens > self.max_chunk_size and current_parts:
                # Aktuellen Chunk abschließen
                chunks.append(
                    self._make_chunk(
                        text="\n\n".join(current_parts),
                        doc=doc,
                        section_title=section.title,
                        page_number=section.page_number,
                    )
                )
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        # Rest speichern
        if current_parts:
            chunks.append(
                self._make_chunk(
                    text="\n\n".join(current_parts),
                    doc=doc,
                    section_title=section.title,
                    page_number=section.page_number,
                )
            )

        return chunks

    def _split_by_sentences(
        self, text: str, section: Section, doc: ParsedDocument
    ) -> list[Chunk]:
        """Splittet Text auf Satzebene."""
        sentences = _split_into_sentences(text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.max_chunk_size and current_parts:
                chunks.append(
                    self._make_chunk(
                        text=" ".join(current_parts),
                        doc=doc,
                        section_title=section.title,
                        page_number=section.page_number,
                    )
                )
                current_parts = []
                current_tokens = 0

            current_parts.append(sentence)
            current_tokens += sent_tokens

        if current_parts:
            chunks.append(
                self._make_chunk(
                    text=" ".join(current_parts),
                    doc=doc,
                    section_title=section.title,
                    page_number=section.page_number,
                )
            )

        return chunks

    def _chunk_from_text(self, doc: ParsedDocument) -> list[Chunk]:
        """Fallback: Erstellt Chunks aus dem reinen Volltext."""
        section = Section(title="Inhalt", content=doc.text, level=0)
        return self._split_large_section(section, doc)

    def _add_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fügt Overlap zwischen aufeinanderfolgenden Chunks hinzu."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result: list[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_chunk = chunks[i]

            # Overlap-Text vom Ende des vorherigen Chunks
            overlap_chars = self.overlap * 4  # Token -> Zeichen
            if len(prev_text) > overlap_chars:
                overlap_text = prev_text[-overlap_chars:]
                # An Satzgrenze schneiden
                sentence_boundary = overlap_text.find(". ")
                if sentence_boundary > 0:
                    overlap_text = overlap_text[sentence_boundary + 2 :]
                curr_chunk.text = overlap_text + "\n\n" + curr_chunk.text
            else:
                curr_chunk.text = prev_text + "\n\n" + curr_chunk.text

            result.append(curr_chunk)

        return result

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Führt zu kleine Chunks mit dem vorherigen zusammen."""
        if not chunks:
            return chunks

        result: list[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            current = chunks[i]
            current_tokens = _estimate_tokens(current.text)

            if current_tokens < self.min_chunk_size and result:
                prev = result[-1]
                merged_tokens = _estimate_tokens(prev.text) + current_tokens
                if merged_tokens <= self.max_chunk_size * 1.2:
                    prev.text = prev.text + "\n\n" + current.text
                    continue

            result.append(current)

        return result

    def _make_chunk(
        self,
        text: str,
        doc: ParsedDocument,
        section_title: str,
        page_number: int | None = None,
    ) -> Chunk:
        """Erstellt ein Chunk-Objekt mit Dokument-Metadaten."""
        return Chunk(
            text=text,
            source_file=doc.file_path,
            file_name=doc.file_name,
            chunk_index=0,  # wird später gesetzt
            section_title=section_title,
            page_number=page_number,
        )
