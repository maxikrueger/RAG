"""RAG-Pipeline: Verbindet Suche mit LLM-Antwortgenerierung.

Orchestriert den gesamten Frage-Antwort-Prozess:
1. Relevante Chunks suchen
2. Kontext aufbauen
3. LLM-Antwort generieren
4. Quellenangaben anhängen
"""

from collections.abc import Generator
from dataclasses import dataclass

import config
from core.llm.base import LLMProvider
from core.logging_config import get_search_logger
from core.search.hybrid_search import HybridSearch, SearchResult

logger = get_search_logger()

QMS_SYSTEM_PROMPT = """Du bist ein KI-Assistent für ein Qualitätsmanagementsystem (QMS).
Deine Aufgabe ist es, Fragen zu QMS-Dokumenten präzise und hilfreich zu beantworten.

Regeln:
- Antworte ausschließlich basierend auf den bereitgestellten Dokumenten-Ausschnitten.
- Wenn die Antwort nicht in den Dokumenten zu finden ist, sage das ehrlich.
- Nenne immer die Quellen (Dateinamen und Abschnitte) für deine Aussagen.
- Antworte auf Deutsch, es sei denn die Frage ist auf Englisch gestellt.
- Fasse die relevanten Informationen zusammen, zitiere nicht einfach nur.
- Wenn mehrere Dokumente relevante Informationen enthalten, beziehe alle ein.
- Verwende eine klare, professionelle Sprache."""


@dataclass
class Source:
    """Eine Quellenangabe."""

    file_name: str
    source_file: str
    section_title: str
    page_number: int
    relevance_score: float


@dataclass
class RAGResponse:
    """Antwort der RAG-Pipeline."""

    answer: str
    sources: list[Source]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    chunks_used: int = 0


class RAGPipeline:
    """Verbindet hybride Suche mit LLM-Antwortgenerierung."""

    def __init__(
        self,
        search: HybridSearch,
        llm_provider: LLMProvider,
        system_prompt: str | None = None,
    ) -> None:
        self._search = search
        self._llm = llm_provider
        self._system_prompt = system_prompt or QMS_SYSTEM_PROMPT

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        """Beantwortet eine Frage basierend auf den QMS-Dokumenten.

        Args:
            question: Die Nutzerfrage.
            top_k: Anzahl der Kontext-Chunks.
            filters: Metadaten-Filter.
            chat_history: Bisheriger Chatverlauf für Kontext.

        Returns:
            RAGResponse mit Antwort und Quellenangaben.
        """
        top_k = top_k or config.SEARCH_TOP_K

        # 1. Relevante Chunks suchen
        results = self._search.search(question, top_k=top_k, filters=filters)

        if not results:
            return RAGResponse(
                answer="Zu dieser Frage konnte ich keine relevanten Dokumente finden. "
                "Bitte formulieren Sie die Frage um oder prüfen Sie, ob die "
                "relevanten Dokumente indiziert wurden.",
                sources=[],
                model=self._llm.model_name(),
            )

        # 2. Kontext aufbauen
        context = self._build_context(results)

        # 3. Prompt mit Chat-History aufbauen
        prompt = self._build_prompt(question, chat_history)

        # 4. LLM-Antwort generieren
        llm_response = self._llm.generate(
            prompt=prompt,
            context=context,
            system=self._system_prompt,
        )

        # 5. Quellen extrahieren
        sources = self._extract_sources(results)

        logger.info(
            "RAG-Antwort: '%s' | %d Chunks | %d/%d Tokens",
            question[:60],
            len(results),
            llm_response.input_tokens,
            llm_response.output_tokens,
        )

        return RAGResponse(
            answer=llm_response.text,
            sources=sources,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            model=llm_response.model,
            chunks_used=len(results),
        )

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Generator[str | list[Source], None, None]:
        """Wie query(), aber streamt die Antwort Token für Token.

        Yields:
            str-Tokens während des Streamings.
            Am Ende eine list[Source] als letztes Element.
        """
        top_k = top_k or config.SEARCH_TOP_K

        results = self._search.search(question, top_k=top_k, filters=filters)

        if not results:
            yield (
                "Zu dieser Frage konnte ich keine relevanten Dokumente finden. "
                "Bitte formulieren Sie die Frage um oder prüfen Sie, ob die "
                "relevanten Dokumente indiziert wurden."
            )
            yield []
            return

        context = self._build_context(results)
        prompt = self._build_prompt(question, chat_history)

        yield from self._llm.stream(
            prompt=prompt,
            context=context,
            system=self._system_prompt,
        )

        # Quellen als letztes Element
        yield self._extract_sources(results)

    def _build_context(self, results: list[SearchResult]) -> str:
        """Baut den Kontext-String aus den Suchergebnissen auf."""
        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            source_info = f"[Quelle {i}: {result.file_name}"
            if result.section_title:
                source_info += f" > {result.section_title}"
            if result.page_number and result.page_number > 0:
                source_info += f", S. {result.page_number}"
            source_info += "]"

            parts.append(f"{source_info}\n{result.text}")

        return "\n\n---\n\n".join(parts)

    def _build_prompt(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Baut den Prompt mit optionaler Chat-History auf."""
        if not chat_history:
            return question

        # Letzte N Nachrichten als Kontext mitgeben
        max_history = 6  # Letzte 3 Frage-Antwort-Paare
        recent_history = chat_history[-max_history:]

        history_text = "\n".join(
            f"{'Nutzer' if msg['role'] == 'user' else 'Assistent'}: {msg['content'][:500]}"
            for msg in recent_history
        )

        return (
            f"Bisheriger Gesprächsverlauf:\n{history_text}\n\n"
            f"Aktuelle Frage: {question}"
        )

    def _extract_sources(self, results: list[SearchResult]) -> list[Source]:
        """Extrahiert eindeutige Quellenangaben aus den Suchergebnissen."""
        seen_files: set[str] = set()
        sources: list[Source] = []

        for result in results:
            if result.file_name not in seen_files:
                seen_files.add(result.file_name)
                sources.append(
                    Source(
                        file_name=result.file_name,
                        source_file=result.source_file,
                        section_title=result.section_title,
                        page_number=result.page_number,
                        relevance_score=result.score,
                    )
                )

        return sources
