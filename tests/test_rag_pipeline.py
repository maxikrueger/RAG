"""Tests der RAG-Pipeline (Suche → Kontext → LLM → Quellen)."""

from core.rag import QMS_SYSTEM_PROMPT, RAGPipeline
from core.search.hybrid_search import SearchResult


class _StubSearch:
    """HybridSearch-Double mit festen Treffern."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[dict] = []

    def search(self, query, top_k=None, filters=None, mode="hybrid"):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._results)


def _result(file_name: str, text: str, page: int = 0, section: str = "") -> SearchResult:
    return SearchResult(
        text=text,
        source_file=f"/docs/{file_name}",
        file_name=file_name,
        section_title=section,
        chunk_index=0,
        page_number=page,
        score=0.9,
    )


def test_answer_and_token_counts_come_from_the_provider(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Freigabe durch die Leitung.")])

    response = RAGPipeline(search, fake_llm).query("Wer gibt Dokumente frei?")

    assert response.answer == "Feste Testantwort."
    assert response.model == "fake-llm"
    assert response.chunks_used == 1
    assert response.output_tokens > 0


def test_context_contains_every_chunk_with_a_source_label(fake_llm):
    search = _StubSearch(
        [
            _result("VA-0815.pdf", "Erster Treffer.", page=3, section="1. Zweck"),
            _result("QMH.pdf", "Zweiter Treffer."),
        ]
    )

    RAGPipeline(search, fake_llm).query("Frage")

    context = fake_llm.calls[0]["context"]
    assert "[Quelle 1: VA-0815.pdf > 1. Zweck, S. 3]" in context
    assert "[Quelle 2: QMH.pdf]" in context
    assert "Erster Treffer." in context
    assert "Zweiter Treffer." in context


def test_system_prompt_is_passed_to_the_provider(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    RAGPipeline(search, fake_llm).query("Frage")

    assert fake_llm.calls[0]["system"] == QMS_SYSTEM_PROMPT


def test_custom_system_prompt_overrides_the_default(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    RAGPipeline(search, fake_llm, system_prompt="Eigener Prompt").query("Frage")

    assert fake_llm.calls[0]["system"] == "Eigener Prompt"


def test_sources_are_deduplicated_per_file(fake_llm):
    search = _StubSearch(
        [
            _result("VA-0815.pdf", "Treffer A"),
            _result("VA-0815.pdf", "Treffer B"),
            _result("QMH.pdf", "Treffer C"),
        ]
    )

    response = RAGPipeline(search, fake_llm).query("Frage")

    assert [source.file_name for source in response.sources] == ["VA-0815.pdf", "QMH.pdf"]
    assert response.chunks_used == 3


def test_no_search_hits_means_no_llm_call(fake_llm):
    response = RAGPipeline(_StubSearch([]), fake_llm).query("Frage ohne Treffer")

    assert fake_llm.calls == []
    assert response.sources == []
    assert "keine relevanten Dokumente" in response.answer
    assert response.model == "fake-llm"


def test_chat_history_is_prepended_to_the_question(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    RAGPipeline(search, fake_llm).query(
        "Und wer gibt sie frei?",
        chat_history=[
            {"role": "user", "content": "Worum geht es in VA-0815?"},
            {"role": "assistant", "content": "Um die Lenkung von Dokumenten."},
        ],
    )

    prompt = fake_llm.calls[0]["prompt"]
    assert "Nutzer: Worum geht es in VA-0815?" in prompt
    assert "Assistent: Um die Lenkung von Dokumenten." in prompt
    assert prompt.endswith("Aktuelle Frage: Und wer gibt sie frei?")


def test_only_the_last_turns_of_the_history_are_sent(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Nachricht {i}"}
        for i in range(20)
    ]

    RAGPipeline(search, fake_llm).query("Frage", chat_history=history)

    prompt = fake_llm.calls[0]["prompt"]
    assert "Nachricht 19" in prompt
    assert "Nachricht 0" not in prompt


def test_question_is_sent_unchanged_without_history(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    RAGPipeline(search, fake_llm).query("Nur die Frage")

    assert fake_llm.calls[0]["prompt"] == "Nur die Frage"


def test_top_k_and_filters_are_forwarded_to_the_search(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    RAGPipeline(search, fake_llm).query("Frage", top_k=3, filters={"file_name": "VA-0815.pdf"})

    assert search.calls[0]["top_k"] == 3
    assert search.calls[0]["filters"] == {"file_name": "VA-0815.pdf"}


def test_streaming_yields_tokens_then_the_sources(fake_llm):
    search = _StubSearch([_result("VA-0815.pdf", "Inhalt.")])

    chunks = list(RAGPipeline(search, fake_llm).stream_query("Frage"))

    *tokens, sources = chunks
    assert all(isinstance(token, str) for token in tokens)
    assert "".join(token + " " for token in tokens).strip() == "Feste Testantwort."
    assert [source.file_name for source in sources] == ["VA-0815.pdf"]


def test_streaming_without_hits_yields_a_message_and_empty_sources(fake_llm):
    chunks = list(RAGPipeline(_StubSearch([]), fake_llm).stream_query("Frage"))

    assert "keine relevanten Dokumente" in chunks[0]
    assert chunks[-1] == []
