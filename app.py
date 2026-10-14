"""QMS-RAG: KI-gestützte Dokumentensuche für Qualitätsmanagementsysteme.

Streamlit-Einstiegspunkt. Startet die Web-Anwendung mit Chat-Interface.
Aufruf: streamlit run app.py
"""

import streamlit as st

import config
from core.logging_config import setup_logging

# Datenverzeichnisse und Logging initialisieren
config.ensure_directories()
setup_logging()

st.set_page_config(
    page_title="QMS-Dokumentensuche",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_embedding_provider():
    """Cached: Erstellt den EmbeddingProvider (Modell nur einmal laden)."""
    if "embedding_provider" not in st.session_state:
        from core.embeddings.local_provider import LocalEmbeddingProvider

        st.session_state.embedding_provider = LocalEmbeddingProvider()
    return st.session_state.embedding_provider


def _get_vector_store():
    """Cached: Erstellt den VectorStore.

    Lädt das Embedding-Modell nicht mit — das passiert erst beim ersten
    Embedding-Aufruf.
    """
    if "vector_store" not in st.session_state:
        from core.search.vector_store import VectorStore

        st.session_state.vector_store = VectorStore(_get_embedding_provider())
    return st.session_state.vector_store


def _get_rag_pipeline():
    """Cached: Erstellt die RAG-Pipeline (Suche + LLM)."""
    if "rag_pipeline" not in st.session_state:
        from core.llm import create_llm_provider
        from core.rag import RAGPipeline
        from core.search.hybrid_search import HybridSearch

        search = HybridSearch(_get_vector_store())
        llm = create_llm_provider()
        st.session_state.rag_pipeline = RAGPipeline(search, llm)
    return st.session_state.rag_pipeline


def _render_sidebar() -> None:
    """Rendert die Sidebar mit Einstellungen und Indizierung."""
    with st.sidebar:
        st.header("QMS-Dokumentensuche")

        # Status-Anzeige
        st.markdown(f"**LLM:** `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`")
        st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL.split('/')[-1]}`")

        docs_status = str(config.DOCS_PATH) if config.DOCS_PATH else "Nicht konfiguriert"
        st.markdown(f"**Dokumente:** `{docs_status}`")

        # Datenbank-Info (ohne das Embedding-Modell zu laden)
        try:
            store = _get_vector_store()
            if store.table_exists():
                st.markdown(
                    f"**Index:** {store.count_files()} Dateien, "
                    f"{store.count_chunks()} Chunks"
                )
            else:
                st.markdown("**Index:** Leer — bitte Indizierung starten")
        except Exception:
            st.markdown("**Index:** Nicht verfügbar")

        st.divider()

        # Indizierung
        if st.button("Indizierung starten", type="secondary", use_container_width=True):
            if not config.DOCS_PATH or not config.DOCS_PATH.exists():
                st.error("Dokumentpfad nicht konfiguriert oder nicht erreichbar.")
            else:
                with st.spinner("Indizierung läuft..."):
                    from core.ingestion.pipeline import IngestionPipeline

                    pipeline = IngestionPipeline(
                        _get_embedding_provider(), _get_vector_store()
                    )
                    stats = pipeline.run()

                    st.success(
                        f"{stats.new_files} neue, {stats.updated_files} aktualisiert, "
                        f"{stats.skipped_files} übersprungen — "
                        f"{stats.total_chunks} Chunks in {stats.duration_seconds:.1f}s"
                    )
                    if stats.errors:
                        with st.expander(f"{stats.failed_files} Fehler"):
                            for error in stats.errors:
                                st.warning(error)

                    # RAG-Pipeline-Cache invalidieren
                    st.session_state.pop("rag_pipeline", None)

        # Vollständige Neuindizierung
        if st.button("Komplett neu indizieren", use_container_width=True):
            if not config.DOCS_PATH or not config.DOCS_PATH.exists():
                st.error("Dokumentpfad nicht konfiguriert oder nicht erreichbar.")
            else:
                with st.spinner("Vollständige Neuindizierung..."):
                    from core.ingestion.pipeline import IngestionPipeline

                    pipeline = IngestionPipeline(
                        _get_embedding_provider(), _get_vector_store()
                    )
                    stats = pipeline.run(force=True)

                    st.success(
                        f"{stats.total_chunks} Chunks aus {stats.new_files} Dateien "
                        f"in {stats.duration_seconds:.1f}s"
                    )

                    st.session_state.pop("rag_pipeline", None)

        st.divider()

        # Chat leeren
        if st.button("Chat leeren", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources) -> None:
    """Rendert die Quellenangaben unter einer Antwort."""
    if not sources:
        return

    with st.expander(f"📄 {len(sources)} Quellen"):
        for source in sources:
            page_info = f", S. {source.page_number}" if source.page_number else ""
            section_info = f" > {source.section_title}" if source.section_title else ""
            st.markdown(f"- **{source.file_name}**{section_info}{page_info}")


def main() -> None:
    _render_sidebar()

    st.title("QMS-Dokumentensuche")

    # Chat-History initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Bisherige Nachrichten anzeigen
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                _render_sources(msg["sources"])

    # Chat-Eingabe
    if prompt := st.chat_input("Stellen Sie eine Frage zu Ihren QMS-Dokumenten..."):
        # Nutzerfrage anzeigen und speichern
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Antwort generieren
        with st.chat_message("assistant"):
            try:
                rag = _get_rag_pipeline()

                # Chat-History für Kontext aufbauen
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # Ohne aktuelle Frage
                ]

                with st.spinner("Suche in QMS-Dokumenten..."):
                    response = rag.query(
                        question=prompt,
                        chat_history=chat_history if chat_history else None,
                    )

                st.markdown(response.answer)
                _render_sources(response.sources)

                # Antwort speichern
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    }
                )

                # Token-Info in der Sidebar
                with st.sidebar:
                    st.caption(
                        f"Letzte Anfrage: {response.input_tokens} Input / "
                        f"{response.output_tokens} Output Tokens | "
                        f"{response.chunks_used} Chunks | {response.model}"
                    )

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                    st.error(
                        "API-Key ungültig oder nicht konfiguriert. "
                        "Bitte ANTHROPIC_API_KEY in der .env prüfen."
                    )
                elif "connect" in error_msg.lower():
                    st.error(
                        f"Verbindungsfehler: {error_msg}\n\n"
                        "Prüfen Sie Ihre Internetverbindung oder "
                        "wechseln Sie auf einen lokalen Provider (Ollama)."
                    )
                else:
                    st.error(f"Fehler: {error_msg}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Fehler: {error_msg}", "sources": []}
                )


if __name__ == "__main__":
    main()
