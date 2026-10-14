"""Lokaler Embedding-Provider mit sentence-transformers.

Nutzt multilingual-e5-small als Standardmodell für deutsche QMS-Dokumente.
E5-Modelle benötigen spezielle Prefixes für Queries und Dokumente.
"""

import os

import config
from core.embeddings.base import EmbeddingProvider
from core.logging_config import get_logger

logger = get_logger("embeddings.local")

# E5-Modelle benötigen diese Prefixes
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider basierend auf sentence-transformers (lokal)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self._is_e5 = "e5" in self._model_name.lower()

    def _load_model(self):
        """Lazy-Loading des Modells beim ersten Aufruf."""
        if self._model is None:
            logger.info("Lade Embedding-Modell: %s", self._model_name)

            # Offline-Modus: Keine HuggingFace-Requests wenn Modell lokal gecached ist
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

            from sentence_transformers import SentenceTransformer

            # local_files_only=True vermeidet HTTP-Requests bei gecachtem Modell
            try:
                self._model = SentenceTransformer(
                    self._model_name, local_files_only=True
                )
            except OSError:
                # Erster Start: Modell muss heruntergeladen werden
                logger.info("Modell nicht lokal — lade von HuggingFace herunter...")
                self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding-Modell geladen: %s (Dimension: %d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()

        # E5-Modelle benötigen "passage: " Prefix für Dokumente
        if self._is_e5:
            texts = [f"{_E5_PASSAGE_PREFIX}{t}" for t in texts]

        embeddings = model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()

        # E5-Modelle benötigen "query: " Prefix für Suchanfragen
        if self._is_e5:
            query = f"{_E5_QUERY_PREFIX}{query}"

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    def model_name(self) -> str:
        return self._model_name
