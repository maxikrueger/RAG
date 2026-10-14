"""Zentrale Konfiguration für QMS-RAG.

Alle Einstellungen werden aus Umgebungsvariablen / .env geladen.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env laden (falls vorhanden)
load_dotenv()

# =============================================================================
# Pfade
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
LANCEDB_DIR = DATA_DIR / "lancedb"
LOG_DIR = DATA_DIR / "logs"

# Wurzelverzeichnis der QMS-Dokumente (lokaler Ordner oder Netzlaufwerk)
DOCS_PATH = Path(os.getenv("DOCS_PATH", ""))


def ensure_directories() -> None:
    """Erstellt die lokalen Datenverzeichnisse, falls sie fehlen.

    Wird beim Start der Anwendung aufgerufen, nicht beim Import: so bleibt
    das Importieren von config nebenwirkungsfrei (z.B. in Tests).
    """
    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LLM Provider
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")  # "claude" | "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# =============================================================================
# Embedding Provider
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# =============================================================================
# Chunking
# =============================================================================
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "800"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# =============================================================================
# Suche
# =============================================================================
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))

# =============================================================================
# Streamlit
# =============================================================================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# Unterstützte Dateiformate
# =============================================================================
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".csv", ".html", ".htm", ".txt",
}

# =============================================================================
# LanceDB Tabellen
# =============================================================================
CHUNKS_TABLE = "qms_chunks"
