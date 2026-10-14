"""Logging-Konfiguration für QMS-RAG.

Konfiguriert Rotating File Handlers für verschiedene Log-Kategorien
und einen Console Handler für die Entwicklung.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import config


def setup_logging() -> None:
    """Initialisiert das Logging-System mit File- und Console-Handlern."""
    config.ensure_directories()
    log_dir = config.LOG_DIR
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # Root-Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Bestehende Handler entfernen (bei mehrfachem Aufruf)
    root_logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handlers (je 10 MB, 5 Backup-Dateien)
    log_files = {
        "app": "application.log",
        "search": "search.log",
        "ingestion": "ingestion.log",
    }

    for name, filename in log_files.items():
        file_handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Spezialisierte Logger erstellen
        logger = logging.getLogger(f"qms.{name}")
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Gibt einen Logger für das angegebene Modul zurück.

    Args:
        name: Modulname, z.B. "parser.pdf" -> Logger "qms.app.parser.pdf"
    """
    return logging.getLogger(f"qms.app.{name}")


def get_search_logger() -> logging.Logger:
    """Logger für Suchanfragen."""
    return logging.getLogger("qms.search")


def get_ingestion_logger() -> logging.Logger:
    """Logger für den Indizierungsprozess."""
    return logging.getLogger("qms.ingestion")

