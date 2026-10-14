"""Tests der Konfiguration und des Logging-Setups."""

import importlib
import logging

import config
from core.logging_config import (
    get_ingestion_logger,
    get_logger,
    get_search_logger,
    setup_logging,
)


def test_importing_config_creates_no_directories(tmp_path, monkeypatch):
    """config darf beim Import nichts auf die Platte schreiben."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "frisch"))

    try:
        data_dir = importlib.reload(config).DATA_DIR

        assert data_dir == tmp_path / "frisch"
        assert not data_dir.exists()
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)


def test_ensure_directories_creates_data_and_log_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "daten"))
    reloaded = importlib.reload(config)

    try:
        reloaded.ensure_directories()

        assert reloaded.LANCEDB_DIR.is_dir()
        assert reloaded.LOG_DIR.is_dir()
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)


def test_environment_overrides_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_TOP_K", "3")
    monkeypatch.setenv("HYBRID_VECTOR_WEIGHT", "0.25")
    monkeypatch.setenv("DOCS_PATH", str(tmp_path / "qms"))

    reloaded = importlib.reload(config)
    try:
        assert reloaded.SEARCH_TOP_K == 3
        assert reloaded.HYBRID_VECTOR_WEIGHT == 0.25
        assert tmp_path / "qms" == reloaded.DOCS_PATH
    finally:
        for name in ("SEARCH_TOP_K", "HYBRID_VECTOR_WEIGHT", "DOCS_PATH"):
            monkeypatch.delenv(name)
        importlib.reload(config)


def test_default_configuration_is_consistent():
    assert config.MIN_CHUNK_SIZE < config.MAX_CHUNK_SIZE
    assert 0.0 <= config.HYBRID_VECTOR_WEIGHT <= 1.0
    assert config.SEARCH_TOP_K > 0


def test_setup_logging_writes_the_category_log_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "logs-test"))
    reloaded = importlib.reload(config)

    try:
        import core.logging_config as logging_config

        importlib.reload(logging_config)
        logging_config.setup_logging()

        logging_config.get_search_logger().info("Suchanfrage")
        logging_config.get_ingestion_logger().info("Indizierung")

        for handler in logging.getLogger().handlers:
            handler.flush()
        for name in logging.Logger.manager.loggerDict:
            if name.startswith("qms."):
                for handler in logging.getLogger(name).handlers:
                    handler.flush()

        assert "Suchanfrage" in (reloaded.LOG_DIR / "search.log").read_text(encoding="utf-8")
        assert "Indizierung" in (reloaded.LOG_DIR / "ingestion.log").read_text(encoding="utf-8")
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)
        import core.logging_config as logging_config

        importlib.reload(logging_config)
        logging.getLogger().handlers.clear()


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "idempotent"))
    importlib.reload(config)

    try:
        import core.logging_config as logging_config

        importlib.reload(logging_config)
        logging_config.setup_logging()
        first = len(logging.getLogger().handlers)
        logging_config.setup_logging()

        assert len(logging.getLogger().handlers) == first
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)
        logging.getLogger().handlers.clear()


def test_module_loggers_live_under_the_qms_namespace():
    assert get_logger("parser.pdf").name == "qms.app.parser.pdf"
    assert get_search_logger().name == "qms.search"
    assert get_ingestion_logger().name == "qms.ingestion"
    assert callable(setup_logging)
