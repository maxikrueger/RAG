"""LLM Provider Package.

Stellt eine Factory-Funktion bereit, die den konfigurierten Provider erstellt.
"""

import config
from core.llm.base import LLMProvider, LLMResponse


def create_llm_provider() -> LLMProvider:
    """Erstellt den LLM-Provider basierend auf der Konfiguration.

    Liest LLM_PROVIDER aus config.py und erstellt die passende Implementierung.

    Returns:
        Konfigurierter LLMProvider.

    Raises:
        ValueError: Unbekannter Provider-Typ.
    """
    provider_type = config.LLM_PROVIDER.lower()

    if provider_type == "claude":
        from core.llm.claude_provider import ClaudeProvider
        return ClaudeProvider()
    elif provider_type == "ollama":
        from core.llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unbekannter LLM-Provider: '{provider_type}'. "
            f"Unterstützt: 'claude', 'ollama'"
        )


__all__ = ["LLMProvider", "LLMResponse", "create_llm_provider"]
