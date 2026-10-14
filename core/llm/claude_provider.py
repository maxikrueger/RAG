"""Claude LLM-Provider über die Anthropic API.

Unterstützt synchrone Generierung und Streaming.
"""

from collections.abc import Generator

import anthropic

import config
from core.llm.base import LLMProvider, LLMResponse
from core.logging_config import get_logger

logger = get_logger("llm.claude")


class ClaudeProvider(LLMProvider):
    """LLM-Provider für Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: object | None = None,
    ) -> None:
        """Erstellt den Provider.

        Args:
            api_key: API-Key. Standard: config.ANTHROPIC_API_KEY.
            model: Modellname. Standard: config.LLM_MODEL.
            client: Vorkonfigurierter Client (für Tests).

        Raises:
            ValueError: Kein API-Key konfiguriert.
        """
        self._model = model or config.LLM_MODEL

        if client is not None:
            self._client = client
            return

        self._api_key = api_key or config.ANTHROPIC_API_KEY
        if not self._api_key:
            raise ValueError(
                "Kein ANTHROPIC_API_KEY konfiguriert. Key in der .env setzen "
                "oder mit LLM_PROVIDER=ollama einen lokalen Provider nutzen."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, prompt: str, context: str = "", system: str = "") -> LLMResponse:
        messages = self._build_messages(prompt, context)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system or "",
            messages=messages,
        )

        text = self._extract_text(response)
        logger.info(
            "Claude Antwort: %d Input-Tokens, %d Output-Tokens",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(
        self, prompt: str, context: str = "", system: str = ""
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, context)

        with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            system=system or "",
            messages=messages,
        ) as stream:
            yield from stream.text_stream

    @staticmethod
    def _extract_text(response) -> str:
        """Fügt die Text-Blöcke einer Antwort zusammen."""
        return "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )

    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning("Claude API nicht erreichbar: %s", e)
            return False

    def _build_messages(
        self, prompt: str, context: str
    ) -> list[dict[str, str]]:
        if context:
            user_content = (
                f"Kontext aus QMS-Dokumenten:\n\n{context}\n\n"
                f"---\n\nFrage: {prompt}"
            )
        else:
            user_content = prompt

        return [{"role": "user", "content": user_content}]
