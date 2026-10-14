"""Ollama LLM-Provider für lokale Modelle.

Kommuniziert mit einem lokalen Ollama-Server über die REST API.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Generator

import config
from core.llm.base import LLMProvider, LLMResponse
from core.logging_config import get_logger

logger = get_logger("llm.ollama")

# Lokale Modelle brauchen auf CPU durchaus eine Minute pro Antwort.
_REQUEST_TIMEOUT_SECONDS = 300


class OllamaProvider(LLMProvider):
    """LLM-Provider für lokalen Ollama-Server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._model = model or config.OLLAMA_MODEL

    def generate(self, prompt: str, context: str = "", system: str = "") -> LLMResponse:
        messages = self._build_messages(prompt, context, system)

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Ollama-Server nicht erreichbar unter {self._base_url}: {e}"
            ) from e

        text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        logger.info(
            "Ollama Antwort (%s): %d Prompt-Tokens, %d Output-Tokens",
            self._model,
            prompt_eval_count,
            eval_count,
        )

        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
        )

    def stream(
        self, prompt: str, context: str = "", system: str = ""
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, context, system)

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                for line in resp:
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            break
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Ollama-Server nicht erreichbar unter {self._base_url}: {e}"
            ) from e

    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                resp.read()
            return True
        except Exception as e:
            logger.debug("Ollama nicht erreichbar: %s", e)
            return False

    def _build_messages(
        self, prompt: str, context: str, system: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        if context:
            user_content = (
                f"Kontext aus QMS-Dokumenten:\n\n{context}\n\n"
                f"---\n\nFrage: {prompt}"
            )
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})
        return messages
