"""Tests der LLM-Provider und der Provider-Factory.

Es wird keine API aufgerufen: der Anthropic-Client ist ein Fake, der
Ollama-HTTP-Aufruf ist gepatcht.
"""

import io
import json
from types import SimpleNamespace

import pytest

import config
from core.llm import create_llm_provider
from core.llm.claude_provider import ClaudeProvider
from core.llm.ollama_provider import OllamaProvider

# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def test_factory_builds_the_claude_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    assert isinstance(create_llm_provider(), ClaudeProvider)


def test_factory_builds_the_ollama_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")

    assert isinstance(create_llm_provider(), OllamaProvider)


def test_factory_ignores_case(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "OLLAMA")

    assert isinstance(create_llm_provider(), OllamaProvider)


def test_factory_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "orakel")

    with pytest.raises(ValueError, match="Unbekannter LLM-Provider"):
        create_llm_provider()


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

class _FakeMessages:
    def __init__(self, text: str = "Antwort des Modells") -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self._text)],
            usage=SimpleNamespace(input_tokens=120, output_tokens=42),
        )


class _FakeClient:
    def __init__(self, text: str = "Antwort des Modells") -> None:
        self.messages = _FakeMessages(text)


def test_claude_provider_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider()


def test_claude_provider_returns_text_and_token_counts():
    client = _FakeClient()

    response = ClaudeProvider(model="test-model", client=client).generate(
        "Wer gibt Dokumente frei?", context="Freigabe durch die Leitung."
    )

    assert response.text == "Antwort des Modells"
    assert response.model == "test-model"
    assert response.input_tokens == 120
    assert response.output_tokens == 42


def test_claude_provider_sends_context_and_question_in_one_message():
    client = _FakeClient()

    ClaudeProvider(model="test-model", client=client).generate(
        "Wer gibt frei?", context="Freigabe durch die Leitung.", system="Sei praezise."
    )

    call = client.messages.calls[0]
    assert call["system"] == "Sei praezise."
    assert call["model"] == "test-model"
    assert len(call["messages"]) == 1
    content = call["messages"][0]["content"]
    assert "Freigabe durch die Leitung." in content
    assert "Frage: Wer gibt frei?" in content


def test_claude_provider_sends_the_bare_question_without_context():
    client = _FakeClient()

    ClaudeProvider(model="test-model", client=client).generate("Nur die Frage")

    assert client.messages.calls[0]["messages"][0]["content"] == "Nur die Frage"


def test_claude_provider_joins_multiple_text_blocks():
    client = _FakeClient()
    client.messages.create = lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(text="Teil 1 "), SimpleNamespace(text="Teil 2")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    response = ClaudeProvider(model="test-model", client=client).generate("Frage")

    assert response.text == "Teil 1 Teil 2"


def test_claude_provider_reports_its_model_name():
    assert ClaudeProvider(model="test-model", client=_FakeClient()).model_name() == "test-model"


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

def _patch_urlopen(monkeypatch, payload: dict | list[dict], recorder: list):
    """Ersetzt urlopen und schreibt die abgesendeten Requests mit."""
    import urllib.request

    lines = payload if isinstance(payload, list) else [payload]
    body = b"".join(json.dumps(line).encode("utf-8") + b"\n" for line in lines)

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            return False

    def fake_urlopen(request, timeout=None):
        recorder.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")) if request.data else None,
                "timeout": timeout,
            }
        )
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_ollama_provider_returns_text_and_token_counts(monkeypatch):
    sent: list[dict] = []
    _patch_urlopen(
        monkeypatch,
        {
            "message": {"content": "Lokale Antwort"},
            "prompt_eval_count": 70,
            "eval_count": 15,
        },
        sent,
    )

    response = OllamaProvider(base_url="http://localhost:11434", model="llama3.1:8b").generate(
        "Frage", context="Kontext"
    )

    assert response.text == "Lokale Antwort"
    assert response.model == "llama3.1:8b"
    assert response.input_tokens == 70
    assert response.output_tokens == 15


def test_ollama_provider_sends_system_prompt_as_its_own_message(monkeypatch):
    sent: list[dict] = []
    _patch_urlopen(monkeypatch, {"message": {"content": "ok"}}, sent)

    OllamaProvider(model="llama3.1:8b").generate(
        "Frage", context="Kontext", system="Sei praezise."
    )

    messages = sent[0]["body"]["messages"]
    assert messages[0] == {"role": "system", "content": "Sei praezise."}
    assert messages[1]["role"] == "user"
    assert "Kontext" in messages[1]["content"]


def test_ollama_provider_targets_the_chat_endpoint_with_a_timeout(monkeypatch):
    sent: list[dict] = []
    _patch_urlopen(monkeypatch, {"message": {"content": "ok"}}, sent)

    OllamaProvider(base_url="http://localhost:11434/").generate("Frage")

    assert sent[0]["url"] == "http://localhost:11434/api/chat"
    assert sent[0]["timeout"] is not None


def test_ollama_provider_streams_token_by_token(monkeypatch):
    sent: list[dict] = []
    _patch_urlopen(
        monkeypatch,
        [
            {"message": {"content": "Lokale "}, "done": False},
            {"message": {"content": "Antwort"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ],
        sent,
    )

    tokens = list(OllamaProvider().stream("Frage"))

    assert tokens == ["Lokale ", "Antwort"]
    assert sent[0]["body"]["stream"] is True


def test_ollama_provider_reports_an_unreachable_server(monkeypatch):
    import urllib.error
    import urllib.request

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match="nicht erreichbar"):
        OllamaProvider().generate("Frage")


def test_ollama_availability_check_does_not_raise(monkeypatch):
    import urllib.error
    import urllib.request

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert OllamaProvider().is_available() is False
