"""Tests for the LLM provider layer.

All tests use ReplayClient or in-memory fixtures; Ollama is never required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from leeway.agents.llm import (
    OllamaClient,
    RecordingClient,
    ReplayClient,
    _fixture_key,
    get_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Greeting(BaseModel):
    message: str


_SYSTEM = "You are a test assistant."
_USER = "Say hello."
_KEY = _fixture_key(_SYSTEM, _USER, _Greeting.__name__)


def _write_fixture(directory: Path, model: BaseModel) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_fixture_key(_SYSTEM, _USER, type(model).__name__)}.json"
    path.write_text(model.model_dump_json())
    return path


# ---------------------------------------------------------------------------
# _fixture_key
# ---------------------------------------------------------------------------


def test_fixture_key_is_deterministic():
    k1 = _fixture_key("s", "u", "M")
    k2 = _fixture_key("s", "u", "M")
    assert k1 == k2


def test_fixture_key_differs_by_field():
    assert _fixture_key("s1", "u", "M") != _fixture_key("s2", "u", "M")
    assert _fixture_key("s", "u1", "M") != _fixture_key("s", "u2", "M")
    assert _fixture_key("s", "u", "M1") != _fixture_key("s", "u", "M2")


# ---------------------------------------------------------------------------
# ReplayClient
# ---------------------------------------------------------------------------


def test_replay_client_returns_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        fixtures_dir = Path(tmp)
        _write_fixture(fixtures_dir, _Greeting(message="hello"))
        client = ReplayClient(fixtures_dir=fixtures_dir)
        result = client.complete_json(_SYSTEM, _USER, _Greeting)
    assert isinstance(result, _Greeting)
    assert result.message == "hello"


def test_replay_client_raises_on_missing_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        client = ReplayClient(fixtures_dir=Path(tmp))
        with pytest.raises(KeyError, match="No fixture"):
            client.complete_json(_SYSTEM, _USER, _Greeting)


# ---------------------------------------------------------------------------
# RecordingClient
# ---------------------------------------------------------------------------


class _EchoClient:
    """Fake inner client that returns a fixed response."""

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        return _Greeting(message="recorded")


def test_recording_client_writes_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        fixtures_dir = Path(tmp)
        recorder = RecordingClient(_EchoClient(), fixtures_dir=fixtures_dir)
        result = recorder.complete_json(_SYSTEM, _USER, _Greeting)

        assert result.message == "recorded"
        written = fixtures_dir / f"{_KEY}.json"
        assert written.exists()
        assert json.loads(written.read_text())["message"] == "recorded"


def test_recording_then_replay_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        fixtures_dir = Path(tmp)
        RecordingClient(_EchoClient(), fixtures_dir=fixtures_dir).complete_json(
            _SYSTEM, _USER, _Greeting
        )
        replayed = ReplayClient(fixtures_dir=fixtures_dir).complete_json(
            _SYSTEM, _USER, _Greeting
        )
    assert replayed.message == "recorded"


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


def test_get_client_default_is_ollama(monkeypatch):
    monkeypatch.delenv("LEEWAY_LLM", raising=False)
    client = get_client()
    assert isinstance(client, OllamaClient)


def test_get_client_replay(monkeypatch):
    monkeypatch.setenv("LEEWAY_LLM", "replay")
    client = get_client()
    assert isinstance(client, ReplayClient)


def test_get_client_ollama_explicit(monkeypatch):
    monkeypatch.setenv("LEEWAY_LLM", "ollama")
    client = get_client()
    assert isinstance(client, OllamaClient)


def test_get_client_unknown_raises(monkeypatch):
    monkeypatch.setenv("LEEWAY_LLM", "openai")
    with pytest.raises(ValueError, match="Unknown LEEWAY_LLM"):
        get_client()


# ---------------------------------------------------------------------------
# OllamaClient model selection
# ---------------------------------------------------------------------------


def test_ollama_client_respects_env_model(monkeypatch):
    monkeypatch.setenv("LEEWAY_MODEL", "llama3.2:3b")
    client = OllamaClient()
    assert client.model == "llama3.2:3b"


def test_ollama_client_default_model(monkeypatch):
    monkeypatch.delenv("LEEWAY_MODEL", raising=False)
    client = OllamaClient()
    assert client.model == "qwen2.5:7b"


def test_ollama_client_explicit_model_overrides_env(monkeypatch):
    monkeypatch.setenv("LEEWAY_MODEL", "llama3.2:3b")
    client = OllamaClient(model="mistral:7b")
    assert client.model == "mistral:7b"
