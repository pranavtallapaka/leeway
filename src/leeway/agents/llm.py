"""LLM provider interface for Leeway agents.

All agents call complete_json() on an LLMClient; none touch HTTP directly.
No paid API SDKs. No API keys.

Providers
---------
OllamaClient   — local Ollama (default); free, offline after first pull.
ReplayClient   — returns saved fixtures; used in tests and CI.
RecordingClient — wraps any client and writes responses to fixtures for replay.

Environment variables
---------------------
LEEWAY_MODEL   model tag for Ollama (default: qwen2.5:7b)
LEEWAY_LLM     "ollama" | "replay"  (default: "ollama")
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Fixtures directory (relative to project root)
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "agent_runs"


def _fixture_key(system: str, user: str, schema_name: str) -> str:
    """Stable SHA-256 key for a (system, user, schema) tuple."""
    blob = json.dumps({"system": system, "user": user, "schema": schema_name}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _fixture_path(key: str) -> Path:
    return FIXTURES_DIR / f"{key}.json"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Return a validated Pydantic model from the LLM, given a JSON schema."""
        ...


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

_OLLAMA_DEFAULT_URL = "http://localhost:11434/api/chat"
_OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"


class OllamaClient:
    """Calls a local Ollama instance. Free, local, no API key required."""

    def __init__(
        self,
        *,
        base_url: str = _OLLAMA_DEFAULT_URL,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model or os.environ.get("LEEWAY_MODEL", _OLLAMA_DEFAULT_MODEL)
        self.timeout = timeout
        self._http = httpx.Client(timeout=timeout)

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        json_schema = schema.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": json_schema,
            "options": {"temperature": 0},
            "stream": False,
        }

        raw = self._call(payload)
        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            # Retry once with the validation error appended
            retry_user = (
                f"{user}\n\n"
                f"Your previous response failed validation:\n{exc}\n"
                f"Return valid JSON matching the schema."
            )
            payload["messages"][-1]["content"] = retry_user
            raw = self._call(payload)
            return schema.model_validate_json(raw)

    def _call(self, payload: dict) -> str:
        response = self._http.post(self.base_url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def is_reachable(self) -> bool:
        """Return True if Ollama is responding at base_url."""
        try:
            health_url = self.base_url.rsplit("/api/chat", 1)[0]
            self._http.get(health_url, timeout=2.0)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ReplayClient
# ---------------------------------------------------------------------------


class ReplayClient:
    """Returns pre-recorded responses from fixtures/agent_runs/.

    Used for tests, CI, and the hosted Streamlit demo so no model is needed.
    Raises KeyError if a fixture for the requested (system, user, schema) is missing.
    """

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self.fixtures_dir = fixtures_dir

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        key = _fixture_key(system, user, schema.__name__)
        path = self.fixtures_dir / f"{key}.json"
        if not path.exists():
            raise KeyError(
                f"No fixture for schema={schema.__name__!r} key={key!r}.\n"
                f"Run with LEEWAY_LLM=ollama and RecordingClient to generate it."
            )
        return schema.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# RecordingClient
# ---------------------------------------------------------------------------


class RecordingClient:
    """Wraps any LLMClient and saves each response to fixtures/ for later replay."""

    def __init__(self, inner: LLMClient, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self.inner = inner
        self.fixtures_dir = fixtures_dir

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        result = self.inner.complete_json(system, user, schema)
        key = _fixture_key(system, user, schema.__name__)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        path = self.fixtures_dir / f"{key}.json"
        path.write_text(result.model_dump_json(indent=2))
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_client() -> LLMClient:
    """Return the configured LLMClient based on LEEWAY_LLM env var.

    "ollama"  (default) — OllamaClient pointing at localhost:11434
    "replay"            — ReplayClient reading from fixtures/agent_runs/
    """
    provider = os.environ.get("LEEWAY_LLM", "ollama").lower()
    if provider == "replay":
        return ReplayClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LEEWAY_LLM={provider!r}. Valid values: 'ollama', 'replay'.")
