"""Leeway agents package.

Autonomous decision agents that select contingency plans.
Agents may never call the executor or send commands.
All agents receive an LLMClient and never call HTTP directly.

Phase A status: stub only.
"""

from leeway.agents.llm import LLMClient, OllamaClient, RecordingClient, ReplayClient, get_client

__all__ = ["LLMClient", "OllamaClient", "ReplayClient", "RecordingClient", "get_client"]
