"""CLI-backed LLM client.

Runs TradingAgents on a local CLI tool (Claude Code / Gemini CLI / Codex CLI)
instead of a metered API — i.e. on your existing subscription. Select with
``llm_provider: "cli"`` and set the model to a backend name: ``claude`` /
``gemini`` / ``codex`` / ``qwen``.
"""
from typing import Any, Optional

from .base_client import BaseLLMClient
from .cli_chat_model import BACKENDS, CLIChatModel


class CLIClient(BaseLLMClient):
    """Client that delegates to a local LLM/coding CLI."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        timeout = int(self.kwargs.get("timeout", 300) or 300)
        return CLIChatModel(backend=self.model.lower(), timeout=timeout)

    def validate_model(self) -> bool:
        return self.model.lower() in BACKENDS
