"""CLIChatModel — a LangChain chat model backed by a local CLI tool.

Instead of calling a metered LLM API, this shells out to an installed
coding/LLM CLI (Claude Code, Gemini CLI, Codex CLI, …) running on your
existing subscription. Lets TradingAgents run with (near-)zero API cost.

Backend = a command template. Add a CLI by adding one entry to ``BACKENDS``.

Limitations (honest):
- Plain text in / text out. Native tool-calling and structured-output
  protocols are NOT supported by these CLIs. ``bind_tools`` is a no-op and
  ``with_structured_output`` falls back to "ask for JSON + parse". So the
  reasoning/debate/trader agents work well; data-fetching analyst agents
  that rely on tool calls run in a degraded (no live tool) mode.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

# backend name -> how to invoke it headlessly
# mode: "stdin" = prompt piped to stdin; "arg" = prompt as final argv
BACKENDS = {
    "claude": {"cmd": ["claude", "-p"], "mode": "stdin"},
    "gemini": {"cmd": ["gemini", "-p"], "mode": "arg"},
    "codex":  {"cmd": ["codex", "exec"], "mode": "arg"},
    "qwen":   {"cmd": ["qwen", "-p"], "mode": "arg"},
}

_ROLE = {"system": "System", "human": "User", "ai": "Assistant", "tool": "Tool"}


def _to_messages(x: Any) -> List[BaseMessage]:
    """Normalize any LangChain invoke input (str / message / list / PromptValue)
    into a list of BaseMessage."""
    from langchain_core.messages import HumanMessage
    if isinstance(x, str):
        return [HumanMessage(content=x)]
    if isinstance(x, BaseMessage):
        return [x]
    if hasattr(x, "to_messages"):
        return x.to_messages()
    if isinstance(x, list):
        out: List[BaseMessage] = []
        for i in x:
            if isinstance(i, BaseMessage):
                out.append(i)
            elif isinstance(i, str):
                out.append(HumanMessage(content=i))
            elif isinstance(i, dict) and "content" in i:
                out.append(HumanMessage(content=str(i["content"])))
        return out or [HumanMessage(content=str(x))]
    return [HumanMessage(content=str(x))]


def _render(messages: Any) -> str:
    parts = []
    for m in _to_messages(messages):
        role = _ROLE.get(m.type, m.type.capitalize())
        parts.append(f"{role}: {m.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def run_cli(backend: str, prompt: str, timeout: int = 300) -> str:
    spec = BACKENDS.get(backend)
    if not spec:
        raise ValueError(f"unknown CLI backend '{backend}'. known: {list(BACKENDS)}")
    exe = spec["cmd"][0]
    if not shutil.which(exe):
        raise RuntimeError(f"CLI '{exe}' not installed/on PATH for backend '{backend}'")
    kw = dict(capture_output=True, text=True, timeout=timeout,
              encoding="utf-8", errors="replace")
    if spec["mode"] == "stdin":
        out = subprocess.run(spec["cmd"], input=prompt, **kw)
    else:
        out = subprocess.run(spec["cmd"] + [prompt], **kw)
    if out.returncode != 0:
        raise RuntimeError(f"{backend} CLI error: {(out.stderr or '').strip()[:500]}")
    return (out.stdout or "").strip()


class CLIChatModel(BaseChatModel):
    """LangChain chat model that delegates to a local CLI tool."""

    backend: str = "claude"
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return f"cli-{self.backend}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = run_cli(self.backend, _render(messages), self.timeout)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    # CLIs can't do native tool-calling — ignore tools so agents still run.
    def bind_tools(self, tools: Any, **kwargs: Any) -> "CLIChatModel":
        return self

    # Best-effort structured output: ask for JSON, parse into the schema.
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def _parse(messages):
            hint = ("\n\nRespond with ONLY a valid JSON object, no prose, "
                    "no markdown fences.")
            prompt = _render(messages) + hint
            raw = run_cli(self.backend, prompt, self.timeout)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            if hasattr(schema, "parse_obj"):
                return schema.parse_obj(data)
            return data
        return RunnableLambda(_parse)
