"""CLIChatModel — a LangChain chat model backed by a local CLI tool.

Instead of calling a metered LLM API, this shells out to an installed
coding/LLM CLI (Claude Code, Gemini CLI, Codex CLI, …) running on your
existing subscription. Lets TradingAgents run with (near-)zero API cost.

Backend = a command template. Add a CLI by adding one entry to ``BACKENDS``.

Tool calling: CLIs are text-in/text-out and don't speak LangChain's native
function-calling protocol. We bridge it with a small TEXT protocol — the model
is asked to emit a JSON ``{"tool": name, "args": {...}}`` to call a tool, which
we turn into a real ``AIMessage.tool_calls`` so LangGraph's ToolNode executes
the tool and loops the result back. This lets the data-fetching analyst agents
run on real data over a CLI backend.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
        content = m.content
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        parts.append(f"{role}: {content}")
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


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _extract_tool_call(raw: str) -> Optional[dict]:
    """Find a {"tool": ..., "args": {...}} object in the model's text output."""
    s = _strip_fences(raw)
    candidates = [s]
    # also try the first {...} blob anywhere in the text
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return {"tool": obj["tool"], "args": obj.get("args") or {}}
    return None


def _tool_docs(tools: List[Any]) -> str:
    lines = []
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", "tool")
        desc = (getattr(t, "description", "") or "").strip().replace("\n", " ")
        args = getattr(t, "args", None)
        arg_names = list(args.keys()) if isinstance(args, dict) else []
        lines.append(f'- {name}(args: {arg_names}): {desc[:300]}')
    return "\n".join(lines)


class CLIChatModel(BaseChatModel):
    """LangChain chat model that delegates to a local LLM/coding CLI."""

    backend: str = "claude"
    timeout: int = 300
    bound_tools: Optional[List[Any]] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return f"cli-{self.backend}"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "CLIChatModel":
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.bound_tools:
            return self._generate_with_tools(messages)
        text = run_cli(self.backend, _render(messages), self.timeout)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate_with_tools(self, messages: List[BaseMessage]) -> ChatResult:
        instr = (
            "\n\n---\nYou may call ONE tool to fetch data. Available tools:\n"
            + _tool_docs(self.bound_tools)
            + "\n\nTo CALL A TOOL, respond with ONLY this JSON and nothing else:\n"
            '{"tool": "<tool_name>", "args": {"<arg>": <value>}}\n'
            "Fill args using the ticker/date/context above. Call tools one at a "
            "time; you'll see each result, then may call another or write your "
            "final report.\nWhen you are DONE fetching data, write your final "
            "report as normal prose (NOT JSON)."
        )
        raw = run_cli(self.backend, _render(messages) + instr, self.timeout)
        call = _extract_tool_call(raw)
        if call:
            tc = {"name": call["tool"], "args": call["args"],
                  "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"}
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="", tool_calls=[tc]))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=raw))])

    # Best-effort structured output: ask for JSON, parse into the schema.
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def _parse(messages):
            hint = ("\n\nRespond with ONLY a valid JSON object, no prose, "
                    "no markdown fences.")
            prompt = _render(messages) + hint
            raw = _strip_fences(run_cli(self.backend, prompt, self.timeout))
            data = json.loads(raw)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            if hasattr(schema, "parse_obj"):
                return schema.parse_obj(data)
            return data
        return RunnableLambda(_parse)
