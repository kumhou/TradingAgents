# Running TradingAgents on local CLIs (no API bills)

This fork adds a **`cli` LLM provider** so TradingAgents can run on installed
LLM/coding CLIs — **Claude Code**, **Gemini CLI**, **Codex CLI**, **Qwen Code** —
which run on your existing **subscription** instead of metered APIs. Each
agent call shells out to the CLI instead of hitting a paid endpoint.

> Why: the multi-agent pipeline makes many LLM calls per run → expensive on
> metered APIs. Routing to a subscription CLI makes a run (near-)free.

## Requirements
- The CLI you want must be installed and logged in:
  - Claude Code: already a `claude` on PATH (`claude -p` works)
  - Gemini CLI: `npm i -g @google/gemini-cli`
  - Codex CLI: `npm i -g @openai/codex`
  - Qwen Code: `npm i -g @qwen-code/qwen-code` (or its current package)
- No API key needed for the LLM (data still uses free Yahoo Finance).

## How to run

### A. Interactive CLI
```
python -m cli.main
```
At the provider prompt pick **"Local CLI (Claude/Gemini/Codex …)"**, then pick a
backend (`claude` / `gemini` / `codex` / `qwen`) for both quick and deep models.

### B. Environment variables
```
set TRADINGAGENTS_LLM_PROVIDER=cli
set TRADINGAGENTS_DEEP_THINK_LLM=claude
set TRADINGAGENTS_QUICK_THINK_LLM=claude
python -m cli.main
```

### C. Programmatic
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG.copy()
cfg["llm_provider"] = "cli"
cfg["deep_think_llm"] = "claude"   # or gemini / codex / qwen
cfg["quick_think_llm"] = "claude"

ta = TradingAgentsGraph(selected_analysts=["market"], config=cfg)
state, decision = ta.propagate("NVDA", "2024-11-01")
print(decision)
```

## Add another CLI
Edit `BACKENDS` in `tradingagents/llm_clients/cli_chat_model.py`:
```python
"mybackend": {"cmd": ["mycli", "--print"], "mode": "stdin"},  # or "arg"
```
and add it to `_CLI_OPTIONS` in `llm_clients/model_catalog.py`.

## Honest limitations
CLIs are **text-in / text-out** — they don't speak LangChain's native
tool-calling or structured-output protocols:
- `bind_tools` is a **no-op** → the data-fetching **analyst** agents run in a
  *degraded* mode (they reason without live tool fetches). The
  reasoning/debate/research/trader agents work fully.
- `with_structured_output` uses an **"ask for JSON + parse"** fallback (works,
  verified for the trader / research-manager / portfolio-manager paths).

For full data-driven analysts, either use a real API provider for those agents,
or refactor data-fetching to run in Python and inject results into the prompt
(planned next step).

## Status
Verified end-to-end on the `claude` backend: full pipeline runs and produces a
final decision (e.g. `NVDA → Overweight`). Multi-model debate (a *different* CLI
per agent role) is a planned enhancement.
