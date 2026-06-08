# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG.copy()
cfg.update(llm_provider="cli", deep_think_llm="claude", quick_think_llm="claude",
           max_debate_rounds=1, max_risk_discuss_rounds=1, online_tools=True)

ta = TradingAgentsGraph(selected_analysts=["market"], debug=False, config=cfg)
state, decision = ta.propagate("ZEC", "2026-06-07", asset_type="crypto")


def show(title, txt):
    print(f"\n##### {title} #####\n{str(txt)[:1600]}")


show("MARKET REPORT", state.get("market_report", ""))
ids = state.get("investment_debate_state", {})
show("BULL", ids.get("bull_history", ""))
show("BEAR", ids.get("bear_history", ""))
show("RESEARCH MANAGER (judge)", ids.get("judge_decision", ""))
show("TRADER PLAN", state.get("trader_investment_plan", ""))
show("FINAL DECISION", decision)
print("\n[DONE]")
