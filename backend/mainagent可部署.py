"""
mainagent可部署.py
─────────────────────────────────────────────────────────────
向後相容的 re-export 層。

原始的所有邏輯已拆分至：
  core/registry.py      — AgentType、AGENTS_REGISTRY、TOOLS_REGISTRY、CONCEPT_TO_AGENT
  core/socrates_bot.py  — SocratesChatBot、ChatState、ConversationPhase
  core/main_agent.py    — MainAgent
  tools/base.py         — register_tool、register_langchain_tool
  tools/dsge.py         — DSGE 工具集（identify_shocks、forecast_economy …）
  tools/data_collection.py — 財務數據收集工具
  tools/knowledge.py    — knowledge_qa、trace_rc_logic

本檔只做 re-export，確保任何 `from mainagent可部署 import X` 的舊引用
繼續正常運作，不需修改 MainAgent.py / SocratesChatBot.py / entry.py。

新功能請直接在對應子模組中開發，**不要**在本檔新增邏輯。
"""

# ── core ──────────────────────────────────────────────────────
from core.registry import (  # noqa: F401
    AgentType,
    AGENTS_REGISTRY,
    CONCEPT_TO_AGENT,
    TOOLS_REGISTRY,
)
from core.socrates_bot import (  # noqa: F401
    ChatState,
    ConversationPhase,
    Neo4jReasoningRetriever,
    SocratesChatBot,
    collect_tools_with_deps,
)
from core.main_agent import (  # noqa: F401
    MainAgent,
    build_system_prompt,
    build_tools_from_state,
)

# ── tools ─────────────────────────────────────────────────────
from tools.base import (  # noqa: F401
    _TOOL_IMPLEMENTATIONS,
    register_langchain_tool,
    register_tool,
)

# 觸發所有工具的副作用式註冊（dsge / data_collection / knowledge）
import tools  # noqa: F401

# ── 個別工具（供直接 import 使用）────────────────────────────
from tools.dsge import (  # noqa: F401
    DSGEEngine,
    SWParams,
    US_STEADY_STATE,
    SHOCK_NAMES,
    SHOCK_NAMES_ZH,
    compare_scenarios,
    fetch_economic_data,
    forecast_economy,
    identify_shocks,
    scenario_analysis,
    variance_decomposition,
)
from tools.knowledge import (  # noqa: F401
    make_knowledge_qa_tool as _make_knowledge_qa_tool,
    make_trace_rc_logic_tool as _make_trace_rc_logic_tool,
)

__all__ = [
    # core — registry
    "AgentType",
    "AGENTS_REGISTRY",
    "CONCEPT_TO_AGENT",
    "TOOLS_REGISTRY",
    # core — socrates
    "ChatState",
    "ConversationPhase",
    "Neo4jReasoningRetriever",
    "SocratesChatBot",
    "collect_tools_with_deps",
    # core — main agent
    "MainAgent",
    "build_system_prompt",
    "build_tools_from_state",
    # tools — base
    "_TOOL_IMPLEMENTATIONS",
    "register_tool",
    "register_langchain_tool",
    # tools — dsge
    "DSGEEngine",
    "SWParams",
    "US_STEADY_STATE",
    "SHOCK_NAMES",
    "SHOCK_NAMES_ZH",
    "identify_shocks",
    "forecast_economy",
    "scenario_analysis",
    "variance_decomposition",
    "compare_scenarios",
    "fetch_economic_data",
]
