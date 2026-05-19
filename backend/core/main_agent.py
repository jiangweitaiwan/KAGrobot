"""
core/main_agent.py
─────────────────────────────────────────────────────────────
MainAgent：無狀態設計，每次 chat() 呼叫動態組裝 LangGraph 執行圖。

Skill 注入
----------
當 chat() 收到 skill: RCSkill 參數時：
  - system prompt = BASE_SYSTEM_PROMPT + skill.prompt_context
  - tools         = skill.tools（若 skill.has_tools）
                    否則 fallback 到 build_tools_from_state()

對外介面：
    MainAgent   — 主類別
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from core.google_llm import create_chat_google_generative_ai
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated

from core.llm_retry import call_with_retry
from core.registry import AGENTS_REGISTRY, AgentType, TOOLS_REGISTRY
from core.socrates_bot import (
    ChatState,
    ConversationPhase,
    _extract_selected_chain,
    _format_selected_chain,
    _resolve_agent_type,
    collect_tools_with_deps,
)
from tools.base import _TOOL_IMPLEMENTATIONS
from tools.knowledge import make_knowledge_qa_tool, make_trace_rc_logic_tool

# 確保所有工具副作用式註冊已執行
import tools  # noqa: F401

# RCSkill type（避免循環 import，用 TYPE_CHECKING）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skill_loader import RCSkill


# ═══════════════════════════════════════════════════════════════
# LangGraph Agent State
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ═══════════════════════════════════════════════════════════════
# Base system prompt（固定，定位在「協助解決用戶 query」）
# ═══════════════════════════════════════════════════════════════

_BASE_SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位專業的金融與經濟分析助手。
    你的核心任務是：理解使用者的分析需求，運用可用工具，提供清晰、有深度的分析結果。

    ## 使用原則

    - **主動判斷**：根據需求與可用工具，主動決定呼叫哪些工具、以何種順序組合
    - **結果解讀**：收到工具結果後，以繁體中文清楚解讀，結合理論背景提供深度分析
    - **參數確認**：若執行工具所需的參數不完整，先向使用者詢問，再執行
    - **回答結構**：先給結論，再解釋原因；專業但易懂，避免過度學術術語
    - **不確定性**：若分析結果存在前提假設或限制，主動向使用者說明
""")


# ═══════════════════════════════════════════════════════════════
# Skill-aware prompt builder
# ═══════════════════════════════════════════════════════════════

def build_system_prompt(
    socrates_state: ChatState,
    skill: Optional["RCSkill"] = None,
) -> str:
    """組裝完整 system prompt。

    結構：
        BASE_SYSTEM_PROMPT          （固定）
        ── 使用者需求 ──             （來自 socrates_state）
        ── RC Skill Context ──      （來自 skill，若有）
        ── 可用工具清單 ──           （skill.tools 或 fallback）
    """
    query = socrates_state.get("query") or "（未提供）"

    # ── 使用者需求區塊 ──
    need_section = textwrap.dedent(f"""\
        ══════════════════════════════
        使用者需求
        ══════════════════════════════
        {query}
    """)

    # ── RC Skill Context 區塊（動態）──
    if skill and skill.prompt_context.strip():
        skill_section = textwrap.dedent(f"""\

        ══════════════════════════════
        分析方法：{skill.rc_name}
        ══════════════════════════════
        {skill.prompt_context.strip()}
        """)
    else:
        skill_section = ""

    # ── 工具清單區塊 ──
    if skill and skill.has_tools:
        # 工具來自 skill
        tool_lines = [
            f"{i}. **{t.name}** — {t.description}"
            for i, t in enumerate(skill.tools, 1)
        ]
        tool_section_header = "可用工具（來自 RC Skill）"
    else:
        # Fallback：從 socrates_state 組裝（舊機制）
        tool_lines = _build_fallback_tool_lines(socrates_state)
        tool_section_header = "可用工具"

    tool_section = (
        f"══════════════════════════════\n"
        f"{tool_section_header}\n"
        f"══════════════════════════════\n"
        + ("\n".join(tool_lines) if tool_lines else "（無額外工具）")
    )

    # 常駐工具提示
    trace_hint = (
        "\n此外，你始終可使用 **trace_rc_logic** 工具查詢 concept 中的理論背景。"
    )

    return (
        _BASE_SYSTEM_PROMPT
        + "\n"
        + need_section
        + skill_section
        + "\n"
        + tool_section
        + trace_hint
    )


def _build_fallback_tool_lines(socrates_state: ChatState) -> List[str]:
    """Fallback：從 socrates_state 組裝工具描述行（維持舊邏輯）。"""
    tools_info = socrates_state.get("tools") or []
    params     = socrates_state.get("parameters") or []
    param_map  = {p["tool_name"]: p.get("parameters", {}) for p in params}
    lines: List[str] = []

    for idx, t in enumerate(tools_info, 1):
        tname      = t["tool_name"]
        desc       = t.get("description", "")
        concept    = t.get("matched_concept", "")
        skills     = ", ".join(t.get("skills", []))
        from_agent = t.get("from_agent", "")
        dep_tag    = " [依賴工具]" if t.get("is_dependency", False) else ""

        pschema    = param_map.get(tname)
        param_hint = ""
        if pschema:
            props    = pschema.get("properties", {})
            required = pschema.get("required", [])
            param_parts = [
                f"     - {pname}: {pdef.get('description', '')} "
                f"{'（必填）' if pname in required else '（選填）'}"
                for pname, pdef in props.items()
            ]
            if param_parts:
                param_hint = "\n" + "\n".join(param_parts)

        lines.append(
            f"{idx}. **{tname}**{dep_tag} — {desc}\n"
            f"   來源: {from_agent} ｜ 概念: {concept} ｜ 技能: {skills}"
            f"{param_hint}"
        )

    return lines


# ═══════════════════════════════════════════════════════════════
# Dynamic Tool Builder（skill-aware）
# ═══════════════════════════════════════════════════════════════

def build_tools(
    socrates_state: ChatState,
    skill: Optional["RCSkill"] = None,
) -> List[StructuredTool]:
    """組裝最終工具清單。

    優先順序：
    1. skill.tools（若 skill 存在且有工具）
    2. fallback：從 _TOOL_IMPLEMENTATIONS 組裝（舊機制）

    無論哪種情況，最後都附上常駐的 trace_rc_logic 工具。
    """
    if skill and skill.has_tools:
        langchain_tools = list(skill.tools)
    else:
        langchain_tools = _build_fallback_tools(socrates_state)

    # 常駐工具
    langchain_tools.append(make_trace_rc_logic_tool())

    return langchain_tools


def _build_fallback_tools(socrates_state: ChatState) -> List[StructuredTool]:
    """Fallback：從 socrates_state 的 tools 資訊組裝（舊機制）。"""
    resolved_tools = socrates_state.get("tools") or []
    seen: set = set()
    langchain_tools: List[StructuredTool] = []

    for tool_info in resolved_tools:
        tname = tool_info["tool_name"]
        if tname in seen:
            continue
        seen.add(tname)

        impl = _TOOL_IMPLEMENTATIONS.get(tname)
        if impl is None:
            continue

        registry_entry = TOOLS_REGISTRY.get(tname, {})
        description    = tool_info.get("description") or registry_entry.get("description", "")

        if isinstance(impl, BaseTool):
            langchain_tools.append(impl)
        else:
            langchain_tools.append(StructuredTool.from_function(
                func=impl,
                name=tname,
                description=description,
            ))

    return langchain_tools


# ═══════════════════════════════════════════════════════════════
# MainAgent
# ═══════════════════════════════════════════════════════════════

class MainAgent:
    """無狀態設計的 MainAgent（Gemini 2.5 Flash）。

    每次 chat() 呼叫都會動態組裝 system prompt 與 LangGraph 執行圖，
    不持有任何對話狀態，所有歷史由呼叫端（entry.py）管理。

    Skill 注入：
        chat(..., skill=RCSkill) 時，skill 的 prompt_context 與 tools
        將取代舊的 socrates_state-based 組裝邏輯。
    """

    def __init__(
        self,
        *,
        model:   str = "gemini-2.5-flash",
        api_key: str = "",
    ):
        self._model_name = model
        self._base_llm   = create_chat_google_generative_ai(
            model=model,
            temperature=0,
            api_key=api_key,
        )

    def chat(
        self,
        user_input:     str,
        socrates_state: ChatState,
        history:        List[BaseMessage],
        skill:          Optional["RCSkill"] = None,
    ) -> dict:
        """單次處理對話請求（完全無狀態）。

        Args:
            user_input:     使用者當次輸入的文字。
            socrates_state: 包含 query / tools / parameters 等意圖收斂結果。
            history:        目前為止的對話歷史（由外部傳入）。
            skill:          選取 path 後對應的 RC Skill（可為 None）。
                            有 skill 時，使用 skill.prompt_context 與 skill.tools；
                            無 skill 時，fallback 到 socrates_state-based 舊機制。

        Returns:
            dict: {
                "content":         str,               # 助理回應文字
                "tool_calls_made": List[str],         # 本次呼叫的工具名稱清單
                "messages":        List[BaseMessage], # 更新後的完整歷史
                "skill_used":      Optional[str],     # 使用的 skill rc_name（若有）
            }
        """
        if socrates_state.get("phase") != ConversationPhase.PATH_CONFIRMED.value:
            raise ValueError(
                f"MainAgent 需要 phase='path_confirmed'。目前為 {socrates_state.get('phase')}"
            )

        system_prompt = build_system_prompt(socrates_state, skill=skill)
        tools         = build_tools(socrates_state, skill=skill)
        has_tools     = bool(tools)

        llm_bound = self._base_llm.bind_tools(tools) if has_tools else self._base_llm

        def call_model(state: AgentState) -> AgentState:
            normalized = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            response = call_with_retry(
                lambda: llm_bound.invoke([SystemMessage(content=system_prompt)] + normalized)
            )
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", call_model)

        if has_tools:
            graph_builder.add_node("tools", ToolNode(tools))
            graph_builder.set_entry_point("agent")
            graph_builder.add_conditional_edges(
                "agent", should_continue, {"tools": "tools", END: END}
            )
            graph_builder.add_edge("tools", "agent")
        else:
            graph_builder.set_entry_point("agent")
            graph_builder.add_edge("agent", END)

        graph = graph_builder.compile()

        current_messages = list(history) + [HumanMessage(content=user_input)]
        result           = graph.invoke({"messages": current_messages})
        updated_messages = list(result["messages"])

        from core.socrates_bot import SocratesChatBot
        content = SocratesChatBot._extract_raw_content(updated_messages)

        tool_calls_made: List[str] = []
        for msg in updated_messages[len(history):]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)

        return {
            "content":         content,
            "tool_calls_made": tool_calls_made,
            "messages":        updated_messages,
            "skill_used":      skill.rc_name if skill else None,
        }

    def describe(
        self,
        socrates_state: ChatState,
        skill: Optional["RCSkill"] = None,
    ) -> str:
        """回傳 Agent 的摘要資訊（供 debug / logging）。"""
        query      = socrates_state.get("query")
        tools      = build_tools(socrates_state, skill=skill)
        tool_names = [t.name for t in tools]
        skill_info = f"  RC Skill: {skill.rc_name}" if skill else "  RC Skill: （無）"
        return "\n".join([
            "MainAgent 摘要 (Stateless)",
            f"  當前需求: {query}",
            skill_info,
            f"  工具清單: {', '.join(tool_names)}",
        ])
