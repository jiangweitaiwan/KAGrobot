"""
core/main_agent.py
─────────────────────────────────────────────────────────────
MainAgent：無狀態設計，每次 chat() 呼叫動態組裝 LangGraph 執行圖。

對外介面：
    MainAgent   — 主類別
"""

from __future__ import annotations

import json
import textwrap
from typing import List, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
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
from settings import get_settings
from tools.base import _TOOL_IMPLEMENTATIONS
from tools.knowledge import make_knowledge_qa_tool, make_trace_rc_logic_tool

# 確保所有工具副作用式註冊已執行
import tools  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# LangGraph Agent State
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ═══════════════════════════════════════════════════════════════
# Reasoning chain 輔助
# ═══════════════════════════════════════════════════════════════

def _format_reasoning_chains(chains: List[dict]) -> str:
    """將 reasoning chain 列表序列化為易讀文字（供 system prompt 使用）。"""
    if not chains:
        return "（無可用推理鏈）"
    parts: List[str] = []
    for i, chain in enumerate(chains, 1):
        start = chain.get("start_concept", {}).get("name", "?")
        step_names = [start]
        for step in chain.get("steps", []):
            to_name = step.get("to_concept", {}).get("name", "?")
            step_names.append(to_name)
            paper = step.get("via_paper")
            if paper and paper.get("title"):
                step_names[-1] += f" (via: {paper['title']})"
        path_str = " → ".join(step_names)
        final_paper = chain.get("final_paper")
        paper_info  = ""
        if final_paper:
            paper_info = f"\n   關聯論文: {final_paper.get('title', 'N/A')}"
            if premises := final_paper.get("conceptual_premises"):
                paper_info += f"\n   前提假設: {premises}"
            if constraints := final_paper.get("contextual_model_constraints"):
                paper_info += f"\n   模型約束: {constraints}"
        parts.append(f"推理鏈 {i}: {path_str}{paper_info}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Dynamic System Prompt Builder
# ═══════════════════════════════════════════════════════════════

def build_system_prompt(socrates_state: ChatState) -> str:
    """從 SocratesChatBot 的 resolved state 動態組裝 system prompt。"""
    query      = socrates_state.get("query") or "（未提供）"
    tools_info = socrates_state.get("tools") or []
    params     = socrates_state.get("parameters") or []

    # ── 工具描述 ──
    tool_lines: List[str] = []
    param_map = {p["tool_name"]: p.get("parameters", {}) for p in params}
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

        tool_lines.append(
            f"{idx}. **{tname}**{dep_tag} — {desc}\n"
            f"   來源代理: {from_agent} ｜ 匹配概念: {concept} ｜ 技能: {skills}"
            f"{param_hint}"
        )

    tool_section = "\n".join(tool_lines) if tool_lines else "（無額外工具）"

    # ── 選定推理鏈（僅包含 Resulting Concept 節點名稱序列）──
    selected_chain = _extract_selected_chain(socrates_state)
    chain_section  = _format_selected_chain(selected_chain)

    # ── 對話歷史摘要 ──
    conv_snippets: List[str] = []
    for msg in socrates_state.get("messages", []):
        role = "使用者" if isinstance(msg, HumanMessage) else "助理"
        text = msg.content[:200] if hasattr(msg, "content") else str(msg)[:200]
        conv_snippets.append(f"  [{role}] {text}")
    conv_summary = "\n".join(conv_snippets[-8:]) if conv_snippets else "（無前置對話）"

    return textwrap.dedent(f"""\
        你是一位專業的金融與經濟分析助手。你的任務是運用可用工具，
        協助使用者解決以下經過釐清的需求。

        ══════════════════════════════
        使用者需求
        ══════════════════════════════
        {query}


        ══════════════════════════════
        推理鏈
        ══════════════════════════════
        {chain_section}

        ══════════════════════════════
        可用工具
        ══════════════════════════════
        {tool_section}

        此外，你始終可使用 **trace_rc_logic** 工具查詢 concept 中的理論背景。

        ══════════════════════════════
        使用原則
        ══════════════════════════════
        - 主動判斷該呼叫哪些工具，可以組合多個工具完成任務
        - 收到工具結果後，用繁體中文清楚解讀
        - 結合推理鏈的理論背景提供深度分析
        - 若參數不完整，先向使用者詢問
        - 回答先給結論，再解釋原因
        - 專業但易懂，避免過度學術術語
    """)


# ═══════════════════════════════════════════════════════════════
# Dynamic Tool Builder
# ═══════════════════════════════════════════════════════════════

def build_tools_from_state(socrates_state: ChatState) -> List[StructuredTool]:
    """從 SocratesChatBot 的 resolved state 動態組裝 LangChain 工具清單。"""
    resolved_tools = socrates_state.get("tools") or []
    chains         = socrates_state.get("reasoning_chains") or []

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

    # 常駐工具：trace_rc_logic（知識圖譜邏輯追蹤）
    langchain_tools.append(make_trace_rc_logic_tool())

    return langchain_tools


# ═══════════════════════════════════════════════════════════════
# MainAgent
# ═══════════════════════════════════════════════════════════════

class MainAgent:
    """無狀態設計的 MainAgent（Gemini 2.5 Flash）。

    每次 chat() 呼叫都會動態組裝 system prompt 與 LangGraph 執行圖，
    不持有任何對話狀態，所有歷史由呼叫端（entry.py）管理。
    """

    def __init__(
        self,
        *,
        model:   str = "gemini-2.5-flash",
        api_key: str = "",
    ):
        if not api_key:
            api_key = get_settings().google_api_key
        self._model_name = model
        self._api_key    = api_key
        self._base_llm   = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
        )

    def chat(
        self,
        user_input:     str,
        socrates_state: ChatState,
        history:        List[BaseMessage],
    ) -> dict:
        """單次處理對話請求（完全無狀態）。

        Args:
            user_input:     使用者當次輸入的文字。
            socrates_state: 包含 query / tools / parameters 等意圖收斂結果。
            history:        目前為止的對話歷史（由外部傳入）。

        Returns:
            dict: {
                "content":         str,               # 助理回應文字
                "tool_calls_made": List[str],         # 本次呼叫的工具名稱清單
                "messages":        List[BaseMessage], # 更新後的完整歷史
            }
        """
        if socrates_state.get("phase") != ConversationPhase.PATH_CONFIRMED.value:
            raise ValueError(
                f"MainAgent 需要 phase='path_confirmed'。目前為 {socrates_state.get('phase')}"
            )

        system_prompt = build_system_prompt(socrates_state)
        tools         = build_tools_from_state(socrates_state)
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

        # 從 core/socrates_bot 借用靜態方法解析回傳內容
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
        }

    def describe(self, socrates_state: ChatState) -> str:
        """回傳 Agent 的摘要資訊（供 debug / logging）。"""
        query      = socrates_state.get("query")
        tools      = build_tools_from_state(socrates_state)
        tool_names = [t.name for t in tools]
        return "\n".join([
            "MainAgent 摘要 (Stateless)",
            f"  當前需求: {query}",
            f"  動態工具: {', '.join(tool_names)}",
        ])
