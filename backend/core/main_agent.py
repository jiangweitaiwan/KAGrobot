# -*- coding: utf-8 -*-
"""core/main_agent.py — 主分析 Agent（無狀態設計）。

MainAgent 每次 chat() 呼叫：
  1. 動態組裝 system prompt（BASE + 需求 + skill context + 工具清單）
  2. 動態組裝工具清單（skill.tools 優先，fallback 到 socrates_state）
  3. 建立單次 LangGraph 執行圖並完成推理

Skill 注入：
  chat(..., skill=RCSkill) 時，skill.prompt_context 與 skill.tools
  取代基於 socrates_state 的 fallback 組裝邏輯。

所有歷史由呼叫端（OrchestratorNodes）管理，MainAgent 本身不持有狀態。
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, TypedDict, TYPE_CHECKING

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated

from core.chat_state import ChatState, ConversationPhase
from core.google_llm import create_chat_google_generative_ai
from core.llm_retry  import call_with_retry
from core.registry   import AGENTS_REGISTRY, TOOLS_REGISTRY, collect_tools_with_deps
from tools.base      import _TOOL_IMPLEMENTATIONS
from tools.knowledge import make_knowledge_qa_tool, make_trace_rc_logic_tool

import tools  # noqa: F401  確保所有工具副作用式註冊已執行

if TYPE_CHECKING:
    from core.skill_loader import RCSkill


# ═══════════════════════════════════════════════════════════════
# LangGraph Agent State（內部用）
# ═══════════════════════════════════════════════════════════════

class _AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ═══════════════════════════════════════════════════════════════
# 常數
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
# MainAgent
# ═══════════════════════════════════════════════════════════════

class MainAgent:
    """無狀態設計的主分析 Agent。

    Parameters
    ----------
    model   : LLM 模型名稱
    api_key : Google API key（選填）
    """

    def __init__(
        self,
        *,
        model:   str = "gemini-2.5-flash",
        api_key: str = "",
    ) -> None:
        self._model_name = model
        self._base_llm   = create_chat_google_generative_ai(
            model=model,
            temperature=0,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input:     str,
        socrates_state: ChatState,
        history:        List[BaseMessage],
        skill:          Optional["RCSkill"] = None,
    ) -> dict:
        """單次處理對話請求（完全無狀態）。

        Parameters
        ----------
        user_input     : 使用者當次輸入
        socrates_state : 包含 query / tools / parameters 等意圖收斂結果
        history        : 目前為止的對話歷史（由外部傳入）
        skill          : 選取 path 後對應的 RC Skill（可為 None）

        Returns
        -------
        {
            "content":         str,
            "tool_calls_made": List[str],
            "messages":        List[BaseMessage],
            "skill_used":      Optional[str],
        }
        """
        if socrates_state.get("phase") != ConversationPhase.PATH_CONFIRMED.value:
            raise ValueError(
                f"MainAgent 需要 phase='path_confirmed'。"
                f"目前為 {socrates_state.get('phase')}"
            )

        system_prompt = self._build_system_prompt(socrates_state, skill)
        tools         = self._build_tools(socrates_state, skill)
        has_tools     = bool(tools)

        llm_bound = self._base_llm.bind_tools(tools) if has_tools else self._base_llm

        graph    = self._build_graph(llm_bound, tools, system_prompt, has_tools)
        messages = list(history) + [HumanMessage(content=user_input)]
        result   = graph.invoke({"messages": messages})

        updated_messages: List[BaseMessage] = list(result["messages"])
        content          = SocratesChatBot_extract_raw_content(updated_messages)

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
        """回傳 Agent 摘要資訊（供 debug / logging）。"""
        query      = socrates_state.get("query")
        tools      = self._build_tools(socrates_state, skill)
        tool_names = [t.name for t in tools]
        skill_info = f"  RC Skill: {skill.rc_name}" if skill else "  RC Skill: （無）"
        return "\n".join([
            "MainAgent 摘要 (Stateless)",
            f"  當前需求: {query}",
            skill_info,
            f"  工具清單: {', '.join(tool_names)}",
        ])

    # ------------------------------------------------------------------
    # LangGraph 建構（每次 chat() 動態建立）
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        llm_bound,
        tools:         List,
        system_prompt: str,
        has_tools:     bool,
    ):
        def call_model(state: _AgentState) -> _AgentState:
            normalized = [
                m for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            response = call_with_retry(
                lambda: llm_bound.invoke(
                    [SystemMessage(content=system_prompt)] + normalized
                )
            )
            return {"messages": [response]}

        def should_continue(state: _AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(_AgentState)
        builder.add_node("agent", call_model)

        if has_tools:
            builder.add_node("tools", ToolNode(tools))
            builder.set_entry_point("agent")
            builder.add_conditional_edges(
                "agent", should_continue, {"tools": "tools", END: END}
            )
            builder.add_edge("tools", "agent")
        else:
            builder.set_entry_point("agent")
            builder.add_edge("agent", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # System prompt 組裝（私有）
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        socrates_state: ChatState,
        skill:          Optional["RCSkill"],
    ) -> str:
        query = socrates_state.get("query") or "（未提供）"

        need_section = textwrap.dedent(f"""\
            ══════════════════════════════
            使用者需求
            ══════════════════════════════
            {query}
        """)

        skill_section = ""
        if skill and skill.prompt_context.strip():
            skill_section = textwrap.dedent(f"""\

            ══════════════════════════════
            分析方法：{skill.rc_name}
            ══════════════════════════════
            {skill.prompt_context.strip()}
            """)

        if skill and skill.has_tools:
            tool_lines  = self._tool_lines_from_skill(skill)
            tool_header = "可用工具（來自 RC Skill）"
        else:
            tool_lines  = self._fallback_tool_lines(socrates_state)
            tool_header = "可用工具"

        tool_section = (
            f"══════════════════════════════\n"
            f"{tool_header}\n"
            f"══════════════════════════════\n"
            + ("\n".join(tool_lines) if tool_lines else "（無額外工具）")
        )

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

    @staticmethod
    def _tool_lines_from_skill(skill: "RCSkill") -> List[str]:
        return [
            f"{i}. **{t.name}** — {t.description}"
            for i, t in enumerate(skill.tools, 1)
        ]

    @staticmethod
    def _fallback_tool_lines(socrates_state: ChatState) -> List[str]:
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

    # ------------------------------------------------------------------
    # Tool 清單組裝（私有）
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        socrates_state: ChatState,
        skill:          Optional["RCSkill"],
    ) -> List[StructuredTool]:
        if skill and skill.has_tools:
            langchain_tools = list(skill.tools)
        else:
            langchain_tools = self._fallback_tools(socrates_state)

        langchain_tools.append(make_trace_rc_logic_tool())
        return langchain_tools

    @staticmethod
    def _fallback_tools(socrates_state: ChatState) -> List[StructuredTool]:
        resolved_tools = socrates_state.get("tools") or []
        seen: set      = set()
        result: List[StructuredTool] = []

        for tool_info in resolved_tools:
            tname = tool_info["tool_name"]
            if tname in seen:
                continue
            seen.add(tname)

            impl = _TOOL_IMPLEMENTATIONS.get(tname)
            if impl is None:
                continue

            registry_entry = TOOLS_REGISTRY.get(tname, {})
            description    = (
                tool_info.get("description") or registry_entry.get("description", "")
            )

            if isinstance(impl, BaseTool):
                result.append(impl)
            else:
                result.append(StructuredTool.from_function(
                    func=impl,
                    name=tname,
                    description=description,
                ))

        return result


# ═══════════════════════════════════════════════════════════════
# Module-level 相容函數
# （供 orchestrator/nodes.py 呼叫 _extract_raw_content 使用）
# ═══════════════════════════════════════════════════════════════

def SocratesChatBot_extract_raw_content(msgs: List[BaseMessage]) -> str:
    """從 agent_messages 中提取最後一則 AI 回覆的原始文字。

    命名刻意不以下底線開頭，表示這是 module-level 公開函數，
    而非 SocratesChatBot 的私有方法。
    實作邏輯移至此處，避免 main_agent 直接呼叫 socrates_bot 的私有方法。
    """
    for msg in reversed(msgs):
        if not (isinstance(msg, AIMessage) and msg.content):
            continue
        if isinstance(msg.content, list):
            if not msg.content:
                raise ValueError("AIMessage content list is empty.")
            item = msg.content[0]
            raw  = item.get("text", "") if isinstance(item, dict) else str(item)
        else:
            raw = str(msg.content)
        raw = raw.strip()
        if not raw:
            raise ValueError("AIMessage content is blank.")
        return raw
    raise ValueError("No AIMessage with content found in messages.")
