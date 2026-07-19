# -*- coding: utf-8 -*-
"""core/socrates_bot.py — 需求釐清對話機器人。

SocratesChatBot 負責：
  1. 透過多輪對話釐清使用者的分析需求（Socratic 問答）
  2. 呼叫 retrieve_concepts tool 查詢知識圖譜候選
  3. 呼叫 retrieve_concept_detail tool 說明各候選的理論背景
  4. 引導使用者在前端選定推理路徑

依賴注入：
  - CandidateRetriever  : 語意搜尋候選 concept
  - ConceptDetailRetriever : 查詢 concept 詳情
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core.chat_state import ChatState, ConversationPhase
from core.google_llm import create_chat_google_generative_ai
from core.llm_retry  import call_with_retry
from core.registry   import resolve_tools_from_candidates
from retrieval.candidate_retriever import CandidateRetriever
from retrieval.concept_retriever   import ConceptDetailRetriever


# ═══════════════════════════════════════════════════════════════
# LangGraph 內部狀態（僅 SocratesChatBot 使用）
# ═══════════════════════════════════════════════════════════════

class _InternalState(TypedDict):
    messages:   Annotated[List[BaseMessage], add_messages]
    


# ═══════════════════════════════════════════════════════════════
# System prompt & OutputSchema
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一位專業經濟學顧問，協助使用者釐清分析需求。

## 工作內容

- 透過對話了解使用者的分析目標，1-2 輪即可，避免過度追問。
- 若使用者提及某個具體概念（訊息中可能帶有「(eid: ...)」標記，代表使用者
  在圖譜介面上點選了該節點），呼叫 retrieve_concept_detail 並帶入該 eid，
  向使用者說明其定義、機制、關鍵假設等理論背景。
- 若下方提供「目前語意上最相關的候選概念」區塊，那只是系統內部參考，
  不需要，也不應該在回覆中列出這份清單或提及其存在。
- 不需要主動引導使用者選路徑——路徑選擇由前端圖譜介面完成，
  你只需要專注於需求釐清與概念說明。

## 輸出格式（強制）
每輪必須輸出合法 JSON，不得有任何前綴文字或 markdown code block：
{"ai_message": "<繁體中文回覆>", "intent_types": [...], "context_summary": "<英文摘要>"}

- intent_types：候選值同前（causal_mechanism, equilibrium_analysis, forecasting,
  event_study, structural_estimation, policy_evaluation, asset_pricing, descriptive）。
- context_summary：每輪重寫的英文摘要，總結使用者目前完整的分析需求，
  供後端 Semantic 機制做 embedding 相似度比對（前端排名、圖譜點亮皆依賴此欄位）。
"""


class OutputSchema(BaseModel):
    ai_message: str = Field(description="繁體中文回覆內容。")
    intent_types: List[str] = Field(
        default=[],
        description=(
            "從使用者 query 解析出的分析意圖類型，可複數。"
            "候選值：causal_mechanism, equilibrium_analysis, forecasting, "
            "event_study, structural_estimation, policy_evaluation, "
            "asset_pricing, descriptive。"
            "需求尚未釐清時回傳空列表。"
        ),
    )
    context_summary: str = Field(
        default="",
        description=(
            "每輪重寫的英文摘要，涵蓋使用者目前完整的分析需求。"
            "不是逐輪疊加，而是根據目前為止的完整對話重新總結，"
            "供 Semantic 機制做 embedding 相似度比對。"
        ),
    )


# ═══════════════════════════════════════════════════════════════
# SocratesChatBot
# ═══════════════════════════════════════════════════════════════

class SocratesChatBot:
    """需求釐清對話機器人。

    Parameters
    ----------
    candidate_retriever  : CandidateRetriever
    concept_retriever    : ConceptDetailRetriever
    model                : LLM 模型名稱
    api_key              : Google API key（選填）
    top_k_entities       : 語意搜尋回傳數量
    """

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        concept_retriever:   ConceptDetailRetriever,
        *,
        model:          str = "gemini-2.5-flash",
        api_key:        str = "",
        top_k_entities: int = 3,
    ) -> None:
        self._candidate_retriever = candidate_retriever
        self._concept_retriever   = concept_retriever
        self._top_k               = top_k_entities

        self._base_llm = create_chat_google_generative_ai(
            model=model,
            temperature=0.1,
            api_key=api_key,
        )

        self._concept_detail_tool = self._make_concept_detail_tool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def init_state() -> ChatState:
        return ChatState(
            messages       = [],
            agent_messages = [],
            phase          = ConversationPhase.RUNNING.value,
            intent_types   = None,
            context_summary = None,   # ← 新增            
        )

    def process_message(self, state: ChatState, user_input: str) -> ChatState:
    if state.get("phase") == ConversationPhase.PATH_CONFIRMED.value:
        return state

    state["messages"].append(HumanMessage(content=user_input))

    grounding = self._build_candidate_grounding(state)
    effective_prompt = self._build_effective_prompt(state, grounding)
    prior_msgs = self._normalize_prior_messages(state.get("agent_messages") or [])

    llm_with_tools = self._base_llm.bind_tools([self._concept_detail_tool])

    graph_result = self._build_graph(llm_with_tools).invoke({
        "messages": [SystemMessage(content=effective_prompt)]
                    + prior_msgs
                    + [HumanMessage(content=user_input)],
    })

    state["agent_messages"] = graph_result["messages"]

    bot_resp    = self._parse_bot_response(graph_result["messages"])
    ai_text     = bot_resp.get("ai_message", "")
    new_intents = bot_resp.get("intent_types", [])
    new_summary = bot_resp.get("context_summary", "")
    if new_intents:
        state["intent_types"] = new_intents
    if new_summary:
        state["context_summary"] = new_summary

    state["messages"].append(AIMessage(content=ai_text))
    return state

    def _build_candidate_grounding(self, state: ChatState) -> str:
        """用上一輪的 context_summary 查一次 top-k candidates，
        僅供 LLM 內部參考不對使用者展示，也不佔用一次 tool call。"""
        query_text = (state.get("context_summary") or "").strip()
        if not query_text:
            return ""
        candidates = self._candidate_retriever.retrieve_candidates(query_text, top_k=5)
        if not candidates:
            return ""
        lines = [f"- {c['rc_name']} (eid={c['eid']})" for c in candidates]
        return (
            "\n\n══════════════════════════════\n"
            "內部參考：目前語意上最相關的候選概念（不對使用者展示此清單）\n"
            "══════════════════════════════\n"
            + "\n".join(lines)
        )

    def confirm_path(self, state: ChatState) -> ChatState:
        """標記路徑已確認，凍結 socratic 階段。"""
        state["phase"] = ConversationPhase.PATH_CONFIRMED.value
        return state

    def get_last_reply(self, state: ChatState) -> str:
        """取得最新一則 AI 回覆文字。"""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_effective_prompt(
        state:         ChatState,
        grounding: Optional[str],
    ) -> str:
        prior          = state.get("agent_messages") or []
        first_system   = next(
            (m for m in prior if isinstance(m, SystemMessage)), None
        )
        base_prompt    = first_system.content if first_system else _SYSTEM_PROMPT

        if not grounding:
            return base_prompt

        return (
            base_prompt
            + "══════════════════════════════\n"
            + grounding
            + "若使用者追問特定概念的理論細節，呼叫 retrieve_concept_detail。\n"
        )

    @staticmethod
    def _normalize_prior_messages(
        prior: List[BaseMessage],
    ) -> List[BaseMessage]:
        """過濾掉 SystemMessage（避免重複），保留其餘歷史。"""
        return [m for m in prior if not isinstance(m, SystemMessage)]

    # ------------------------------------------------------------------
    # LangGraph（內部 agentic loop）
    # ------------------------------------------------------------------

    def _build_graph(self, llm_with_tools):
        tool_map = {
            self._concept_detail_tool.name: self._concept_detail_tool,
        }

        def agent_node(state: _InternalState) -> dict:
            response = call_with_retry(
                lambda: llm_with_tools.invoke(state["messages"])
            )
            return {"messages": [response]}

        def tool_executor_node(state: _InternalState) -> dict:
            last          = state["messages"][-1]
            tool_messages: List[ToolMessage] = []
            state_updates: dict = {}

            for tc in getattr(last, "tool_calls", []):
                tool = tool_map.get(tc["name"])
                if tool is None:
                    tool_messages.append(ToolMessage(
                        content=f"Unknown tool: {tc['name']}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))
                    continue

                raw_result = tool.invoke(tc["args"])
                slim, updates = self._process_tool_result(
                    tc["name"], raw_result
                )
                state_updates.update(updates)
                tool_messages.append(ToolMessage(
                    content=slim,
                    tool_call_id=tc["id"],
                    name=tc["name"],
                ))

            return {"messages": tool_messages, **state_updates}

        def route(state: _InternalState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tool_executor"
            return END

        g = StateGraph(_InternalState)
        g.add_node("agent",         agent_node)
        g.add_node("tool_executor", tool_executor_node)
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent", route,
            {"tool_executor": "tool_executor", END: END},
        )
        g.add_edge("tool_executor", "agent")
        return g.compile()

    def _process_tool_result(
        self,
        tool_name:  str,
        raw_result: str,
    ) -> tuple[str, dict]:
        """解析 tool 回傳值，拆出 slim 訊息與 state 更新。"""
        try:
            data = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            data = {}


        if tool_name == self._concept_detail_tool.name:
            if data.get("found"):
                return self._format_concept_detail(data), {}
            return data.get("message") or raw_result, {}

        return raw_result, {}

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------



    def _make_concept_detail_tool(self) -> StructuredTool:
    concept_retriever = self._concept_retriever

    def retrieve_concept_detail(eid: str = "", rc_name: str = "") -> str:
        """查詢指定 Concept 的詳細資訊(定義/機制/關鍵假設/功能/論文背景)。
        若使用者訊息帶有「(eid: ...)」標記(代表在圖譜上明確點選過該節點)，
        務必傳入該 eid 以確保精確查詢；只有在使用者純粹用文字描述、
        沒有明確 eid 可用時，才退回用 rc_name 模糊比對。"""
        if eid:
            result = concept_retriever.retrieve_by_eid(eid)
        else:
            result = concept_retriever.retrieve(rc_name)
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=retrieve_concept_detail,
        name="retrieve_concept_detail",
        description=(
            "查詢指定 Concept 的詳細資訊：definition / mechanism / "
            "key_assumptions / function（僅 RC 有值）/ 所屬論文的理論前提與模型約束。"
            "優先傳入 eid（使用者訊息中帶有「(eid: ...)」標記時一定要帶），"
            "查無 eid 才用 rc_name 模糊比對。"
        ),
    )

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw_content(msgs: List[BaseMessage]) -> str:
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
        raise ValueError("No AIMessage with content found in agent_messages.")

    @staticmethod
    def _parse_bot_response(msgs: List[BaseMessage]) -> dict:
        raw = SocratesChatBot._extract_raw_content(msgs)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
            if "ai_message" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {"ai_message": raw, "intent_types": []}

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_concept_detail(data: dict) -> str:
        lines: List[str] = [f"概念：{data.get('rc_name', '')}"]
        if data.get("definition"):
            lines.append(f"定義：{data['definition']}")
        if data.get("mechanism"):
            lines.append(f"機制：{data['mechanism']}")
        if data.get("key_assumptions"):
            lines.append(f"關鍵假設：{data['key_assumptions']}")
        if data.get("function"):
            lines.append(f"功能（Resulting Concept）：{data['function']}")
        for p in (data.get("papers") or []):
            lines.append(f"  ▸ {p.get('title', '（未命名）')}")
            if p.get("premises"):
                lines.append(f"    前提假設：{p['premises']}")
            if p.get("constraints"):
                lines.append(f"    模型約束：{p['constraints']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CLI（開發用）
    # ------------------------------------------------------------------

    def run_interactive(self) -> ChatState:
        import json as _json
        state = self.init_state()
        print("=== SocratesChatBot ===\n")
        state = self.process_message(state, "你好，請開始引導我。")
        print(f"Bot: {self.get_last_reply(state)}\n")
        while state.get("phase") not in (
            ConversationPhase.CONCEPT_RESOLVED.value,
            ConversationPhase.PATH_CONFIRMED.value,
        ):
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[中斷]")
                break
            if not user_input:
                continue
            state = self.process_message(state, user_input)
            print(f"Bot: {self.get_last_reply(state)}\n")

        print("=== 最終結果 ===")
        print(_json.dumps({
            "query":        state.get("query"),
            "phase":        state.get("phase"),
            "candidates":   state.get("candidates"),
            "tools":        state.get("tools"),
            "intent_types": state.get("intent_types"),
        }, ensure_ascii=False, indent=2))
        return state