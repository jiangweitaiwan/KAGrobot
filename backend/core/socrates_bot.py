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
    candidates: Optional[List[dict]]
    query:      Optional[str]
    


# ═══════════════════════════════════════════════════════════════
# System prompt & OutputSchema
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一位專業經濟學顧問，協助使用者從知識圖譜的推理路徑中，
選出最符合其分析需求的模型或方法。

## 工作流程

### 階段 1：需求釐清
- 了解使用者的分析目標（1-2 輪即可，避免過度追問）。

### 階段 2：查詢知識圖譜（retrieve_concepts）
- 釐清需求後，主動呼叫 `retrieve_concepts`，傳入精確的**英文**查詢語句。
- 若結果為空（found=false），告知使用者並請重新描述需求後再試。
- retrieve 完成後，向使用者說明找到的候選概念，並告知圖譜正在載入至右側面板。

### 階段 3：路徑說明與引導（圖譜已展開後）
- 若 system prompt 包含「目前已展開的知識圖譜路徑」區塊，代表圖譜已展開。
- 根據各候選的下游終點與基礎概念向使用者說明各候選的分析方向。
- 若使用者想了解某個概念的理論細節，主動呼叫 `retrieve_concept_detail`。
- 若使用者希望換一個分析方向，重新呼叫 `retrieve_concepts`。
- 引導使用者在右側圖譜點選路徑後按確認按鈕，路徑確認由前端完成。

## 輸出格式（強制）
每輪必須輸出合法 JSON，不得有任何前綴文字或 markdown code block：
{"ai_message": "<繁體中文回覆>", "intent_types": [...], "context_summary": "<英文摘要>"}

- 對話以繁體中文；retrieve_concepts 的查詢語句用英文。
- intent_types：在使用者需求已足夠清晰時，填入對應的分析類型標籤（可複數）。
  需求尚不明確時填入空列表 []。
  候選值：causal_mechanism, equilibrium_analysis, forecasting, event_study,
          structural_estimation, policy_evaluation, asset_pricing, descriptive
- context_summary：用一到兩句**英文**重新總結「使用者目前完整的分析需求」，
  每輪都根據目前為止的完整對話重寫（不是逐輪疊加），供後端 Semantic 機制
  做 embedding 相似度比對。需求尚不明確時，摘要目前已知的線索即可。
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

        self._retrieve_tool       = self._make_retrieve_tool()
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
            query          = None,
            candidates     = None,
            selected_index = None,
            tools          = None,
            parameters     = None,
            intent_types   = None,
            context_summary = None,   # ← 新增            
        )

    def process_message(
        self,
        state:         ChatState,
        user_input:    str,
        graph_context: Optional[str] = None,
    ) -> ChatState:
        if state.get("phase") == ConversationPhase.PATH_CONFIRMED.value:
            return state

        state["messages"].append(HumanMessage(content=user_input))

        effective_prompt = self._build_effective_prompt(state, graph_context)
        prior_msgs       = self._normalize_prior_messages(
            state.get("agent_messages") or []
        )

        llm_with_tools = self._base_llm.bind_tools([
            self._retrieve_tool,
            self._concept_detail_tool,
        ])

        graph_result = self._build_graph(llm_with_tools).invoke({
            "messages":   [SystemMessage(content=effective_prompt)]
                          + prior_msgs
                          + [HumanMessage(content=user_input)],
            "candidates": state.get("candidates") or [],
            "query":      state.get("query"),
        })

        state["agent_messages"] = graph_result["messages"]

        if graph_result.get("query"):
            state["query"] = graph_result["query"]

        new_candidates = graph_result.get("candidates")
        if new_candidates:
            state["candidates"] = new_candidates
            tools, params = resolve_tools_from_candidates(new_candidates)
            state["tools"]      = tools
            state["parameters"] = params
            state["phase"]      = ConversationPhase.CONCEPT_RESOLVED.value

        bot_resp    = self._parse_bot_response(graph_result["messages"])
        ai_text     = bot_resp.get("ai_message", "")
        new_intents = bot_resp.get("intent_types", [])
        new_summary = bot_resp.get("context_summary", "")
        if new_intents:
            state["intent_types"] = new_intents
        if new_summary:
            state["context_summary"] = new_summary   # ← 新增：每輪覆蓋重寫            

        state["messages"].append(AIMessage(content=ai_text))
        return state

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
        graph_context: Optional[str],
    ) -> str:
        prior          = state.get("agent_messages") or []
        first_system   = next(
            (m for m in prior if isinstance(m, SystemMessage)), None
        )
        base_prompt    = first_system.content if first_system else _SYSTEM_PROMPT

        if not graph_context:
            return base_prompt

        return (
            base_prompt
            + "\n\n══════════════════════════════\n"
            + "目前已展開的知識圖譜路徑\n"
            + "══════════════════════════════\n"
            + graph_context
            + "\n\n請根據上方各候選的下游終點與基礎概念向使用者說明分析方向。"
            + "若使用者追問特定概念的理論細節，呼叫 retrieve_concept_detail。\n"
            + "引導使用者在右側圖譜點選路徑後按確認按鈕。"
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
            self._retrieve_tool.name:       self._retrieve_tool,
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

        if tool_name == self._retrieve_tool.name:
            if data.get("found"):
                return (
                    data.get("display") or "（查無結果）",
                    {
                        "candidates": data.get("candidates", []),
                        "query":      data.get("query", ""),
                    },
                )
            return data.get("message") or raw_result, {}

        if tool_name == self._concept_detail_tool.name:
            if data.get("found"):
                return self._format_concept_detail(data), {}
            return data.get("message") or raw_result, {}

        return raw_result, {}

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------

    def _make_retrieve_tool(self) -> StructuredTool:
        candidate_retriever = self._candidate_retriever
        top_k               = self._top_k

        def retrieve_concepts(query: str) -> str:
            """查詢知識圖譜，找出與使用者需求最相關的 concept 候選清單。"""
            candidates = candidate_retriever.retrieve_candidates(
                query, top_k=top_k
            )
            if not candidates:
                return json.dumps({
                    "found":      False,
                    "message":    "No relevant concepts found in the knowledge graph.",
                    "candidates": [],
                    "display":    "",
                    "query":      query,
                }, ensure_ascii=False)

            lines = [
                f"[{c['index']}] {c['rc_name'] or '（未命名）'}"
                f"  （相似度 {c['score']}）"
                for c in candidates
            ]
            return json.dumps({
                "found":      True,
                "query":      query,
                "candidates": candidates,
                "display":    "\n".join(lines),
            }, ensure_ascii=False)

        return StructuredTool.from_function(
            func=retrieve_concepts,
            name="retrieve_concepts",
            description=(
                "查詢知識圖譜，找出與使用者需求最相關的 concept 候選清單。"
                "需求釐清完成後呼叫此 tool，傳入精確的英文查詢語句。"
                "若使用者希望換分析方向，亦可再次呼叫。"
            ),
        )

    def _make_concept_detail_tool(self) -> StructuredTool:
        concept_retriever = self._concept_retriever

        def retrieve_concept_detail(rc_name: str) -> str:
            """查詢指定 concept 的定義、機制、關鍵假設及所屬論文的理論前提與模型約束。"""
            result = concept_retriever.retrieve(rc_name)
            return json.dumps(result, ensure_ascii=False)

        return StructuredTool.from_function(
            func=retrieve_concept_detail,
            name="retrieve_concept_detail",
            description=(
                "查詢指定 Concept 的詳細資訊，包含：\n"
                "  - concept 的 definition（概念定義）\n"
                "  - concept 的 mechanism（作用機制）\n"
                "  - concept 的 key_assumptions（關鍵假設）\n"
                "  - concept 的 function（功能說明，僅 Resulting Concept 才有值，"
                "一般 concept 此欄位為空，可用以判斷是否為 RC）\n"
                "  - 產生此 concept 的 paper 標題\n"
                "  - paper 的 conceptual_premises（理論前提假設）\n"
                "  - paper 的 contextual_model_constraints（模型情境約束）\n"
                "當使用者詢問某個概念或路徑的理論背景、適用條件時呼叫。\n"
                "輸入 rc_name 為 Concept 的名稱（允許部分匹配）。"
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