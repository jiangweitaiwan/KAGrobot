"""
core/socrates_bot.py
─────────────────────────────────────────────────────────────
SocratesChatBot：需求釐清 + concept 候選選擇。

改動摘要：
  • ConversationPhase 新增 CONCEPT_RESOLVED / PATH_CONFIRMED
    （原 RESOLVED 保留為向下相容別名）
  • process_message 新增 graph_context 參數，動態注入路徑摘要
    至 system prompt，不污染 agent_messages 歷史
  • _handle_resolved 改設 CONCEPT_RESOLVED（不再直接 RESOLVED）
  • 新增 confirm_path() 供外部將 phase 推進至 PATH_CONFIRMED
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import neo4jreasoningretriever as nrr
from core.registry import (
    AGENTS_REGISTRY, AgentType, CONCEPT_TO_AGENT, TOOLS_REGISTRY,
)
from core.llm_retry import call_with_retry
from settings import get_settings

Neo4jReasoningRetriever = nrr.Neo4jReasoningRetriever


# ═══════════════════════════════════════════════════════════════
# ConversationPhase & ChatState
# ═══════════════════════════════════════════════════════════════

class ConversationPhase(Enum):
    RUNNING          = "running"
    CONCEPT_RESOLVED = "concept_resolved"   # 已選 concept，圖已展開，繼續對話
    PATH_CONFIRMED   = "path_confirmed"     # 使用者確認 path，移交 main_agent
    # 向下相容別名
    RESOLVED         = "concept_resolved"


class ChatState(TypedDict, total=False):
    messages:       List[BaseMessage]
    agent_messages: List[BaseMessage]
    phase:          str
    query:          Optional[str]
    # candidates: [{index, eid, rc_name, type, score}]
    candidates:     Optional[List[dict]]
    selected_index: Optional[int]
    tools:          Optional[List[dict]]
    parameters:     Optional[List[dict]]


# ═══════════════════════════════════════════════════════════════
# Tool dependency resolver
# ═══════════════════════════════════════════════════════════════

def collect_tools_with_deps(tool_names: List[str]) -> List[str]:
    visited: set       = set()
    ordered: List[str] = []

    def _walk(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for dep in TOOLS_REGISTRY.get(name, {}).get("dependencies", []):
            _walk(dep)
        ordered.append(name)

    for tn in tool_names:
        _walk(tn)
    return ordered


# ═══════════════════════════════════════════════════════════════
# LangGraph internal state
# ═══════════════════════════════════════════════════════════════

class _InternalState(TypedDict):
    messages:   Annotated[List[BaseMessage], add_messages]
    candidates: Optional[List[dict]]
    query:      Optional[str]


# ═══════════════════════════════════════════════════════════════
# System prompt & OutputSchema
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一位專業經濟學顧問，目標是協助使用者從知識圖譜的
Resulting Concept 中，選出最符合其實際需求的模型或分析方法。

## 工作流程

### 階段 1：需求釐清
- 了解使用者需求（1-2 輪即可，避免過度追問）

### 階段 2：查詢知識圖譜
- 呼叫 `retrieve_concepts` tool，傳入精確的**英文**查詢語句。
- 若結果為空（found=false），告知使用者並請重新描述需求後再試。

### 階段 3：展示候選，等待使用者選擇
- 將 `retrieve_concepts` 回傳的 `display` 欄位內容展示給使用者。
- 請使用者輸入選項編號（如 "1"）或概念名稱。
- 此時 action = "ask"。

### 階段 4：確認選擇
- 使用者明確給出編號或可辨識的名稱後：
  * 在 ai_message 中說明已選擇的項目。
  * 將 action 設為 "resolved"。
- 若使用者的輸入模糊或無法對應到任何候選，繼續追問，action 仍為 "ask"。

### 階段 5：圖譜路徑說明（若已提供路徑資訊）
- 若 system prompt 中包含「目前已展開的知識圖譜路徑」區塊，代表圖譜已展開。
- 使用者可能詢問路徑的含義、論文背景、適用情境等，請根據路徑資訊回答。
- 若使用者希望換一個分析方向，重新呼叫 `retrieve_concepts` 取得新的候選。
- 此階段 action 維持 "ask"（路徑確認由前端按鈕完成，不在此處處理）。

## 輸出格式（強制）
每輪必須輸出合法 JSON，不得有任何前綴文字或 markdown code block：
{"action": "ask"|"resolved", "ai_message": "<繁體中文回覆>", "selected_index": null}

**selected_index 規則：**
- action="ask" 時：selected_index 必須為 null。
- action="resolved" 時：selected_index 必須填入使用者所選項目的編號（整數，與展示的 [1][2]... 一致）。

- 對話以繁體中文；retrieve_concepts 的查詢語句用英文。
- 未完成選擇前，action 絕對不能為 "resolved"。
- 圖譜已展開後，不再需要 resolved，路徑確認由使用者在前端操作。
"""


class OutputSchema(BaseModel):
    action: Literal["ask", "resolved"] = Field(
        description="本輪狀態：ask=需要追問/等待使用者選擇；resolved=已明確完成選擇。"
    )
    ai_message: str = Field(description="繁體中文回覆內容。")
    selected_index: Optional[int] = Field(
        default=None,
        description=(
            "action='resolved' 時必填。使用者所選候選項目的 1-based 編號。"
            "action='ask' 時設為 null。"
        ),
    )


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _resolve_agent_type(
    rc_name: str,
    concept_to_agent: Dict[str, AgentType],
) -> Optional[AgentType]:
    name = rc_name.lower().strip()
    if name in concept_to_agent:
        return concept_to_agent[name]
    for key, agent in concept_to_agent.items():
        if key.lower().strip() in name or name in key.lower().strip():
            return agent
    return None


def _resolve_tools_from_candidate(
    chosen_candidate: dict,
    concept_to_agent: Dict[str, AgentType],
) -> tuple:
    """從選定的 candidate 解析工具清單與參數。

    Parameters
    ----------
    chosen_candidate : {index, eid, rc_name, type, score}
    """
    rc_name = chosen_candidate.get("rc_name") or ""
    agent_t = _resolve_agent_type(rc_name, concept_to_agent)

    if agent_t is None:
        return [], []

    base_tools = AGENTS_REGISTRY.get(agent_t, {}).get("tools", [])
    all_tools  = collect_tools_with_deps(base_tools)

    resolved_tools:  List[dict] = []
    resolved_params: List[dict] = []

    for tname in all_tools:
        entry = TOOLS_REGISTRY.get(tname)
        if entry is None:
            continue
        is_dep = tname not in base_tools
        resolved_tools.append({
            "tool_name":       tname,
            "description":     entry["description"],
            "skills":          entry["skills"],
            "from_agent":      agent_t.value,
            "matched_concept": rc_name,
            "is_dependency":   is_dep,
        })
        if entry.get("parameters"):
            resolved_params.append({
                "tool_name":  tname,
                "parameters": entry["parameters"],
            })

    return resolved_tools, resolved_params


def _parse_selection(text: str, candidates: List[dict]) -> Optional[int]:
    """從使用者輸入解析選擇，回傳 0-based index。"""
    clean = text.strip()
    for token in clean.replace("、", " ").replace("，", " ").split():
        if token.isdigit():
            num = int(token)
            if 1 <= num <= len(candidates):
                return num - 1
    lower = clean.lower()
    for i, cand in enumerate(candidates):
        rc_name = (cand.get("rc_name") or "").lower()
        if rc_name and (rc_name in lower or lower in rc_name):
            return i
    return None


# ═══════════════════════════════════════════════════════════════
# Reasoning chain display helpers（供 main_agent 使用）
# ═══════════════════════════════════════════════════════════════

def _extract_selected_chain(socrates_state: ChatState) -> Optional[dict]:
    """從 socrates_state 取出選定 candidate 的摘要資訊。"""
    candidates: List[dict] = socrates_state.get("candidates") or []
    idx: Optional[int]     = socrates_state.get("selected_index")
    if idx is None or not candidates:
        return None
    if idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


def _format_selected_chain(candidate: Optional[dict]) -> str:
    """將選定 candidate 格式化為 system prompt 用的文字。"""
    if candidate is None:
        return "（未選定）"
    rc_name = candidate.get("rc_name") or "（未命名）"
    score   = candidate.get("score", "")
    score_s = f"（相似度 {score}）" if score else ""
    return f"選定分析概念：{rc_name}{score_s}"


# ═══════════════════════════════════════════════════════════════
# SocratesChatBot
# ═══════════════════════════════════════════════════════════════

class SocratesChatBot:
    """需求釐清 + concept 候選選擇的蘇格拉底對話 Agent。"""

    def __init__(
        self,
        retriever: Neo4jReasoningRetriever,
        *,
        model:            str  = "gemini-2.5-flash",
        api_key:          str  = "",
        concept_to_agent: Optional[Dict[str, AgentType]] = None,
        top_k_entities:   int  = 3,
    ):
        self._retriever        = retriever
        self._concept_to_agent = concept_to_agent or CONCEPT_TO_AGENT
        self._top_k            = top_k_entities

        if not api_key:
            api_key = get_settings().google_api_key

        base_llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=0.1)

        self._retrieve_tool  = self._make_retrieve_tool()
        self._llm_with_tools = base_llm.bind_tools([self._retrieve_tool])
        self._graph          = self._build_graph()

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
        )

    def process_message(
        self,
        state:         ChatState,
        user_input:    str,
        graph_context: Optional[str] = None,
    ) -> ChatState:
        """處理單輪對話。

        Parameters
        ----------
        state       : 目前的 ChatState。
        user_input  : 使用者輸入文字。
        graph_context : 已展開圖譜的路徑摘要文字（concept_resolved 階段傳入）。
                        動態附加至 system prompt，不寫進 agent_messages 歷史。
        """
        # PATH_CONFIRMED 後不再處理
        if state.get("phase") == ConversationPhase.PATH_CONFIRMED.value:
            return state

        state["messages"].append(HumanMessage(content=user_input))
        prior = state.get("agent_messages") or []

        normalized_prior: List[BaseMessage] = []
        first_system: Optional[SystemMessage] = None
        for m in prior:
            if isinstance(m, SystemMessage):
                if first_system is None:
                    first_system = m
                continue
            normalized_prior.append(m)

        # ── 動態組裝 system prompt ──────────────────────────────
        base_prompt = (
            first_system.content
            if first_system
            else _SYSTEM_PROMPT
        )
        if graph_context:
            effective_prompt = (
                base_prompt
                + "\n\n══════════════════════════════\n"
                + "目前已展開的知識圖譜路徑\n"
                + "══════════════════════════════\n"
                + graph_context
                + "\n\n使用者可能詢問路徑內容，請根據上方資訊協助解釋各路徑的意涵、"
                + "論文背景與適用情境。\n"
                + "若使用者希望換一個分析方向，可重新呼叫 retrieve_concepts 取得新候選。\n"
                + "路徑的最終確認由使用者在前端按鈕完成，此處不需輸出 resolved。"
            )
        else:
            effective_prompt = base_prompt

        system_msg   = SystemMessage(content=effective_prompt)
        graph_result = self._graph.invoke({
            "messages":   [system_msg] + normalized_prior + [HumanMessage(content=user_input)],
            "candidates": state.get("candidates") or [],
            "query":      state.get("query"),
        })
        new_agent_msgs: List[BaseMessage] = graph_result["messages"]
        state["agent_messages"] = new_agent_msgs

        if graph_result.get("query"):
            state["query"] = graph_result["query"]
        if graph_result.get("candidates"):
            state["candidates"] = graph_result["candidates"]

        bot_resp = self._parse_bot_response(new_agent_msgs)
        ai_text  = bot_resp.get("ai_message", "")
        action   = bot_resp.get("action", "ask")

        state["messages"].append(AIMessage(content=ai_text))

        # concept_resolved 階段不允許再次觸發 _handle_resolved
        if action == "resolved" and state.get("phase") != ConversationPhase.CONCEPT_RESOLVED.value:
            self._handle_resolved(state, user_input, bot_resp)

        return state

    def confirm_path(self, state: ChatState) -> ChatState:
        """外部確認 path 後呼叫，將 phase 推進到 PATH_CONFIRMED。

        由 socratic_node 在收到 selected_path_id 時呼叫。
        不改動 messages / agent_messages，僅更新 phase。
        """
        state["phase"] = ConversationPhase.PATH_CONFIRMED.value
        return state

    def get_last_reply(self, state: ChatState) -> str:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    # ------------------------------------------------------------------
    # LangGraph graph
    # ------------------------------------------------------------------

    def _build_graph(self):
        llm      = self._llm_with_tools
        tool_map = {self._retrieve_tool.name: self._retrieve_tool}

        def agent_node(state: _InternalState) -> dict:
            response = call_with_retry(lambda: llm.invoke(state["messages"]))
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
                        tool_call_id=tc["id"], name=tc["name"],
                    ))
                    continue

                raw_result = tool.invoke(tc["args"])
                try:
                    data = json.loads(raw_result)
                except (json.JSONDecodeError, TypeError):
                    data = {}

                if data.get("found"):
                    state_updates["candidates"] = data.get("candidates", [])
                    state_updates["query"]      = data.get("query", "")
                    slim = data.get("display") or "（查無結果）"
                else:
                    slim = data.get("message") or raw_result

                tool_messages.append(ToolMessage(
                    content=slim, tool_call_id=tc["id"], name=tc["name"],
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
            {"tool_executor": "tool_executor", END: END})
        g.add_edge("tool_executor", "agent")
        return g.compile()

    # ------------------------------------------------------------------
    # retrieve_concepts tool
    # ------------------------------------------------------------------

    def _make_retrieve_tool(self) -> StructuredTool:
        retriever = self._retriever
        top_k     = self._top_k

        def retrieve_concepts(query: str) -> str:
            """查詢知識圖譜，找出與使用者需求最相關的 concept 候選清單。"""
            candidates = retriever.retrieve_candidates(query, top_k=top_k)

            if not candidates:
                return json.dumps({
                    "found":      False,
                    "message":    "No relevant concepts found in the knowledge graph.",
                    "candidates": [],
                    "display":    "",
                    "query":      query,
                }, ensure_ascii=False)

            lines: List[str] = []
            for c in candidates:
                lines.append(
                    f"[{c['index']}] {c['rc_name'] or '（未命名）'}"
                    f"  （相似度 {c['score']}）"
                )

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
                "圖譜已展開後，若使用者希望換方向，亦可再次呼叫。"
            ),
        )

    # ------------------------------------------------------------------
    # JSON parsing
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
            if "action" in data and "ai_message" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {"action": "ask", "ai_message": raw, "selected_index": None}

    # ------------------------------------------------------------------
    # Handle resolved（concept 選定，phase → CONCEPT_RESOLVED）
    # ------------------------------------------------------------------

    def _handle_resolved(
        self,
        state:      ChatState,
        user_input: str,
        bot_resp:   dict,
    ) -> None:
        candidates = state.get("candidates") or []
        if not candidates:
            return

        chosen_idx: Optional[int] = None
        llm_index = bot_resp.get("selected_index")
        if isinstance(llm_index, (int, float)) and llm_index is not None:
            llm_index = int(llm_index)
            if 1 <= llm_index <= len(candidates):
                chosen_idx = llm_index - 1
        if chosen_idx is None:
            chosen_idx = _parse_selection(user_input, candidates)
        if chosen_idx is None:
            return

        chosen        = candidates[chosen_idx]
        tools, params = _resolve_tools_from_candidate(
            chosen, self._concept_to_agent)

        state["selected_index"] = chosen_idx
        state["tools"]          = tools
        state["parameters"]     = params
        state["phase"]          = ConversationPhase.CONCEPT_RESOLVED.value  # ← 改為 CONCEPT_RESOLVED

    # ------------------------------------------------------------------
    # CLI interactive mode
    # ------------------------------------------------------------------

    def run_interactive(self) -> ChatState:
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
        print(json.dumps({
            "query":          state.get("query"),
            "phase":          state.get("phase"),
            "selected_index": state.get("selected_index"),
            "candidates":     state.get("candidates"),
            "tools":          state.get("tools"),
            "parameters":     state.get("parameters"),
        }, ensure_ascii=False, indent=2))
        return state
