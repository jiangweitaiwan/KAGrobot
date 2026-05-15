"""
core/socrates_bot.py
─────────────────────────────────────────────────────────────
SocratesChatBot：需求釐清 + 協助使用者選擇推理路徑。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import StructuredTool
from core.google_llm import create_chat_google_generative_ai
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import neo4jreasoningretriever as nrr
from core.registry import (
    AGENTS_REGISTRY, AgentType, CONCEPT_TO_AGENT, TOOLS_REGISTRY,
)
from core.llm_retry import call_with_retry

Neo4jReasoningRetriever = nrr.Neo4jReasoningRetriever


# ═══════════════════════════════════════════════════════════════
# ConversationPhase & ChatState
# ═══════════════════════════════════════════════════════════════

class ConversationPhase(Enum):
    RUNNING          = "running"
    CONCEPT_RESOLVED = "concept_resolved"
    PATH_CONFIRMED   = "path_confirmed"
    RESOLVED         = "concept_resolved"


class ChatState(TypedDict, total=False):
    messages:       List[BaseMessage]
    agent_messages: List[BaseMessage]
    phase:          str
    query:          Optional[str]
    candidates:     Optional[List[dict]]
    selected_index: Optional[int]
    tools:          Optional[List[dict]]
    parameters:     Optional[List[dict]]
    intent_types:   Optional[List[str]]


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
{"ai_message": "<繁體中文回覆>", "intent_types": [...]}

- 對話以繁體中文；retrieve_concepts 的查詢語句用英文。
- intent_types：在使用者需求已足夠清晰時，填入對應的分析類型標籤（可複數）。
  需求尚不明確時填入空列表 []。
  候選值：causal_mechanism, equilibrium_analysis, forecasting, event_study,
          structural_estimation, policy_evaluation, asset_pricing, descriptive
  範例：使用者說「蘋果新產品對股價的影響」→ ["event_study", "asset_pricing"]
  範例：使用者說「明年台灣 GDP 預測」→ ["forecasting"]
  範例：使用者說「利率政策對通膨的傳導機制」→ ["causal_mechanism", "policy_evaluation"]
- 路徑確認由使用者在前端操作，不在此處處理。
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


def _resolve_tools_from_candidates(
    candidates: List[dict],
    concept_to_agent: Dict[str, AgentType],
) -> tuple:
    seen_tools: set        = set()
    resolved_tools:  List[dict] = []
    resolved_params: List[dict] = []

    for cand in candidates:
        rc_name = cand.get("rc_name") or ""
        agent_t = _resolve_agent_type(rc_name, concept_to_agent)
        if agent_t is None:
            continue

        base_tools = AGENTS_REGISTRY.get(agent_t, {}).get("tools", [])
        all_tools  = collect_tools_with_deps(base_tools)

        for tname in all_tools:
            if tname in seen_tools:
                continue
            seen_tools.add(tname)

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


# ═══════════════════════════════════════════════════════════════
# Reasoning chain display helpers
# ═══════════════════════════════════════════════════════════════

def _extract_selected_chain(socrates_state: ChatState) -> Optional[dict]:
    candidates: List[dict] = socrates_state.get("candidates") or []
    idx: Optional[int]     = socrates_state.get("selected_index")
    if idx is None or not candidates:
        return None
    if idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


def _format_selected_chain(candidate: Optional[dict]) -> str:
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

        self._base_llm = create_chat_google_generative_ai(
            model=model,
            temperature=0.1,
            api_key=api_key,
        )

        # 兩個工具在 __init__ 建立一次，不再需要每次傳入 all_graphs
        self._retrieve_tool       = self._make_retrieve_tool()
        self._concept_detail_tool = self._make_concept_detail_tool()
        self._graph = None

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
        )

    def process_message(
        self,
        state:         ChatState,
        user_input:    str,
        graph_context: Optional[str]            = None,
        all_graphs:    Optional[Dict[int, Any]] = None,  # 保留參數相容性，不再使用
    ) -> ChatState:
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
                + "\n\n請根據上方各候選的下游終點與基礎概念向使用者說明分析方向。"
                + "若使用者追問特定概念的理論細節，呼叫 retrieve_concept_detail。\n"
                + "引導使用者在右側圖譜點選路徑後按確認按鈕。"
            )
        else:
            effective_prompt = base_prompt

        llm_with_tools = self._base_llm.bind_tools([
            self._retrieve_tool,
            self._concept_detail_tool,
        ])

        system_msg   = SystemMessage(content=effective_prompt)
        graph_result = self._build_graph(llm_with_tools).invoke({
            "messages":   [system_msg] + normalized_prior + [HumanMessage(content=user_input)],
            "candidates": state.get("candidates") or [],
            "query":      state.get("query"),
        })
        new_agent_msgs: List[BaseMessage] = graph_result["messages"]
        state["agent_messages"] = new_agent_msgs

        if graph_result.get("query"):
            state["query"] = graph_result["query"]

        new_candidates = graph_result.get("candidates")
        if new_candidates:
            state["candidates"] = new_candidates
            tools, params = _resolve_tools_from_candidates(
                new_candidates, self._concept_to_agent)
            state["tools"]      = tools
            state["parameters"] = params
            state["phase"] = ConversationPhase.CONCEPT_RESOLVED.value

        bot_resp    = self._parse_bot_response(new_agent_msgs)
        ai_text     = bot_resp.get("ai_message", "")
        new_intents = bot_resp.get("intent_types", [])

        if new_intents:
            state["intent_types"] = new_intents

        state["messages"].append(AIMessage(content=ai_text))

        return state

    def confirm_path(self, state: ChatState) -> ChatState:
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

    def _build_graph(self, llm_with_tools):
        retrieve_tool       = self._retrieve_tool
        concept_detail_tool = self._concept_detail_tool
        tool_map = {
            retrieve_tool.name:       retrieve_tool,
            concept_detail_tool.name: concept_detail_tool,
        }

        def agent_node(state: _InternalState) -> dict:
            response = call_with_retry(lambda: llm_with_tools.invoke(state["messages"]))
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

                if tc["name"] == retrieve_tool.name:
                    if data.get("found"):
                        state_updates["candidates"] = data.get("candidates", [])
                        state_updates["query"]      = data.get("query", "")
                        slim = data.get("display") or "（查無結果）"
                    else:
                        slim = data.get("message") or raw_result

                elif tc["name"] == concept_detail_tool.name:
                    if data.get("found"):
                        slim = _format_concept_detail(data)
                    else:
                        slim = data.get("message") or raw_result

                else:
                    slim = raw_result

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
                "若使用者希望換分析方向，亦可再次呼叫。"
            ),
        )

    # ------------------------------------------------------------------
    # retrieve_concept_detail tool — 直接查 Neo4j，不依賴 all_graphs
    # ------------------------------------------------------------------

    def _make_concept_detail_tool(self) -> StructuredTool:
        retriever = self._retriever

        def retrieve_concept_detail(rc_name: str) -> str:
            """查詢指定 concept 的描述與所屬論文的理論前提及模型約束。"""
            result = retriever.retrieve_concept_detail(rc_name)
            return json.dumps(result, ensure_ascii=False)

        return StructuredTool.from_function(
            func=retrieve_concept_detail,
            name="retrieve_concept_detail",
            description=(
                "查詢指定 Resulting Concept 的詳細資訊，包含：\n"
                "  - concept 的 description（概念說明）\n"
                "  - 產生此 concept 的 paper 標題\n"
                "  - paper 的 conceptual_premises（理論前提假設）\n"
                "  - paper 的 contextual_model_constraints（模型情境約束）\n"
                "當使用者詢問某個概念或路徑的理論背景、適用條件時呼叫。\n"
                "輸入 rc_name 為 Resulting Concept 的名稱（允許部分匹配）。"
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
            if "ai_message" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {"ai_message": raw, "intent_types": []}

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
        import json as _json
        print(_json.dumps({
            "query":        state.get("query"),
            "phase":        state.get("phase"),
            "candidates":   state.get("candidates"),
            "tools":        state.get("tools"),
            "intent_types": state.get("intent_types"),
        }, ensure_ascii=False, indent=2))
        return state


# ═══════════════════════════════════════════════════════════════
# Formatting helper
# ═══════════════════════════════════════════════════════════════

def _format_concept_detail(data: dict) -> str:
    lines: List[str] = []
    lines.append(f"概念：{data.get('rc_name', '')}")
    if data.get("description"):
        lines.append(f"說明：{data['description']}")
    if data.get("function"):
        lines.append(f"功能：{data['function']}")
    if data.get("analysis_types"):
        lines.append(f"分析類型：{', '.join(data['analysis_types'])}")
    papers = data.get("papers") or []
    if papers:
        lines.append("相關論文：")
        for p in papers:
            lines.append(f"  ▸ {p.get('title', '（未命名）')}")
            if p.get("premises"):
                lines.append(f"    前提假設：{p['premises']}")
            if p.get("constraints"):
                lines.append(f"    模型約束：{p['constraints']}")
    return "\n".join(lines)