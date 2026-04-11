"""
core/socrates_bot.py
─────────────────────────────────────────────────────────────
SocratesChatBot：需求釐清 + 模型選擇的對話機器人。

Changes vs original:
  • ChatState gains  graph_data: Optional[Dict[str, Any]]
  • _make_retrieve_tool: also calls retriever.retrieve_graph() and
    embeds the result as "graph_data" in the tool's JSON response.
  • tool_executor_node: reads "graph_data" from the tool response and
    stores it in LangGraph state so entry.py can forward it to the
    front-end.
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
ReasoningChain          = nrr.ReasoningChain
ConceptNode             = nrr.ConceptNode
PaperNode               = nrr.PaperNode


# ═══════════════════════════════════════════════════════════════
# ConversationPhase & ChatState
# ═══════════════════════════════════════════════════════════════

class ConversationPhase(Enum):
    RUNNING  = "running"
    RESOLVED = "resolved"


class ChatState(TypedDict, total=False):
    messages:         List[BaseMessage]
    agent_messages:   List[BaseMessage]
    phase:            str
    query:            Optional[str]
    reasoning_chains: Optional[List[dict]]
    candidates:       Optional[List[dict]]
    selected_index:   Optional[int]
    tools:            Optional[List[dict]]
    parameters:       Optional[List[dict]]
    # ── NEW ──────────────────────────────────────────────────
    graph_data:       Optional[Dict[str, Any]]   # D3 graph payload


# ═══════════════════════════════════════════════════════════════
# Tool dependency resolver  (unchanged)
# ═══════════════════════════════════════════════════════════════

def collect_tools_with_deps(tool_names: List[str]) -> List[str]:
    visited: set  = set()
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
    chains_raw: Optional[List[dict]]
    query:      Optional[str]
    graph_data: Optional[Dict[str, Any]]   # ← NEW


# ═══════════════════════════════════════════════════════════════
# System prompt & OutputSchema  (unchanged)
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一位專業經濟學顧問，目標是協助使用者從知識圖譜的
Resulting Concept 中，選出最符合其實際需求的模型或分析方法。

## 工作流程

### 階段 1：需求釐清
- 了解使用者需求

### 階段 2：查詢知識圖譜
- 呼叫 `retrieve_concepts` tool，傳入精確的**英文**查詢語句。
- 若結果為空（found=false），告知使用者並請重新描述需求後再試。

### 階段 3：展示候選，等待使用者選擇
- 將 `retrieve_concepts` 回傳的 `display` 欄位內容展示給使用者。
- 請使用者輸入選項編號（如 "1"）或模型名稱。
- 此時 action = "ask"。

### 階段 4：確認選擇
- 使用者明確給出編號或可辨識的名稱後：
  * 在 ai_message 中說明已選擇的項目與摘要。
  * 將 action 設為 "resolved"。
- 若使用者的輸入模糊或無法對應到任何候選，繼續追問，action 仍為 "ask"。

## 輸出格式（強制）
每輪必須輸出合法 JSON，不得有任何前綴文字或 markdown code block：
{"action": "ask"|"resolved", "ai_message": "<繁體中文回覆>", "selected_index": null}

**selected_index 規則（非常重要）：**
- action="ask" 時：selected_index 必須為 null。
- action="resolved" 時：selected_index 必須填入使用者所選項目的**編號**（整數，與展示的 [1][2]... 一致）。
  例如：使用者選了 [2]，則 selected_index=2。
  你必須在此時確保 selected_index 不為 null，這是系統路由的關鍵依據。

- 對話以繁體中文；retrieve_concepts 的查詢語句用英文。
- 未完成選擇前，action 絕對不能為 "resolved"。
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
# Serialization helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _concept_to_dict(c: ConceptNode) -> dict:
    return {
        "element_id":          c.element_id,
        "name":                c.name,
        "type":                c.type,
        "matched_name":        getattr(c, "matched_name", None),
        "matched_description": getattr(c, "matched_description", None),
    }


def _paper_to_dict(p: Optional[PaperNode]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "element_id":                   p.element_id,
        "title":                        p.title,
        "conceptual_premises":          p.conceptual_premises,
        "contextual_model_constraints": p.contextual_model_constraints,
    }


def _serialize_chain(chain: ReasoningChain) -> dict:
    return {
        "start_concept": _concept_to_dict(chain.start_concept),
        "steps": [
            {
                "from_concept": _concept_to_dict(s.from_concept),
                "via_paper":    _paper_to_dict(s.via_paper),
                "to_concept":   _concept_to_dict(s.to_concept),
            }
            for s in chain.steps
        ],
        "final_resulting_concept": _concept_to_dict(chain.final_resulting_concept),
        "final_paper":             _paper_to_dict(chain.final_paper),
    }


# ═══════════════════════════════════════════════════════════════
# Internal helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _resolve_agent_type(
    concept: ConceptNode,
    concept_to_agent: Dict[str, AgentType],
) -> Optional[AgentType]:
    name = (concept.name or "").lower().strip()
    if name in concept_to_agent:
        return concept_to_agent[name]
    for key, agent in concept_to_agent.items():
        if key.lower().strip() in name or name in key.lower().strip():
            return agent
    return None


def _build_candidates_from_chains(
    chains: List[ReasoningChain],
    concept_to_agent: Dict[str, AgentType],
) -> List[dict]:
    candidates: List[dict] = []
    for idx, chain in enumerate(chains, 1):
        rc         = chain.final_resulting_concept
        paper      = chain.final_paper
        agent_type = _resolve_agent_type(rc, concept_to_agent)
        primary_tool = None
        if agent_type:
            tools_list   = AGENTS_REGISTRY.get(agent_type, {}).get("tools", [])
            primary_tool = tools_list[0] if tools_list else None
        candidates.append({
            "index":        idx,
            "rc_name":      rc.name,
            "agent_type":   agent_type.value if agent_type else None,
            "primary_tool": primary_tool,
            "paper_title":  paper.title if paper else None,
            "conceptual_premises":          paper.conceptual_premises if paper else None,
            "contextual_model_constraints": paper.contextual_model_constraints if paper else None,
        })
    return candidates


def _resolve_tools_from_candidate(
    chosen_candidate: dict,
    chains_raw: List[dict],
    concept_to_agent: Dict[str, AgentType],
) -> tuple:
    rc_name = (chosen_candidate.get("rc_name") or "").lower()
    seen_tools:      set = set()
    resolved_tools:  List[dict] = []
    resolved_params: List[dict] = []

    for chain_dict in chains_raw:
        fc      = chain_dict.get("final_resulting_concept") or {}
        fc_name = (fc.get("name") or "").lower()
        if not fc_name:
            continue
        if not (fc_name == rc_name or rc_name in fc_name or fc_name in rc_name):
            continue

        dummy   = ConceptNode(element_id=fc.get("element_id", ""),
                              name=fc.get("name", ""), type=fc.get("type"))
        agent_t = _resolve_agent_type(dummy, concept_to_agent)
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
                "matched_concept": fc.get("name"),
                "is_dependency":   is_dep,
            })
            if entry.get("parameters"):
                resolved_params.append({
                    "tool_name":  tname,
                    "parameters": entry["parameters"],
                })
        break

    return resolved_tools, resolved_params


def _parse_selection(text: str, candidates: List[dict]) -> Optional[int]:
    clean = text.strip()
    for token in clean.replace("、", " ").replace("，", " ").split():
        if token.isdigit():
            num = int(token)
            if 1 <= num <= len(candidates):
                return num - 1
    lower = clean.lower()
    for i, cand in enumerate(candidates):
        rc_name   = (cand.get("rc_name")      or "").lower()
        tool_name = (cand.get("primary_tool") or "").lower()
        if (rc_name   and (rc_name   in lower or lower in rc_name)) or \
           (tool_name and (tool_name in lower or lower in tool_name)):
            return i
    return None


# ═══════════════════════════════════════════════════════════════
# Reasoning chain helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _extract_selected_chain(socrates_state: ChatState) -> Optional[dict]:
    chains: List[dict] = socrates_state.get("reasoning_chains") or []
    idx: Optional[int] = socrates_state.get("selected_index")
    if idx is None or not chains:
        return None
    if idx < 0 or idx >= len(chains):
        return None
    return chains[idx]


def _format_selected_chain(chain: Optional[dict]) -> str:
    if chain is None:
        return "（未選定推理鏈）"
    rc_names: List[str] = []
    start_name = (chain.get("start_concept") or {}).get("name") or ""
    if start_name:
        rc_names.append(start_name)
    for step in chain.get("steps") or []:
        to_name = (step.get("to_concept") or {}).get("name") or ""
        if to_name and to_name not in rc_names:
            rc_names.append(to_name)
    final_name = (chain.get("final_resulting_concept") or {}).get("name") or ""
    if final_name and final_name not in rc_names:
        rc_names.append(final_name)
    if not rc_names:
        return "（推理鏈無節點資訊）"
    nodes_display = [
        f"{name}（最終目標）" if i == len(rc_names) - 1 else name
        for i, name in enumerate(rc_names)
    ]
    return "選定分析路徑：\n  " + "  →  ".join(nodes_display)


# ═══════════════════════════════════════════════════════════════
# SocratesChatBot
# ═══════════════════════════════════════════════════════════════

class SocratesChatBot:
    """需求釐清 + 模型選擇的蘇格拉底對話 Agent。"""

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
    # Public API  (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def init_state() -> ChatState:
        return ChatState(
            messages         = [],
            agent_messages   = [],
            phase            = ConversationPhase.RUNNING.value,
            query            = None,
            reasoning_chains = None,
            candidates       = None,
            selected_index   = None,
            tools            = None,
            parameters       = None,
            graph_data       = None,   # ← NEW
        )

    def process_message(self, state: ChatState, user_input: str) -> ChatState:
        if state.get("phase") == ConversationPhase.RESOLVED.value:
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

        system_msg = first_system or SystemMessage(content=_SYSTEM_PROMPT)
        graph_result = self._graph.invoke({
            "messages":   [system_msg] + normalized_prior + [HumanMessage(content=user_input)],
            "candidates": state.get("candidates") or [],
            "chains_raw": state.get("chains_raw") or [],
            "query":      state.get("query"),
            "graph_data": state.get("graph_data"),   # ← pass through
        })
        new_agent_msgs: List[BaseMessage] = graph_result["messages"]
        state["agent_messages"] = new_agent_msgs

        if graph_result.get("query"):
            state["query"] = graph_result["query"]
        if graph_result.get("candidates"):
            state["candidates"] = graph_result["candidates"]
        if graph_result.get("chains_raw"):
            state["reasoning_chains"] = graph_result["chains_raw"]
        # ── NEW: persist graph_data onto state ──
        if graph_result.get("graph_data"):
            state["graph_data"] = graph_result["graph_data"]

        bot_resp = self._parse_bot_response(new_agent_msgs)
        ai_text  = bot_resp.get("ai_message", "")
        action   = bot_resp.get("action", "ask")

        state["messages"].append(AIMessage(content=ai_text))

        if action == "resolved":
            self._handle_resolved(state, user_input, bot_resp)
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
                    state_updates["chains_raw"] = data.get("chains_raw", [])
                    state_updates["query"]      = data.get("query", "")
                    # ── NEW: capture graph_data ──────────────────────
                    if data.get("graph_data"):
                        state_updates["graph_data"] = data["graph_data"]
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
    # retrieve_concepts tool  — UPDATED
    # ------------------------------------------------------------------

    def _make_retrieve_tool(self) -> StructuredTool:
        retriever        = self._retriever
        top_k            = self._top_k
        concept_to_agent = self._concept_to_agent

        def retrieve_concepts(query: str) -> str:
            """查詢知識圖譜，找出與使用者需求最相關的 Resulting Concept 清單。"""
            # ── original: linear chains for candidate list ──
            chains: List[ReasoningChain] = retriever.retrieve(
                query, top_k_entities=top_k)

            # ── NEW: full branching graph for D3 visualisation ──
            try:
                graph_data = retriever.retrieve_graph(
                    query, top_k_entities=top_k)
            except Exception:
                graph_data = None

            if not chains:
                return json.dumps({
                    "found":      False,
                    "message":    "No relevant concepts found in the knowledge graph.",
                    "candidates": [],
                    "chains_raw": [],
                    "display":    "",
                    "query":      query,
                    "graph_data": graph_data,
                }, ensure_ascii=False)

            chains_raw = [_serialize_chain(c) for c in chains]
            candidates = _build_candidates_from_chains(chains, concept_to_agent)

            lines: List[str] = []
            for c in candidates:
                tool_name = c.get("primary_tool") or "（尚未對應）"
                tool_desc = TOOLS_REGISTRY.get(tool_name, {}).get("description", "")
                desc_part = f" — {tool_desc}" if tool_desc else ""
                lines.append(
                    f"[{c['index']}] {c['rc_name'] or '（未命名）'}"
                    f"｜工具：{tool_name}{desc_part}"
                )

            return json.dumps({
                "found":      True,
                "query":      query,
                "candidates": candidates,
                "chains_raw": chains_raw,
                "display":    "\n".join(lines),
                "graph_data": graph_data,   # ← NEW
            }, ensure_ascii=False)

        return StructuredTool.from_function(
            func=retrieve_concepts,
            name="retrieve_concepts",
            description=(
                "查詢知識圖譜，找出與使用者需求最相關的 Resulting Concept 清單。"
                "需求釐清完成後呼叫此 tool，傳入精確的英文查詢語句。"
            ),
        )

    # ------------------------------------------------------------------
    # JSON parsing  (unchanged)
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
    # Handle resolved  (unchanged)
    # ------------------------------------------------------------------

    def _handle_resolved(
        self,
        state:      ChatState,
        user_input: str,
        bot_resp:   dict,
    ) -> None:
        candidates = state.get("candidates") or []
        chains_raw = state.get("reasoning_chains") or []
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
            chosen, chains_raw, self._concept_to_agent)

        state["selected_index"] = chosen_idx
        state["tools"]          = tools
        state["parameters"]     = params
        state["phase"]          = ConversationPhase.RESOLVED.value

    # ------------------------------------------------------------------
    # CLI interactive mode  (unchanged)
    # ------------------------------------------------------------------

    def run_interactive(self) -> ChatState:
        state = self.init_state()
        print("=== SocratesChatBot ===\n")
        state = self.process_message(state, "你好，請開始引導我。")
        print(f"Bot: {self.get_last_reply(state)}\n")
        while state.get("phase") != ConversationPhase.RESOLVED.value:
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
            "query":            state.get("query"),
            "selected_index":   state.get("selected_index"),
            "tools":            state.get("tools"),
            "parameters":       state.get("parameters"),
            "reasoning_chains": state.get("reasoning_chains"),
        }, ensure_ascii=False, indent=2))
        return state
