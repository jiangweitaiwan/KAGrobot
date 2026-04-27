# -*- coding: utf-8 -*-
"""entry.py  — FastAPI + LangGraph orchestrator.

Workflow:
  socratic_node
      └─ (resolved)             → resolved_node
      └─ (still running)        → END

  resolved_node
      └─ (no selected_path_id)  → END   # 回傳 graph_data，等前端點選
      └─ (has selected_path_id) → main_agent_node

  main_agent_node → END

entry_point router accepts phases: "socratic" | "resolved" | "main_agent"

[修改說明]
  - OrchestratorState 新增 confirmed_path_id 欄位，記錄上一次已確認的 path id
  - _build_handoff_task() 從 resolved_node 移至 main_agent_node 內呼叫
  - main_agent_node 比對 selected_path_id vs confirmed_path_id 判斷是否重選：
      * 重選時：重建 handoff_task、清空 main_state
      * 繼續對話時：沿用現有 main_state，以最後一則人類訊息繼續
  - resolved_node 不再負責建構 handoff_task，只負責 retrieve_graph（第一段）
  - entry_router 不需要修改
"""

from __future__ import annotations

import os
from settings import get_settings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]    = "main"

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from SocratesChatBot import (
    SocratesChatBot,
    ChatState,
    ConversationPhase,
    Neo4jReasoningRetriever,
)
from MainAgent import MainAgent


# ======================================================================
# Orchestrator state
# ======================================================================

class OrchestratorState(TypedDict, total=False):
    messages:           Annotated[List[BaseMessage], add_messages]
    phase:              str
    socrates_state:     ChatState
    main_state:         Annotated[List[BaseMessage], add_messages]
    handoff_task:       Optional[str]
    selected_path_id:   Optional[int]   # 前端用戶在圖譜上點選並確認的 path id
    confirmed_path_id:  Optional[int]   # 上一次已確認並執行分析的 path id


# ======================================================================
# Singletons
# ======================================================================

def _build_retriever() -> Neo4jReasoningRetriever:
    s        = get_settings()
    base_dir = Path(__file__).resolve().parent
    model_path    = str(base_dir / "sbert_onnx" / "sbert_multilingual.onnx")
    tokenizer_dir = str(base_dir / "sbert_onnx")
    retriever = Neo4jReasoningRetriever(
        s.neo4j_uri, s.neo4j_password, model_path, tokenizer_dir)
    try:
        retriever.load_embeddings()
    except Exception:
        pass
    return retriever


shared_retriever    = _build_retriever()
s                   = get_settings()
shared_socrates_bot = SocratesChatBot(
    retriever=shared_retriever,
    model=s.socrates_model,
    api_key=s.google_api_key,
)
shared_main_agent   = MainAgent(
    model=s.main_agent_model,
    api_key=s.google_api_key,
)


# ======================================================================
# Pure helpers
# ======================================================================

def _last_human_content(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _find_path_by_id(
    graph_data: Optional[Dict[str, Any]],
    path_id: int,
) -> Optional[Dict[str, Any]]:
    if not graph_data:
        return None
    for path in graph_data.get("paths", []):
        if path.get("path_id") == path_id:
            return path
    return None


def _build_resolved_graph_data(
    socrates_state: ChatState,
) -> Optional[Dict[str, Any]]:
    """附上 selected_path_id 標記，供前端立即 highlight 對應路徑。"""
    graph_data = socrates_state.get("graph_data")
    if not graph_data:
        return None

    selected_idx = socrates_state.get("selected_index")
    candidates   = socrates_state.get("candidates") or []
    if selected_idx is None or not candidates:
        return graph_data

    chosen  = candidates[selected_idx] if selected_idx < len(candidates) else None
    rc_name = (chosen.get("rc_name") or "").lower() if chosen else ""

    selected_path_id: Optional[int] = None
    for path in graph_data.get("paths", []):
        if (path.get("final_label") or "").lower() == rc_name:
            selected_path_id = path["path_id"]
            break

    return {**graph_data, "selected_path_id": selected_path_id}


def _build_handoff_task(
    socrates_state: ChatState,
    selected_path:  Optional[Dict[str, Any]] = None,
) -> str:
    """建構 MainAgent 的 handoff task 字串。

    優先使用用戶在知識圖譜上點選的 selected_path（含完整節點序列與論文資訊）。
    若無 selected_path，退回原本的 candidates[selected_index] 邏輯。
    """
    query = socrates_state.get("query") or "（未提供）"

    if selected_path:
        model_name  = selected_path.get("final_label") or "（未指定）"
        graph_data  = socrates_state.get("graph_data") or {}
        nodes_by_id = {n["id"]: n for n in graph_data.get("nodes", [])}

        # concept 節點序列
        path_labels: List[str] = []
        for nid in (selected_path.get("node_ids") or []):
            node = nodes_by_id.get(nid)
            if node and node.get("node_type") == "concept":
                path_labels.append(node.get("label", nid))
        path_description = " → ".join(path_labels) if path_labels else model_name

        # paper 節點資訊
        paper_info_lines: List[str] = []
        for nid in (selected_path.get("node_ids") or []):
            node = nodes_by_id.get(nid)
            if node and node.get("node_type") == "paper":
                meta        = node.get("meta") or {}
                line        = f"  論文：{node.get('label', '')}"
                premises    = meta.get("conceptual_premises") or ""
                constraints = meta.get("contextual_model_constraints") or ""
                if premises:
                    line += f"\n    前提假設：{premises}"
                if constraints:
                    line += f"\n    模型約束：{constraints}"
                paper_info_lines.append(line)
        paper_section = "\n".join(paper_info_lines) if paper_info_lines else "（無論文資訊）"

    else:
        model_name   = "（未指定）"
        candidates   = socrates_state.get("candidates") or []
        selected_idx = socrates_state.get("selected_index")
        if selected_idx is not None and candidates:
            chosen = candidates[selected_idx] if selected_idx < len(candidates) else None
            if chosen:
                model_name = chosen.get("rc_name") or model_name
        path_description = model_name
        paper_section    = "（無路徑資訊）"

    tools_info    = socrates_state.get("tools") or []
    primary_tools = [t["tool_name"] for t in tools_info if not t.get("is_dependency", False)]
    tools_str     = "、".join(primary_tools) if primary_tools else "（無）"

    return (
        f"使用者需求：{query}\n"
        f"選定模型：{model_name}\n"
        f"推理路徑：{path_description}\n"
        f"關聯論文資訊：\n{paper_section}\n"
        f"可用工具：{tools_str}\n"
        f"請根據以上需求、推理路徑與工具，開始執行分析。"
    )


# ======================================================================
# LangGraph nodes
# ======================================================================

def socratic_node(state: OrchestratorState) -> OrchestratorState:
    messages = state.get("messages") or []
    user_msg = _last_human_content(messages)
    if not user_msg:
        return {}

    old_socrates_state = state.get("socrates_state") or SocratesChatBot.init_state()
    new_socrates_state = shared_socrates_bot.process_message(old_socrates_state, user_msg)

    is_resolved = new_socrates_state.get("phase") == ConversationPhase.RESOLVED.value

    if is_resolved:
        model_name   = "（未指定）"
        candidates   = new_socrates_state.get("candidates") or []
        selected_idx = new_socrates_state.get("selected_index")
        if selected_idx is not None and candidates:
            chosen = candidates[selected_idx] if selected_idx < len(candidates) else None
            if chosen:
                model_name = chosen.get("rc_name") or model_name
        reply_text = (
            f"好的，已確認您選擇「{model_name}」。\n"
            f"知識圖譜載入中，請在右側圖譜中點選您想分析的路徑，"
            f"確認後系統將開始執行分析。"
        )
        next_phase = "resolved"
    else:
        reply_text = shared_socrates_bot.get_last_reply(new_socrates_state)
        next_phase = "socratic"

    return OrchestratorState(
        messages       = [AIMessage(content=reply_text)],
        phase          = next_phase,
        socrates_state = new_socrates_state,
    )


def resolved_node(state: OrchestratorState) -> OrchestratorState:
    """resolved 後的處理節點（僅負責第一段：retrieve_graph）。

    selected_path_id 存在時，直接導向 main_agent_node（由 resolved_router 處理）。
    selected_path_id 不存在時：
      - 呼叫 retrieve_graph()，寫入 socrates_state["graph_data"]
      - 回傳給前端，等待用戶點選路徑

    [修改] _build_handoff_task() 已移至 main_agent_node，
           resolved_node 不再負責建構 handoff_task。
    """
    socrates_state   = state.get("socrates_state")
    selected_path_id = state.get("selected_path_id")

    if socrates_state is None:
        return OrchestratorState(
            messages=[AIMessage(content="對話狀態遺失，請重新開始。")],
            phase="socratic",
        )

    # ── 尚無 path 選擇，執行 retrieve_graph ──────────────────────
    if selected_path_id is None:
        query = socrates_state.get("query") or ""
        try:
            graph_data = shared_retriever.retrieve_graph(query)
        except Exception:
            graph_data = None
        socrates_state["graph_data"] = graph_data

        return OrchestratorState(
            phase          = "resolved",
            socrates_state = socrates_state,
        )

    # ── 有 path 選擇，直接導向 main_agent（handoff_task 由 main_agent_node 建構）──
    return OrchestratorState(
        phase          = "main_agent",
        socrates_state = socrates_state,
    )


def main_agent_node(state: OrchestratorState) -> OrchestratorState:
    """MainAgent 執行節點。

    [修改] 支援在 main_agent 階段重選 path_id：
      - 比對 selected_path_id vs confirmed_path_id 判斷是否為重選
      - 重選時：重新呼叫 _build_handoff_task() 建構新任務、清空 main_state
      - 繼續對話時：沿用現有 main_state，以最後一則人類訊息繼續
    """
    socrates_state = state.get("socrates_state")
    if not socrates_state:
        return OrchestratorState(
            messages=[AIMessage(content="目前尚未完成需求釐清，請先重新描述你的需求。")],
            phase="socratic",
        )

    selected_path_id  = state.get("selected_path_id")
    confirmed_path_id = state.get("confirmed_path_id")

    # ── 判斷是否為重選 ─────────────────────────────────────────────
    is_reselect = (
        selected_path_id is not None
        and selected_path_id != confirmed_path_id
    )

    if is_reselect:
        # 重選：重建 handoff_task，清空舊歷史
        selected_path = _find_path_by_id(
            socrates_state.get("graph_data"),
            selected_path_id,
        )
        user_msg  = _build_handoff_task(socrates_state, selected_path)
        history: List[BaseMessage] = []

        path_label = (selected_path or {}).get("final_label") or f"路徑 {selected_path_id}"
        reselect_notice = AIMessage(
            content=f"已切換至新路徑「{path_label}」，重新開始分析。"
        )
    else:
        # 繼續對話：使用最後一則人類訊息
        user_msg = _last_human_content(state.get("messages") or [])
        history  = list(state.get("main_state") or [])
        reselect_notice = None

    if not user_msg:
        return {}

    result = shared_main_agent.chat(
        user_input=user_msg,
        socrates_state=socrates_state,
        history=history,
    )

    output_messages: List[BaseMessage] = []
    if reselect_notice:
        output_messages.append(reselect_notice)
    output_messages.append(AIMessage(content=result["content"]))

    return OrchestratorState(
        messages          = output_messages,
        phase             = "main_agent",
        socrates_state    = socrates_state,
        handoff_task      = None,
        main_state        = result.get("messages") or [],
        # 更新 confirmed_path_id：重選時更新為新 id，否則保持不變
        confirmed_path_id = selected_path_id if is_reselect else confirmed_path_id,
    )


# ======================================================================
# LangGraph routers
# ======================================================================

def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase in ("socratic", "resolved", "main_agent"):
        return phase
    return "socratic"


def socratic_router(state: OrchestratorState) -> str:
    return "resolved" if state.get("phase") == "resolved" else "end"


def resolved_router(state: OrchestratorState) -> str:
    return "main_agent" if state.get("phase") == "main_agent" else "end"


# ======================================================================
# Build workflow
# ======================================================================

workflow = StateGraph(OrchestratorState)
workflow.add_node("socratic",   socratic_node)
workflow.add_node("resolved",   resolved_node)
workflow.add_node("main_agent", main_agent_node)

workflow.set_conditional_entry_point(
    entry_router,
    {
        "socratic":   "socratic",
        "resolved":   "resolved",
        "main_agent": "main_agent",
    },
)
workflow.add_conditional_edges(
    "socratic", socratic_router,
    {"resolved": "resolved", "end": END},
)
workflow.add_conditional_edges(
    "resolved", resolved_router,
    {"main_agent": "main_agent", "end": END},
)
workflow.add_edge("main_agent", END)

memory           = MemorySaver()
orchestrator_app = workflow.compile(checkpointer=memory)


# ======================================================================
# FastAPI
# ======================================================================

class ChatRequest(BaseModel):
    session_id:       str
    message:          str
    phase:            Optional[str] = None
    selected_path_id: Optional[int] = None   # 前端圖譜點選後傳入


class ChatResponse(BaseModel):
    reply:          str
    phase:          str
    socrates_phase: Optional[str]            = None
    graph_data:     Optional[Dict[str, Any]] = None


api = FastAPI()

cors_origins = [o.strip() for o in s.cors_allow_origins.split(",") if o.strip()]
if cors_origins:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@api.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    config        = {"configurable": {"thread_id": request.session_id}}
    initial_phase = request.phase or "socratic"

    invoke_input: dict = {
        "messages": [HumanMessage(content=request.message)],
        "phase":    initial_phase,
    }
    if request.selected_path_id is not None:
        invoke_input["selected_path_id"] = request.selected_path_id

    result = await orchestrator_app.ainvoke(invoke_input, config=config)

    messages   = result.get("messages") or []
    last_msg   = messages[-1] if messages else AIMessage(content="")
    reply_text = getattr(last_msg, "content", str(last_msg))

    socrates_state = result.get("socrates_state")
    socrates_phase = (
        socrates_state.get("phase")
        if socrates_state and isinstance(socrates_state, dict)
        else None
    )

    # graph_data：僅在 resolved_node 第一段結束（等待前端點選）時回傳
    graph_data: Optional[Dict[str, Any]] = None
    if (
        socrates_state
        and isinstance(socrates_state, dict)
        and socrates_phase == ConversationPhase.RESOLVED.value
        and result.get("phase") == "resolved"
        and request.selected_path_id is None
    ):
        graph_data = _build_resolved_graph_data(socrates_state)

    return ChatResponse(
        reply          = reply_text,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        graph_data     = graph_data,
    )
