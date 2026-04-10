# -*- coding: utf-8 -*-
"""entry.py  — FastAPI + LangGraph orchestrator.

Changes vs original:
  • ChatResponse gains an optional `graph_data` field.
  • /chat endpoint attaches graph_data in two cases:
      1. socrates phase just turned "running" and retrieve_graph data is
         available on socrates_state  (candidate browsing stage).
      2. socrates phase == "resolved"  (user confirmed a selection).
  • SocratesChatBot._make_retrieve_tool now stores the raw retrieve_graph
    payload into socrates_state["graph_data"] so we can pass it through.
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
    messages:       Annotated[List[BaseMessage], add_messages]
    phase:          str
    socrates_state: ChatState
    main_state:     Annotated[List[BaseMessage], add_messages]
    handoff_task:   Optional[str]


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


shared_retriever = _build_retriever()
s = get_settings()

shared_socrates_bot = SocratesChatBot(
    retriever=shared_retriever,
    model=s.socrates_model,
    api_key=s.google_api_key,
)
shared_main_agent = MainAgent(
    model=s.main_agent_model,
    api_key=s.google_api_key,
)


# ======================================================================
# Handoff task builder  (unchanged)
# ======================================================================

def _build_handoff_task(socrates_state: ChatState) -> str:
    query      = socrates_state.get("query") or "（未提供）"
    model_name = "（未指定）"
    candidates = socrates_state.get("candidates") or []
    selected_idx = socrates_state.get("selected_index")
    if selected_idx is not None and candidates:
        chosen = candidates[selected_idx] if selected_idx < len(candidates) else None
        if chosen:
            model_name = chosen.get("rc_name") or model_name

    tools_info   = socrates_state.get("tools") or []
    primary_tools = [t["tool_name"] for t in tools_info
                     if not t.get("is_dependency", False)]
    tools_str = "、".join(primary_tools) if primary_tools else "（無）"

    return (
        f"使用者需求：{query}\n"
        f"選定模型：{model_name}\n"
        f"可用工具：{tools_str}\n"
        f"請根據以上需求與工具，開始執行分析。"
    )


# ======================================================================
# Graph data extractor
# ======================================================================

def _extract_graph_data(socrates_state: Optional[ChatState]) -> Optional[Dict[str, Any]]:
    """Pull the graph_data dict stored by _make_retrieve_tool, if present."""
    if not socrates_state or not isinstance(socrates_state, dict):
        return None
    return socrates_state.get("graph_data")


def _build_resolved_graph_data(socrates_state: ChatState) -> Optional[Dict[str, Any]]:
    """After resolution, annotate the graph_data with the selected path_id
    so the front-end knows which path to highlight immediately."""
    gd = _extract_graph_data(socrates_state)
    if not gd:
        return None

    selected_idx = socrates_state.get("selected_index")
    candidates   = socrates_state.get("candidates") or []

    if selected_idx is None or not candidates:
        return gd

    chosen    = candidates[selected_idx] if selected_idx < len(candidates) else None
    rc_name   = (chosen.get("rc_name") or "").lower() if chosen else ""

    # find the path whose final_label matches the chosen RC name
    selected_path_id: Optional[int] = None
    for path in gd.get("paths", []):
        if (path.get("final_label") or "").lower() == rc_name:
            selected_path_id = path["path_id"]
            break

    return {**gd, "selected_path_id": selected_path_id}


# ======================================================================
# LangGraph nodes
# ======================================================================

def _last_human_content(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def socratic_node(state: OrchestratorState) -> OrchestratorState:
    messages = state.get("messages") or []
    user_msg = _last_human_content(messages)
    if not user_msg:
        return {}

    old_socrates_state = state.get("socrates_state") or SocratesChatBot.init_state()
    new_socrates_state = shared_socrates_bot.process_message(
        old_socrates_state, user_msg)

    is_resolved = (
        new_socrates_state.get("phase") == ConversationPhase.RESOLVED.value)
    next_phase  = "main_agent" if is_resolved else "socratic"

    if is_resolved:
        model_name = "（未指定）"
        candidates   = new_socrates_state.get("candidates") or []
        selected_idx = new_socrates_state.get("selected_index")
        if selected_idx is not None and candidates:
            chosen = candidates[selected_idx] if selected_idx < len(candidates) else None
            if chosen:
                model_name = chosen.get("rc_name") or model_name
        reply_text   = (
            f"好的，已確認您選擇「{model_name}」。\n"
            f"我現在開始為您進行分析，請稍候⋯⋯"
        )
        handoff_task = _build_handoff_task(new_socrates_state)
    else:
        reply_text   = shared_socrates_bot.get_last_reply(new_socrates_state)
        handoff_task = None

    return OrchestratorState(
        messages       = [AIMessage(content=reply_text)],
        phase          = next_phase,
        socrates_state = new_socrates_state,
        handoff_task   = handoff_task,
    )


def main_agent_node(state: OrchestratorState) -> OrchestratorState:
    socrates_state = state.get("socrates_state")
    if not socrates_state:
        return OrchestratorState(
            messages=[AIMessage(content="目前尚未完成需求釐清，請先重新描述你的需求。")],
            phase="socratic",
        )

    handoff_task = state.get("handoff_task")
    if handoff_task:
        user_msg = handoff_task
    else:
        messages = state.get("messages") or []
        user_msg = _last_human_content(messages)
        if not user_msg:
            return {}

    history: List[BaseMessage] = state.get("main_state") or []
    result = shared_main_agent.chat(
        user_input=user_msg,
        socrates_state=socrates_state,
        history=history,
    )

    return OrchestratorState(
        messages       = [AIMessage(content=result["content"])],
        phase          = "main_agent",
        socrates_state = socrates_state,
        handoff_task   = None,
        main_state     = result.get("messages") or [],
    )


# ======================================================================
# LangGraph workflow
# ======================================================================

def router(state: OrchestratorState) -> str:
    return state.get("phase") or "socratic"


def socratic_to_next_router(state: OrchestratorState) -> str:
    return "main_agent" if state.get("phase") == "main_agent" else "end"


workflow = StateGraph(OrchestratorState)
workflow.add_node("socratic",   socratic_node)
workflow.add_node("main_agent", main_agent_node)
workflow.set_conditional_entry_point(router, {
    "socratic":   "socratic",
    "main_agent": "main_agent",
})
workflow.add_conditional_edges("socratic", socratic_to_next_router, {
    "main_agent": "main_agent",
    "end":        END,
})
workflow.add_edge("main_agent", END)

memory           = MemorySaver()
orchestrator_app = workflow.compile(checkpointer=memory)


# ======================================================================
# FastAPI
# ======================================================================

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    phase:      Optional[str] = None


class ChatResponse(BaseModel):
    reply:          str
    phase:          str
    socrates_phase: Optional[str]          = None
    # NEW — D3 graph payload; present during socratic stage and on resolution
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
    config = {"configurable": {"thread_id": request.session_id}}
    initial_phase = request.phase or "socratic"

    result = await orchestrator_app.ainvoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "phase":    initial_phase,
        },
        config=config,
    )

    messages: List[BaseMessage] = result["messages"]
    last_msg   = messages[-1] if messages else AIMessage(content="")
    reply_text = getattr(last_msg, "content", str(last_msg))

    socrates_state = result.get("socrates_state")
    socrates_phase = (
        socrates_state.get("phase")
        if socrates_state and isinstance(socrates_state, dict)
        else None
    )

    # ── graph_data assembly ──────────────────────────────────
    graph_data: Optional[Dict[str, Any]] = None
    if socrates_state and isinstance(socrates_state, dict):
        if socrates_phase == ConversationPhase.RESOLVED.value:
            # resolved: annotate which path was selected
            graph_data = _build_resolved_graph_data(socrates_state)
        else:
            # still running: pass raw graph for browsing
            graph_data = _extract_graph_data(socrates_state)

    return ChatResponse(
        reply          = reply_text,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        graph_data     = graph_data,
    )
