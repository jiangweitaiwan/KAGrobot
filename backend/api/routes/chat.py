# -*- coding: utf-8 -*-
"""api/routes/chat.py — POST /chat endpoint。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from graph_service.projector import GraphProjector

router = APIRouter()



# ── Schema ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:        str
    message:           str
    phase:             Optional[str] = None
    selected_path_id:  Optional[int] = None
    selected_rc_eid:   Optional[str] = None
    selected_rc_label: Optional[str] = None


class ChatResponse(BaseModel):
    reply:          str
    phase:          str
    socrates_phase: Optional[str] = None
    skill_used:     Optional[str] = None


# ── Route ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    body:    ChatRequest,
) -> ChatResponse:
    orchestrator  = request.app.state.orchestrator
    config        = {"configurable": {"thread_id": body.session_id}}
    initial_phase = body.phase or "socratic"

    invoke_input: dict = {
        "messages": [HumanMessage(content=body.message)],
        "phase":    initial_phase,
    }
    if body.selected_path_id is not None:
        invoke_input["selected_path_id"] = body.selected_path_id
    if body.selected_rc_eid is not None:
        invoke_input["selected_rc_eid"] = body.selected_rc_eid
    if body.selected_rc_label is not None:
        invoke_input["selected_rc_label"] = body.selected_rc_label



    result = await orchestrator.ainvoke(invoke_input, config=config)

    # 最後一則 AI 訊息
    messages   = result.get("messages") or []
    last_msg   = messages[-1] if messages else AIMessage(content="")
    reply_text = getattr(last_msg, "content", str(last_msg))

    # socrates_phase
    socrates_state = result.get("socrates_state")
    socrates_phase = (
        socrates_state.get("phase")
        if socrates_state and isinstance(socrates_state, dict)
        else None
    )




    return ChatResponse(
        reply          = reply_text,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        skill_used     = result.get("skill_used"),
    )
