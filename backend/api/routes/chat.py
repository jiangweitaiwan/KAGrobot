# -*- coding: utf-8 -*-
"""api/routes/chat.py — POST /chat endpoint。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from graph_service.projector import GraphProjector

router = APIRouter()


# 圖片副檔名判斷同前端渲染邏輯保持一致（見 skill_code.py 的 _IMAGE_SUFFIXES）
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


# ── Schema ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:        str
    message:           str
    phase:             Optional[str] = None
    selected_path_id:  Optional[int] = None
    selected_rc_eid:   Optional[str] = None
    selected_rc_label: Optional[str] = None


class Attachment(BaseModel):
    ref:              str
    kind:             Literal["image", "file"]
    url:              str
    filename:         str
    created_at:       int
    expires_in_hours: int = 24


class ChatResponse(BaseModel):
    reply:          str
    attachments:    List[Attachment] = []
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

    # ── attachments：MainAgent 透過 get_sandbox_file 產生的結構化附件
    #    （ref/kind/url/filename/created_at/expires_in_hours），這裡只是
    #    原樣轉成回應 schema，不做任何文字解析——圖片/檔案是否顯示完全
    #    不依賴 reply_text 裡有沒有正確寫入 [[attachment:N]] 佔位符。
    attachments = [
        Attachment(**a) for a in (result.get("attachments") or [])
    ]

    return ChatResponse(
        reply          = reply_text,
        attachments    = attachments,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        skill_used     = result.get("skill_used"),
    )