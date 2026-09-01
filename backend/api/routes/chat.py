# -*- coding: utf-8 -*-
"""api/routes/chat.py — POST /chat endpoint。

【更新】登入綁定：
  - session_id 本身仍由前端產生，但 LangGraph 的 thread_id 改為
    "{uid}:{session_id}"，確保不同使用者即使湊巧選到同一個
    session_id 字串，也不會共用同一份對話記憶體狀態。
  - uid 一律來自 Authorization header 的 Firebase ID token，
    不再接受 request body 帶 uid（原本 ChatRequest 就沒有這個欄位，
    現在明確靠 Depends(require_auth) 取得，杜絕未來被誤加回去）。
  - 每輪對話結束後，把可安全序列化的狀態子集備份到 Firestore
    （core/session_store.py），下次遇到 MemorySaver 對該 thread
    已無記錄（冷啟動／重啟）時，從備份還原、接續對話，而不是砍掉重練。
"""

from __future__ import annotations

import time  
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from core.firebase_auth import require_auth
from core.session_store import load_session_snapshot, save_session_snapshot
from graph_service.projector import GraphProjector

router = APIRouter()


# 圖片副檔名判斷同前端渲染邏輯保持一致（見 skill_code.py 的 _IMAGE_SUFFIXES）
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


# ── Schema ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:        str
    message:           str
    phase:             Optional[str] = None
    selected_rc_eid:   Optional[str] = None
    selected_rc_label: Optional[str] = None
    walk_enabled:      bool = False   


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
    uid:     str = Depends(require_auth),
) -> ChatResponse:
    orchestrator  = request.app.state.orchestrator
    thread_id     = f"{uid}:{body.session_id}"
    config        = {"configurable": {"thread_id": thread_id}}
    initial_phase = body.phase or "socratic"

    invoke_input: dict = {
        "messages":     [HumanMessage(content=body.message)],
        "phase":        initial_phase,
        "uid":          uid,
        "session_id":   body.session_id,      # ← 新增
        "walk_enabled": body.walk_enabled,    # ← 新增
    }
    if body.selected_rc_eid is not None:
        invoke_input["selected_rc_eid"] = body.selected_rc_eid
    if body.selected_rc_label is not None:
        invoke_input["selected_rc_label"] = body.selected_rc_label

    existing_snapshot = orchestrator.get_state(config)
    if not existing_snapshot.values:
        restored = await load_session_snapshot(uid, body.session_id)
        if restored:
            invoke_input["phase"]            = restored["phase"]
            invoke_input["socrates_state"]   = restored["socrates_state"]
            invoke_input["main_state"]       = restored["main_state"]
            invoke_input["confirmed_rc_eid"] = restored["confirmed_rc_eid"]
            invoke_input["skill_sandbox_id"] = restored["skill_sandbox_id"]
            invoke_input["thinking_process"] = restored["thinking_process"]

            # ── walk 的執行狀態（walk_session/pointer/pending_question）
            #    綁定的是 LangGraph 內部 interrupt() 的 checkpoint task，
            #    不是我們能序列化還原的資料——這裡冷啟動代表那個暫停點
            #    已經永久遺失，唯一誠實的作法是降級回 socratic，讓使用者
            #    知道上次的慢想被中斷了，而不是假裝可以無縫恢復一個
            #    實際上已經不存在的 interrupt（同 main_state 對含
            #    tool_calls 訊息的降級哲學，見 core/session_store.py）。
            if restored["phase"] in ("walk_pending",):
                invoke_input["phase"] = "socratic"

            if body.selected_rc_eid is None and restored["selected_rc_eid"]:
                invoke_input["selected_rc_eid"]   = restored["selected_rc_eid"]
                invoke_input["selected_rc_label"] = restored["selected_rc_label"]
            if body.phase is None:
                invoke_input["phase"] = invoke_input["phase"]  # 已在上面決定，不再被 restored 覆蓋

    result = await orchestrator.ainvoke(invoke_input, config=config)

    messages   = result.get("messages") or []
    last_msg   = messages[-1] if messages else AIMessage(content="")
    reply_text = getattr(last_msg, "content", str(last_msg))

    socrates_state = result.get("socrates_state")
    socrates_phase = (
        socrates_state.get("phase")
        if socrates_state and isinstance(socrates_state, dict)
        else None
    )

    attachments = [Attachment(**a) for a in (result.get("attachments") or [])]

    now_ms = int(time.time() * 1000)
    turn_messages = [
        {"role": "user", "content": body.message, "attachments": [], "created_at": now_ms},
        {"role": "ai", "content": reply_text,
         "attachments": [a.model_dump() for a in attachments], "created_at": now_ms},
    ]

    await save_session_snapshot(uid, body.session_id, result, turn_messages=turn_messages)

    return ChatResponse(
        reply          = reply_text,
        attachments    = attachments,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        skill_used     = result.get("skill_used"),
    )
