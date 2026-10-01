# -*- coding: utf-8 -*-
"""api/routes/chat.py — POST /chat endpoint。

【更新】登入綁定：
  - session_id 本身仍由前端產生，但 LangGraph 的 thread_id 改為
    "{uid}:{session_id}"，確保不同使用者即使湊巧選到同一個
    session_id 字串，也不會共用同一份對話記憶體狀態。
  - uid 一律來自 Authorization header 的 Firebase ID token，
    不再接受 request body 帶 uid。
  - 每輪對話結束後，把可安全序列化的狀態子集備份到 Firestore
    （core/session_store.py），下次遇到 MemorySaver 對該 thread
    已無記錄（冷啟動／重啟）時，從備份還原、接續對話。

【新增 — HTML Widget】
  ChatResponse 多一個 pending_widget 欄位，透傳 MainAgent.chat() 回傳的
  "pending_widget"（見 core/main_agent.py 檔頭說明）。

【新增 — 會員額度 / 用量計量】
  - Depends(require_quota)：用量已達上限時直接回 429（見 core/quota.py）。
    uid 改由 QuotaContext 取得（底層仍是同一個 verify_id_token 驗證）。
  - walk_enabled 不再信任前端：依會員等級覆蓋。訪客一律 False。
    walk_max_depth 由等級決定，寫進 state 讓 walk_seed 使用（單次慢想成本上限）。
  - usage_scope 包住 ainvoke：整輪所有 LLM 呼叫（socratic / main_agent 及其 tool
    loop）都會被計入，離開 scope 時一次寫入 Firestore。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from core.quota import QuotaContext, require_quota
from core.session_store import (
    load_session_snapshot,
    mark_briefing_adopted_in_display,
    save_session_snapshot,
)
from core.usage import usage_scope

router = APIRouter()


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


class PendingWidget(BaseModel):
    widget_id:        str
    title:            str
    url:              str
    created_at:       int
    expires_in_hours: int = 1


class ChatResponse(BaseModel):
    reply:          str
    attachments:    List[Attachment] = []
    phase:          str
    socrates_phase: Optional[str] = None
    skill_used:     Optional[str] = None
    pending_widget: Optional[PendingWidget] = None


# ── Route ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    body:    ChatRequest,
    ctx:     QuotaContext = Depends(require_quota),
) -> ChatResponse:
    uid           = ctx.uid
    orchestrator  = request.app.state.orchestrator
    thread_id     = f"{uid}:{body.session_id}"
    config        = {"configurable": {"thread_id": thread_id}}
    initial_phase = body.phase or "socratic"

    invoke_input: dict = {
        "messages":       [HumanMessage(content=body.message)],
        "phase":          initial_phase,
        "uid":            uid,
        "session_id":     body.session_id,
        # 前端只是「請求」開啟慢想，實際能不能開由會員等級決定。
        "walk_enabled":   bool(body.walk_enabled and ctx.limits.walk_allowed),
        "walk_max_depth": ctx.limits.walk_max_depth,
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

            # walk 的執行狀態綁定 LangGraph 內部 interrupt() 的 checkpoint task，
            # 不是我們能序列化還原的資料——冷啟動代表那個暫停點已經永久遺失，
            # 唯一誠實的作法是降級回 socratic（同 main_state 對含 tool_calls
            # 訊息的降級哲學，見 core/session_store.py）。
            if restored["phase"] in ("walk_pending",):
                invoke_input["phase"] = "socratic"

            if body.selected_rc_eid is None and restored["selected_rc_eid"]:
                invoke_input["selected_rc_eid"]   = restored["selected_rc_eid"]
                invoke_input["selected_rc_label"] = restored["selected_rc_label"]

    async with usage_scope(uid, body.session_id, plan=ctx.membership.plan):
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

    pending_widget_raw = result.get("pending_widget")
    pending_widget = PendingWidget(**pending_widget_raw) if pending_widget_raw else None

    now_ms = int(time.time() * 1000)
    ai_turn_message: Dict[str, Any] = {
        "role": "ai", "content": reply_text,
        "attachments": [a.model_dump() for a in attachments], "created_at": now_ms,
    }
    if pending_widget is not None:
        ai_turn_message["widget"] = pending_widget.model_dump()

    turn_messages = [
        {"role": "user", "content": body.message, "attachments": [], "created_at": now_ms},
        ai_turn_message,
    ]

    await save_session_snapshot(uid, body.session_id, result, turn_messages=turn_messages)

    if body.selected_rc_eid:
        await mark_briefing_adopted_in_display(uid, body.session_id, body.selected_rc_eid)

    return ChatResponse(
        reply          = reply_text,
        attachments    = attachments,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        skill_used     = result.get("skill_used"),
        pending_widget = pending_widget,
    )
