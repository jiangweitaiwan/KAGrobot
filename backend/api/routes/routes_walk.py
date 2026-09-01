# -*- coding: utf-8 -*-
"""api/routes/routes_walk.py — Random Walk（慢想）SSE endpoints（重構版）"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from core.firebase_auth import require_auth
from core.session_store import save_session_snapshot
from core.walk_stream import format_sse, stream_walk_via_orchestrator

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class WalkAnswerRequest(BaseModel):
    path_id: int
    answer:  str


class WalkStatusResponse(BaseModel):
    phase:                 str
    has_pending_question:  bool
    pending_question:      Optional[Dict[str, Any]] = None


def _walk_config(uid: str, session_id: str) -> dict:
    return {"configurable": {"thread_id": f"{uid}:{session_id}"}}


async def _persisted_sse(
    orchestrator: Any,
    config:       dict,
    invoke_input: Any,
    uid:          str,
    session_id:   str,
    extra_turn_messages: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[str, None]:
    """包住 stream_walk_via_orchestrator：把 user_question / walk_complete /
    walk_aborted 的文字內容累積起來，stream 自然結束後（不論是停在
    interrupt 還是真正跑到 walk_finish）補寫一次 Firestore，確保：
      1. thinking_process（walk_finish 寫入的探索摘要）不會只活在
         MemorySaver 裡，冷啟動後仍能被 /chat 讀到。
      2. display_messages 收錄這段對話裡使用者看到的問題/完成訊息，
         「歷史對話」列表能正確顯示這段慢想過程的文字內容
         （不含可執行狀態，見 session_store.py 的取捨說明）。
    """
    import json as _json

    collected_ai_text: List[str] = []

    async for sse_line in stream_walk_via_orchestrator(orchestrator, config, invoke_input):
        yield sse_line
        try:
            payload = _json.loads(sse_line[len("data: "):].strip())
        except Exception:
            continue

        if payload.get("type") == "user_question":
            collected_ai_text.append(f"🤔 {payload.get('question', '')}")
        elif payload.get("type") in ("walk_complete", "walk_aborted"):
            text = payload.get("summary") or payload.get("message") or ""
            if text:
                collected_ai_text.append(text)

    snapshot = await orchestrator.aget_state(config)
    values   = snapshot.values or {}

    now_ms = int(time.time() * 1000)
    turn_messages = list(extra_turn_messages or [])
    if collected_ai_text:
        turn_messages.append({
            "role":        "ai",
            "content":     "\n".join(collected_ai_text),
            "attachments": [],
            "created_at":  now_ms,
        })

    await save_session_snapshot(
        uid, session_id, values,
        turn_messages=turn_messages or None,
    )


def _sse_response(gen: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=_SSE_HEADERS)


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/walk/status/{session_id}", response_model=WalkStatusResponse)
async def walk_status(
    session_id: str,
    request:    Request,
    uid:        str = Depends(require_auth),
) -> WalkStatusResponse:
    """輕量狀態查詢，供前端在頁面重新整理後判斷該怎麼恢復：

      - 後端仍記得這個 thread（沒有冷啟動）且正卡在 interrupt() →
        has_pending_question=True，前端直接把問題渲染出來、等待
        使用者回答（走 /walk/answer），不要送新的 SSE input 進去
        打斷正在暫停的 interrupt。
      - phase 仍是 walk_pending 但沒有 pending question → 代表正在
        frontier_fetch/decide 之間，沒有活躍的 interrupt，前端可以
        安全地重新開一段 /walk/stream 繼續消費事件。
      - phase 不是 walk_pending（例如冷啟動後被 /chat 的降級邏輯打回
        "socratic"，或這個 thread 根本沒跑過） → 前端應該放棄恢復，
        把本機 phase 校正回 socratic。
    """
    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, session_id)
    snapshot = await orchestrator.aget_state(config)
    values = snapshot.values or {}
    phase  = values.get("phase", "socratic")

    pending = None
    if snapshot.interrupts:
        raw = getattr(snapshot.interrupts[0], "value", None) or {}
        pending = {"path_id": raw.get("path_id"), "question": raw.get("question", "")}

    return WalkStatusResponse(
        phase=phase,
        has_pending_question=pending is not None,
        pending_question=pending,
    )


@router.get("/walk/stream/{session_id}")
async def stream_walk(
    session_id: str,
    request:    Request,
    uid:        str = Depends(require_auth),
) -> StreamingResponse:
    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, session_id)

    snapshot = await orchestrator.aget_state(config)
    phase = (snapshot.values or {}).get("phase")
    if phase != "walk_pending":
        raise HTTPException(
            status_code=409,
            detail="此對話目前不在慢想（random walk）階段，無法開啟串流",
        )

    return _sse_response(
        _persisted_sse(orchestrator, config, {"messages": []}, uid, session_id)
    )


@router.post("/walk/answer/{session_id}")
async def answer_walk(
    session_id: str,
    body:       WalkAnswerRequest,
    request:    Request,
    uid:        str = Depends(require_auth),
) -> StreamingResponse:
    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, session_id)

    snapshot = await orchestrator.aget_state(config)
    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="此對話目前並未暫停等待回答")

    now_ms = int(time.time() * 1000)
    user_turn = [{
        "role": "user", "content": body.answer,
        "attachments": [], "created_at": now_ms,
    }]

    return _sse_response(
        _persisted_sse(
            orchestrator, config, Command(resume=body.answer),
            uid, session_id, extra_turn_messages=user_turn,
        )
    )