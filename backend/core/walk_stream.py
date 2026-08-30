# -*- coding: utf-8 -*-
"""core/walk_stream.py — Random Walk Mode 的 SSE 事件轉譯層。

職責：
  - to_sse_event()：把 WalkTurnResult 轉成前端可消費的 event dict
    （schema 對應設計討論階段定案的草案）。
  - format_sse()：組成 SSE wire format（"data: {json}\\n\\n"）。
  - stream_walk_turns()：驅動 WalkTurnRunner 逐輪跑，逐輪 yield SSE
    字串，直到遇到 "question"（暫停，等使用者回答）或 "walk_complete"
    （終止）才 return。

暫停/resume 設計（對應討論定案「中斷後 resume，不重啟整個 stream」）：
  - 一次 HTTP SSE 連線 = 一段「連續自動推進」的區間，在 question 或
    walk_complete 處自然結束（連線關閉是正常結束，不是異常）。
  - 使用者回答問題後，呼叫端（api/routes 層）應該：
      1. 呼叫 record_user_answer() 把回答摺進 session.context_summary
      2. 對同一個 session/scheduler 重新呼叫 stream_walk_turns()（開一個
         新的 SSE 連線）——因為 scheduler.next_path() 是 idempotent 的
         查詢（不是消耗性的 pop），同一條仍是 ACTIVE 的 path 會被再次選中，
         等於從暫停處自然接續，不需要任何額外的「resume token」機制。
  - LLM 提出的問題本身也會在暫停當下自動摺進 context_summary（見
    _record_question），確保下一次 LLM 呼叫時，「自己問過什麼、使用者
    答了什麼」都在累積脈絡裡，不需要呼叫端手動組裝對話歷史。

刻意不含的部分（留給 api/routes 層接線時處理）：
  - WalkSession 跨 request 的持久化／查找（比照 core/session_store.py
    的模式，用 walk_id 或 f"{uid}:{session_id}" 當 key）。
  - FastAPI StreamingResponse 的實際掛載（見檔尾的 route 骨架示範）。
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from core.walk_bot import WalkTurnResult, WalkTurnRunner
from graph_service.walk_state import WalkScheduler, WalkSession

_DEFAULT_MAX_TURNS = 200   # 安全上限，避免邏輯錯誤造成無限串流


# ══════════════════════════════════════════════════════════════
# Event 轉譯
# ══════════════════════════════════════════════════════════════

def to_sse_event(result: WalkTurnResult) -> dict:
    """把 WalkTurnResult 轉成前端可直接消費的 event dict。

    kind → type 對應：
      frontier_ready  → frontier_ready   （候選節點虛線預覽，非必要可忽略）
      path_terminated → path_dead_end
      advanced        → frontier_expanded（實際走了一步，GlobalBrain 據此點亮節點）
      question        → user_question    （暫停點）
      walk_complete   → walk_complete    （終點，附上 ranking_hint）
    """
    if result.kind == "frontier_ready":
        return {
            "type":      "frontier_ready",
            "path_id":   result.path_id,
            "candidates": result.frontier or [],
        }

    if result.kind == "path_terminated":
        return {
            "type":    "path_dead_end",
            "path_id": result.path_id,
            "reason":  result.dead_end_reason,
        }

    if result.kind == "advanced":
        return {
            "type":    "frontier_expanded",
            "path_id": result.path_id,
            "from_eid": result.from_eid,
            "to_eid":   result.target_eid,
            "to_label": result.target_label,
            "reason":   result.reason,
        }

    if result.kind == "question":
        return {
            "type":    "user_question",
            "path_id": result.path_id,
            "question": result.question,
        }

    if result.kind == "walk_complete":
        return {
            "type":    "walk_complete",
            "ranking": result.ranking_hint or {},
        }

    raise ValueError(f"unknown WalkTurnResult.kind: {result.kind}")


def format_sse(event: dict) -> str:
    """組成 SSE wire format 字串。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ══════════════════════════════════════════════════════════════
# context_summary 記錄 helper
# ══════════════════════════════════════════════════════════════

def _record_question(session: WalkSession, path_id: Optional[int], question: str) -> None:
    """LLM 提問當下自動摺進 context_summary，讓下一次呼叫（resume 後）
    的 LLM 能看到「自己問過什麼」，不需要呼叫端手動組裝對話歷史。
    """
    if not question:
        return
    session.record_decision(f"[path {path_id} 提問]：{question}")


def record_user_answer(session: WalkSession, path_id: Optional[int], answer: str) -> None:
    """使用者回答問題後，呼叫端（api/routes 層）應呼叫此函式，
    把回答摺進 session.context_summary，再重新呼叫 stream_walk_turns()。

    與 _record_question 是一組：一次「提問 → 回答」在 context_summary 裡
    會呈現成連續兩句話，讓 LLM 在下一輪決策時看到完整的問答脈絡。
    """
    if not answer:
        return
    session.record_decision(f"[path {path_id} 使用者回覆]：{answer}")


# ══════════════════════════════════════════════════════════════
# Streaming driver
# ══════════════════════════════════════════════════════════════

async def stream_walk_turns(
    session:   WalkSession,
    scheduler: WalkScheduler,
    runner:    WalkTurnRunner,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> AsyncGenerator[str, None]:
    """驅動 runner 逐輪跑，逐輪 yield 一個 SSE 字串。

    在 "question" 或 "walk_complete" 之後 return（連線自然結束）。
    runner.run_turn() 本身是同步函式（內部會打 Neo4j / LLM API），
    用 asyncio.to_thread 包起來，避免阻塞 event loop——比照
    orchestrator/nodes.py 對同步的 socrates_bot / main_agent 的處理方式。
    """
    for _ in range(max_turns):
        result = await asyncio.to_thread(runner.run_turn, session, scheduler)
        event  = to_sse_event(result)
        yield format_sse(event)

        if result.kind == "question":
            _record_question(session, result.path_id, result.question or "")
            return

        if result.kind == "walk_complete":
            return

    # 超過安全上限：明確 emit 一個錯誤事件，而不是讓連線無聲斷掉，
    # 方便前端與後端 log 都能定位到「walk 邏輯疑似卡在無限迴圈」。
    yield format_sse({
        "type":    "walk_error",
        "message": f"超過 max_turns={max_turns}，強制中止（可能有邏輯錯誤造成無限迴圈）",
    })


# ══════════════════════════════════════════════════════════════
# Route 骨架示範（非完整可執行程式碼，接線時參考）
# ══════════════════════════════════════════════════════════════
#
# 實際掛進 FastAPI 時，架構大致如下（比照 api/routes/routes_brain.py
# 的 dependency 注入慣例）：
#
#   from fastapi import APIRouter, Request, Depends
#   from fastapi.responses import StreamingResponse
#   from core.firebase_auth import require_auth
#
#   router = APIRouter()
#
#   # WalkSession 的跨 request 存放：比照 core/session_store.py 的模式，
#   # 正式版應該是 Firestore 或至少 app.state 上的一個 dict（重啟即遺失，
#   # 但 random walk 本來就設計成不可中途切換 mode，遺失後重開一次不算
#   # 太大的體驗損失）。這裡示意用 app.state.walk_sessions: Dict[str, WalkSession]。
#
#   @router.post("/walk/start")
#   async def start_walk(body: StartWalkRequest, request: Request, uid: str = Depends(require_auth)):
#       start_concepts = await find_start_concepts(body.query, ...)  # 見下方 note
#       session = WalkSession.start(session_id=f"{uid}:{body.walk_id}", start_concepts=start_concepts)
#       request.app.state.walk_sessions[body.walk_id] = (session, WalkScheduler(session))
#       return {"walk_id": body.walk_id}
#
#   @router.get("/walk/stream/{walk_id}")
#   async def stream_walk(walk_id: str, request: Request, uid: str = Depends(require_auth)):
#       session, scheduler = request.app.state.walk_sessions[walk_id]
#       runner = request.app.state.walk_runner  # 單例，見下方 note
#       return StreamingResponse(
#           stream_walk_turns(session, scheduler, runner),
#           media_type="text/event-stream",
#       )
#
#   @router.post("/walk/answer/{walk_id}")
#   async def answer_walk(walk_id: str, body: WalkAnswerRequest, request: Request, uid: str = Depends(require_auth)):
#       session, scheduler = request.app.state.walk_sessions[walk_id]
#       record_user_answer(session, body.path_id, body.answer)
#       return StreamingResponse(
#           stream_walk_turns(session, scheduler, request.app.state.walk_runner),
#           media_type="text/event-stream",
#       )
#
# Note — walk_runner 單例的組裝（比照 app.py lifespan 組裝 socrates_bot /
# main_agent 的方式）：
#
#   frontier_fetcher = make_frontier_fetcher(adjacency_cache, driver, encoder)
#   llm = create_chat_google_generative_ai(model=s.socrates_model, temperature=0.2)
#
#   def llm_caller(system_prompt: str, tool_schema) -> AIMessage:
#       return call_with_retry(
#           lambda: llm.bind_tools([tool_schema]).invoke([SystemMessage(content=system_prompt)])
#       )
#
#   walk_runner = WalkTurnRunner(frontier_fetcher=frontier_fetcher, llm_caller=llm_caller)
#
# Note — find_start_concepts()（/walk/start 用）：
#   對應設計討論第 1 點「起點檢索池只掃描曾是某 RC foundation 的 concept」，
#   這需要一個全圖批次 job 先建好這個候選池的 embedding index（結構上
#   類似 CandidateRetriever，但語料是一般 concept 的 name+mechanism，
#   而非 RC 的 name+function），不是這次要做的範圍，留待下一步。
