# -*- coding: utf-8 -*-
"""api/routes/routes_walk.py — Random Walk Mode 的 4 個 endpoint（改版）。

【變更說明 — 改用 orchestrator.astream() 驅動】
不再有獨立的 WalkTurnRunner.run_turn() 手動迴圈 + app.state.walk_sessions
這個 process 內記憶體 dict（那條路完整定義見 core/walk_bot.py 檔頭的
「併入 orchestrator 後的狀態」說明，該 class 現已是 legacy，只是暫時沒
刪）。walk 的狀態現在活在 orchestrator 自己的 MemorySaver checkpointer
裡，跟 dialogue mode 共用同一套 thread_id 慣例（"{uid}:{walk_id}"）：

  - POST /walk/start         ：只做語意搜尋起點 + 準備初始 state，
                                不呼叫 orchestrator。
  - GET  /walk/stream/{id}    ：第一次真正呼叫 orchestrator.astream()，
                                把 /walk/start 準備好的初始 state 當作
                                input 餵進去。
  - POST /walk/answer/{id}    ：呼叫 orchestrator.astream(Command(
                                resume=answer))，直接對應 LangGraph 原生
                                的 interrupt/resume 機制——不再需要手動
                                把答案塞進 context_summary 再重新開一段
                                stream，這件事現在由
                                orchestrator/walk_nodes.py::
                                WalkOrchestratorNodes.walk_ask 自己做。
  - GET  /walk/ranking/{id}    ：邏輯不變（仍是 CandidateRetriever 語意
                                排名前 N 名 + BFSCoordinator 批次展開
                                foundation，套 graph_service/
                                walk_ranking.py 的公式），只是判斷
                                「walk 是否已結束」的方式，從
                                WalkSession.all_done() 改讀 orchestrator
                                目前記錄的 phase 是否為 "walk_complete"。

app.state.walk_pending_starts：
  一個很小的 process 內記憶體 dict（"{uid}:{walk_id}" → 初始 invoke_input
  dict），只用來銜接「/walk/start 準備好初始狀態」與「/walk/stream 第一
  次真正呼叫 orchestrator」這兩個 endpoint 之間的空隙。/walk/stream 一旦
  消費過就會把它 pop 掉；之後 walk 的完整生命週期就交給 orchestrator 的
  checkpointer 負責。與舊版 app.state.walk_sessions 的差異：舊版整個
  walk 的狀態全程都活在那個 dict 裡（process 重啟即整個遺失）；這裡的
  暫存只覆蓋「/walk/start 呼叫過、但 /walk/stream 還沒呼叫過」這一個
  很窄的時間窗，重啟風險的曝險範圍因此大幅縮小。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from core.firebase_auth import require_auth
from core.walk_stream import stream_walk_via_orchestrator
from graph_service.walk_ranking import compute_walk_ranking
from graph_service.walk_state import WalkSession

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ── Schema（與前一版相同，前端不需要改動）─────────────────────────────

class StartWalkRequest(BaseModel):
    query:     str
    k:         int = 3
    max_depth: int = 8


class StartWalkResponse(BaseModel):
    walk_id: str
    start_concepts: List[Dict[str, str]]


class WalkAnswerRequest(BaseModel):
    path_id: int
    answer:  str


class WalkRankedRCResponse(BaseModel):
    eid:              str
    label:            str
    totalScore:       float
    semantic:         float
    interScore:       float
    convergenceCount: int
    approx:           bool


class WalkRankingResponse(BaseModel):
    walk_id: str
    ranking: List[WalkRankedRCResponse]


# ── Helpers ───────────────────────────────────────────────────────────

def _walk_key(uid: str, walk_id: str) -> str:
    return f"{uid}:{walk_id}"


def _walk_config(uid: str, walk_id: str) -> dict:
    return {"configurable": {"thread_id": _walk_key(uid, walk_id)}}


def _sse_response(orchestrator: Any, config: dict, invoke_input: Any) -> StreamingResponse:
    return StreamingResponse(
        stream_walk_via_orchestrator(orchestrator, config, invoke_input),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── Routes ────────────────────────────────────────────────────────────

@router.post("/walk/start", response_model=StartWalkResponse)
async def start_walk(
    body:    StartWalkRequest,
    request: Request,
    uid:     str = Depends(require_auth),
) -> StartWalkResponse:
    pool = request.app.state.start_concept_pool

    start_refs = await asyncio.to_thread(pool.retrieve, body.query, body.k)
    if not start_refs:
        raise HTTPException(
            status_code=422,
            detail="找不到合適的起點概念，請換個描述再試一次",
        )

    import uuid
    walk_id = uuid.uuid4().hex[:12]
    key     = _walk_key(uid, walk_id)

    session = WalkSession.start(
        session_id=key,
        start_concepts=start_refs,
        max_depth=body.max_depth,
    )

    # mode="random_walk" 讓 entry_router 分流進 walk 迴圈；phase 給一個
    # 不與 "path_confirmed"/"main_agent" 衝突的值即可，純粹方便除錯觀察，
    # entry_router 本身不依賴這個值的具體內容（只在乎它不是那兩個保留值）。
    request.app.state.walk_pending_starts[key] = {
        "mode":         "random_walk",
        "phase":        "random_walk",
        "walk_session": session.to_dict(),
        "walk_pointer": -1,
        "uid":          uid,
    }

    return StartWalkResponse(
        walk_id=walk_id,
        start_concepts=[{"eid": c.eid, "label": c.label} for c in start_refs],
    )


@router.get("/walk/stream/{walk_id}")
async def stream_walk(
    walk_id: str,
    request: Request,
    uid:     str = Depends(require_auth),
) -> StreamingResponse:
    key = _walk_key(uid, walk_id)
    initial_input = request.app.state.walk_pending_starts.pop(key, None)
    if initial_input is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "找不到此 walk_id 的初始狀態——可能已經開始過串流"
                "（此 endpoint 只能對同一個 walk_id 呼叫一次，恢復對話"
                "請改用 /walk/answer）、屬於其他使用者，或 process 重啟"
                "後遺失了尚未開始的 walk"
            ),
        )

    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, walk_id)
    return _sse_response(orchestrator, config, initial_input)


@router.post("/walk/answer/{walk_id}")
async def answer_walk(
    walk_id: str,
    body:    WalkAnswerRequest,
    request: Request,
    uid:     str = Depends(require_auth),
) -> StreamingResponse:
    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, walk_id)

    snapshot = await orchestrator.aget_state(config)
    if not snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail=(
                "此 walk 目前並未暫停等待回答（可能尚未開始、已經結束，"
                "或 process 重啟後遺失了暫停點）"
            ),
        )

    # body.path_id 目前只用來讓前端表明「回答的是哪個 path 的問題」，
    # 與 snapshot.interrupts 裡實際暫停的 path 理論上一致（walk 迴圈序列
    # 化執行、同一時間只會有一條 path 卡在 interrupt，見與使用者關於
    # 「不用 Send 做 K-path 平行」的討論）。這裡不做額外交叉驗證，
    # 信任前端只會對目前真的在等待的那個問題送出回答。
    return _sse_response(orchestrator, config, Command(resume=body.answer))


@router.get("/walk/ranking/{walk_id}", response_model=WalkRankingResponse)
async def walk_ranking(
    walk_id: str,
    request: Request,
    uid:     str = Depends(require_auth),
) -> WalkRankingResponse:
    orchestrator = request.app.state.orchestrator
    config = _walk_config(uid, walk_id)

    snapshot = await orchestrator.aget_state(config)
    values   = snapshot.values or {}

    if values.get("phase") != "walk_complete":
        raise HTTPException(status_code=409, detail="walk 尚未結束，無法計算排名")

    session_dict = values.get("walk_session")
    if session_dict is None:
        raise HTTPException(status_code=404, detail="找不到此 walk_id 的狀態")

    session = WalkSession.from_dict(session_dict)

    rc_foundation_map, rc_semantic_scores = await _build_rc_pool_data(request, session)
    ranked = compute_walk_ranking(session, rc_foundation_map, rc_semantic_scores)

    return WalkRankingResponse(
        walk_id=walk_id,
        ranking=[
            WalkRankedRCResponse(
                eid=r.eid, label=r.label, totalScore=r.total_score,
                semantic=r.semantic, interScore=r.inter_score,
                convergenceCount=r.convergence_count, approx=r.approx,
            )
            for r in ranked
        ],
    )


# ── Private：組出語意 pool 的 foundation/semantic 資料（邏輯不變）───────

async def _build_rc_pool_data(
    request: Request,
    session: WalkSession,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """比照 api/routes/routes_brain.py 的 rc_semantic_scores /
    foundation_recommendations 兩個 endpoint 的資料組裝方式：
    CandidateRetriever 語意排名前 N 名 + BFSCoordinator 批次展開 foundation。
    """
    candidate_retriever = request.app.state.candidate_retriever
    coordinator         = request.app.state.coordinator
    settings            = request.app.state.settings
    pool_size           = getattr(settings, "rc_ranking_pool_size", 15)

    query_text = session.context_summary.strip() or " ".join(
        p.current_label for p in session.paths if p.current_label
    )
    if not query_text.strip():
        return {}, {}

    query_emb  = await asyncio.to_thread(candidate_retriever.embed, query_text)
    scored_rcs = await asyncio.to_thread(candidate_retriever.score_all, query_emb)
    top_rcs    = scored_rcs[:pool_size]

    rc_semantic_scores = {s["element_id"]: s["score"] for s in top_rcs}

    expand_results = await asyncio.gather(*[
        coordinator.get_expand_cached_async(s["element_id"]) for s in top_rcs
    ])
    rc_foundation_map = {
        s["element_id"]: {
            "label": s["node_name"],
            "foundation_eids": [
                f["element_id"] for f in r.get("final_foundations", [])
            ],
        }
        for s, r in zip(top_rcs, expand_results)
    }

    return rc_foundation_map, rc_semantic_scores
