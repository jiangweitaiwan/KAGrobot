# -*- coding: utf-8 -*-
"""api/routes/routes_walk.py — Random Walk Mode 的 4 個 endpoint。

POST /walk/start           → 語意搜尋起點（StartConceptPool），建立 WalkSession
GET  /walk/stream/{walk_id}   → SSE 串流，逐輪推進，停在 question 或 walk_complete
POST /walk/answer/{walk_id}   → 提交回答，回應本身是新一段 SSE 串流（resume）
GET  /walk/ranking/{walk_id}  → walk 結束後，套用 walk_ranking.py 公式算最終排名

WalkSession 的存放：
  目前是 app.state.walk_sessions 這個 process 內記憶體 dict（key 為
  f"{uid}:{walk_id}"）。比照 orchestrator 的 MemorySaver（見
  orchestrator/orchestrator_graph.py）——同樣是「重啟即遺失」的取捨。
  dialogue mode 有 core/session_store.py 做 Firestore 備援，random walk
  mode 目前沒有對應機制；由於這個 mode 設計上「不允許中途切換」且通常
  一次性跑完，重啟後遺失、使用者重新開一次 walk，代價比 dialogue mode
  中途遺失小很多，先接受這個限制，之後有需要再比照 session_store.py
    加一層 Firestore 備援。

排名計算的資料來源（/walk/ranking）：
  rc_foundation_map / rc_semantic_scores 的取得方式與既有
  /brain/rc-semantic-scores、/brain/foundation-recommendations
  （見 api/routes/routes_brain.py）完全同構：CandidateRetriever.score_all()
  取語意排名前 N 名，再用 BFSCoordinator.get_expand_cached_async() 批次
  取得各自的 foundation 清單。刻意重用同一套機制、同一份 cache
  （BFSCoordinator 的 _expand_cache 命中率因此也跟 dialogue mode 共用）。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.firebase_auth import require_auth
from core.walk_bot import WalkTurnRunner
from core.walk_stream import record_user_answer, stream_walk_turns
from graph_service.walk_ranking import compute_walk_ranking
from graph_service.walk_state import WalkScheduler, WalkSession

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ── Schema ────────────────────────────────────────────────────────────

class StartWalkRequest(BaseModel):
    query:     str
    k:         int = 3
    max_depth: int = 8


class StartWalkResponse(BaseModel):
    walk_id: str
    # 回傳起點清單，讓前端不需要再等第一個 frontier_ready 事件
    # 才知道「這次 walk 從哪幾個概念開始」。
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


def _get_walk_entry(
    request: Request,
    uid:     str,
    walk_id: str,
) -> Tuple[WalkSession, WalkScheduler]:
    entry = request.app.state.walk_sessions.get(_walk_key(uid, walk_id))
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="找不到此 walk_id，可能已過期（process 重啟）或屬於其他使用者",
        )
    return entry


def _sse_response(
    session:   WalkSession,
    scheduler: WalkScheduler,
    runner:    WalkTurnRunner,
) -> StreamingResponse:
    return StreamingResponse(
        stream_walk_turns(session, scheduler, runner),
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

    walk_id = uuid.uuid4().hex[:12]
    session = WalkSession.start(
        session_id=_walk_key(uid, walk_id),
        start_concepts=start_refs,
        max_depth=body.max_depth,
    )
    scheduler = WalkScheduler(session)

    request.app.state.walk_sessions[_walk_key(uid, walk_id)] = (session, scheduler)

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
    session, scheduler = _get_walk_entry(request, uid, walk_id)
    return _sse_response(session, scheduler, request.app.state.walk_runner)


@router.post("/walk/answer/{walk_id}")
async def answer_walk(
    walk_id: str,
    body:    WalkAnswerRequest,
    request: Request,
    uid:     str = Depends(require_auth),
) -> StreamingResponse:
    session, scheduler = _get_walk_entry(request, uid, walk_id)
    record_user_answer(session, body.path_id, body.answer)
    return _sse_response(session, scheduler, request.app.state.walk_runner)


@router.get("/walk/ranking/{walk_id}", response_model=WalkRankingResponse)
async def walk_ranking(
    walk_id: str,
    request: Request,
    uid:     str = Depends(require_auth),
) -> WalkRankingResponse:
    session, _scheduler = _get_walk_entry(request, uid, walk_id)

    if not session.all_done():
        raise HTTPException(status_code=409, detail="walk 尚未結束，無法計算排名")

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


# ── Private：組出語意 pool 的 foundation/semantic 資料 ─────────────────

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
