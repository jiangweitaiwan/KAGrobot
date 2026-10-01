# -*- coding: utf-8 -*-
"""api/routes/routes_brain.py — PersonalBrain 相關 endpoints（更新版）

與原版的差異：
  - /graph/expand/{eid}   → 改用 coordinator.get_expand_cached()
  - /graph/rc-paths/{eid} → 改用 coordinator.get_rcpath_cached()
  - 移除對 app.state.expand_cache / app.state.rcpath_cache 的直接存取
    （結果層 cache 統一由 BFSCoordinator 管理）
  - 新增 GET /admin/cache/stats → 回傳各層 cache 統計資訊

其餘邏輯（anchor label 查詢、Schema、semantic scores）維持不變。

【會員板塊】
  - /graph/expand、/graph/expand/batch、/graph/rc-paths、/graph/bertopic-layout
    補上 require_auth（原本沒有驗證，前端 apiFetch 雖然一律帶 token，
    但後端沒有檢查，等於任何人都能打）。
  - /admin/cache/stats 改為 require_admin。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import json
import tempfile
from pathlib import Path

from core.semantic_context import SemanticContext
from graph_service.semantic_scorer import combine_score, type_overlap
from core.firebase_auth import require_auth, require_admin, require_auth_claims
from core.quota import compute_quota, quota_error_detail
from core.session_store import (
    append_briefing_display_message,
    load_rc_briefing_for_context,
    load_session_snapshot,
    save_rc_briefing,
)
from core.rc_briefing import RCBriefing, context_hash, normalize_briefing_payload
from core.usage import usage_scope
router = APIRouter()


# ── Schema ────────────────────────────────────────────────────────────

class FoundationTreeNode(BaseModel):
    element_id: str
    label:      str
    node_type:  str = "support"   # ← 新增："support" | "critique" | "other"

class FoundationTreeEdge(BaseModel):
    source:    str
    target:    str
    edge_type: str = "support"    # ← 新增

class FoundationItem(BaseModel):  # ← 新增，取代 foundations 欄位原本沿用 FoundationTreeNode
    element_id:      str
    label:           str
    depth:           int = 0
    dominant_type:   str = "other"
    support_weight:  float = 0.0
    critique_weight: float = 0.0

class ExpandResponse(BaseModel):
    anchor_eid:   str
    anchor_label: str
    foundations:  List[FoundationItem]      # ← 型別換掉
    nodes:        List[FoundationTreeNode]
    edges:        List[FoundationTreeEdge]


class PathNode(BaseModel):
    id:           str
    label:        str
    node_type:    str
    concept_type: Optional[str]
    matched:      bool
    has_skill:    bool = False    

class PathEdge(BaseModel):
    id:          str
    source:      str
    target:      str
    paper_title: str

class RCPath(BaseModel):
    path_id:     int
    node_ids:    List[str]
    edge_ids:    List[str]
    final_label: str

class RCPathsResponse(BaseModel):
    eid:   str
    label: str
    nodes: List[PathNode]
    edges: List[PathEdge]
    paths: List[RCPath]


class RCSemanticScore(BaseModel):
    eid:          str
    label:        str
    score:        float
    embed_score:  float = 0.0
    intent_score: float = 0.0

class SemanticScoresResponse(BaseModel):
    query:  str
    scores: List[RCSemanticScore]


class SemanticContextResponse(BaseModel):
    session_id:       str
    query:            str
    intent_types:     List[str]
    summary:          str
    thinking_process: str = ""

class FoundationRecommendation(BaseModel):
    element_id: str
    label:      str
    score:      float

class FoundationRecommendationsResponse(BaseModel):
    session_id:      str
    recommendations: List[FoundationRecommendation]


class CacheStatsResponse(BaseModel):
    adjacency_cache: Dict[str, Any]
    rcpath_cache:    Dict[str, Any]
    expand_cache:    Dict[str, Any]


class BertopicNode(BaseModel):
    eid:         Optional[str]
    name:        str
    x:           float
    y:           float
    topic_id:    int
    topic_label: str
    color:       str


class BertopicTopic(BaseModel):
    id:    int
    label: str
    color: str
    words: List[str]


class BertopicLayoutResponse(BaseModel):
    nodes:  List[BertopicNode]
    topics: List[BertopicTopic]

class BatchExpandRequest(BaseModel):
    eids: List[str]


class BatchExpandResponse(BaseModel):
    results: Dict[str, ExpandResponse]    

class RCBriefingResponse(BaseModel):
    session_id: str
    eid:        str
    rc_label:   str
    briefing:   RCBriefing
    labels:     Dict[str, str]     # briefing 引用到的 eid → label，前端直接渲染
    cached:     bool = False

# ── Helpers ───────────────────────────────────────────────────────────

def _anchor_label_from_cache(
    layout_cache: Optional[Dict[str, Any]],
    eid: str,
) -> Optional[str]:
    if not layout_cache:
        return None
    nodes: List[Dict[str, Any]] = layout_cache.get("nodes", [])
    for n in nodes:
        if n.get("id") == eid:
            return n.get("name") or None
    return None


def _anchor_label_from_neo4j(driver, eid: str) -> str:
    with driver.session() as session:
        rec = session.run(
            "MATCH (c:Concept) WHERE elementId(c) = $eid RETURN c.name AS name",
            eid=eid,
        ).single()
    return (rec["name"] if rec and rec["name"] else None) or eid


def _attach_skill_flags(nodes_raw: List[Dict[str, Any]], skill_loader) -> List["PathNode"]:
    """把 raw BFS 節點轉成 PathNode，附上 has_skill。

    只對 concept_type 命中 Resulting_Concept 的節點查詢
    （foundation / 中繼概念不會綁定 skill）。skill_loader.skill_exists()
    本身有 cache，這裡即使重複跑也幾乎零成本。
    """
    out: List[PathNode] = []
    for n in nodes_raw:
        concept_type = n.get("concept_type")
        is_rc = "resulting" in (concept_type or "").lower()
        has_skill = skill_loader.skill_exists(n["label"]) if is_rc else False
        out.append(PathNode(
            id=n["id"],
            label=n["label"],
            node_type=n["node_type"],
            concept_type=concept_type,
            matched=n.get("matched", False),
            has_skill=has_skill,
        ))
    return out


def _get_anchor_label(request: Request, eid: str) -> str:
    label = _anchor_label_from_cache(request.app.state.layout_cache, eid)
    if label:
        return label
    return _anchor_label_from_neo4j(request.app.state.driver, eid)

async def _get_semantic_context(request: Request, uid: str, session_id: str) -> SemanticContext:
    orchestrator = request.app.state.orchestrator
    config = {"configurable": {"thread_id": f"{uid}:{session_id}"}}

    try:
        snapshot = orchestrator.get_state(config)
        values = snapshot.values or {}
    except Exception:
        values = {}

    if not values:
        # MemorySaver 對這個 thread 已無記錄（冷啟動／重啟），
        # 降級讀 Firestore 備份，邏輯與 chat.py 的還原路徑一致。
        restored = await load_session_snapshot(uid, session_id)
        if not restored:
            return SemanticContext()
        socrates_state = restored.get("socrates_state") or {}
        return SemanticContext(
            query=socrates_state.get("query") or "",
            intent_types=socrates_state.get("intent_types") or [],
            summary=socrates_state.get("context_summary") or "",
            thinking_process=restored.get("thinking_process") or "",
        )

    socrates_state = values.get("socrates_state") or {}
    return SemanticContext(
        query=socrates_state.get("query") or "",
        intent_types=socrates_state.get("intent_types") or [],
        summary=socrates_state.get("context_summary") or "",
        thinking_process=values.get("thinking_process") or "",
    )


async def _get_walk_visited(request: Request, uid: str, session_id: str) -> List[str]:
    """取出 walk_session 所有 path 走過的 eid（沒有 walk 時回傳空清單）。
 
    邏輯與 _get_semantic_context 對稱：先讀 orchestrator checkpoint，
    沒有再降級讀 Firestore 備份。
    """
    orchestrator = request.app.state.orchestrator
    config = {"configurable": {"thread_id": f"{uid}:{session_id}"}}
    try:
        values = orchestrator.get_state(config).values or {}
    except Exception:
        values = {}
    if not values:
        restored = await load_session_snapshot(uid, session_id)
        values = restored or {}
 
    walk = values.get("walk_session") or {}
    eids: List[str] = []
    for p in walk.get("paths", []):
        eids.extend(n["eid"] for n in p.get("visited", []))
    return list(dict.fromkeys(eids))

# ── Routes ────────────────────────────────────────────────────────────

@router.get("/graph/expand/{eid}", response_model=ExpandResponse)
async def expand_concept(
    eid: str,
    request: Request,
    _uid: str = Depends(require_auth),
) -> ExpandResponse:
    coordinator  = request.app.state.coordinator
    anchor_label = _get_anchor_label(request, eid)

    result = await coordinator.get_expand_cached_async(eid, max_depth=6, max_branches=4)


    foundations = [
        FoundationItem(
            element_id=f["element_id"],
            label=f["label"],
            depth=f.get("depth", 0),
            dominant_type=f.get("dominant_type", "other"),
            support_weight=f.get("support_weight", 0.0),
            critique_weight=f.get("critique_weight", 0.0),
        )
        for f in result.get("final_foundations", [])
    ]
    tree_nodes = [
        FoundationTreeNode(
            element_id=n["element_id"], label=n["label"],
            node_type=n.get("node_type", "support"),
        )
        for n in result.get("foundation_tree", {}).get("nodes", [])
    ]
    tree_edges = [
        FoundationTreeEdge(
            source=e["source"], target=e["target"],
            edge_type=e.get("edge_type", "support"),
        )
        for e in result.get("foundation_tree", {}).get("edges", [])
    ]

    if not any(n.element_id == eid for n in tree_nodes):
        tree_nodes.append(FoundationTreeNode(element_id=eid, label=anchor_label))

    return ExpandResponse(
        anchor_eid=eid,
        anchor_label=anchor_label,
        foundations=foundations,
        nodes=tree_nodes,
        edges=tree_edges,
    )


@router.post("/graph/expand/batch", response_model=BatchExpandResponse)
async def expand_concept_batch(
    body:    BatchExpandRequest,
    request: Request,
    _uid:    str = Depends(require_auth),
) -> BatchExpandResponse:
    """批次版 /graph/expand/{eid}，供前端一次補齊 ranking pool 缺少的 RC。

    每個 eid 仍各自走 coordinator.get_expand_cached_async，
    命中 _expand_cache 時幾乎零延遲；未命中的才會真正觸發 BFS
    （鄰接層 cache 與其他 BFS 操作共享，不重複打 Neo4j）。
    """
    coordinator = request.app.state.coordinator

    # 去重 + 上限保護，避免前端誤傳過大清單
    eids = list(dict.fromkeys(body.eids))[:50]

    async def _one(eid: str) -> tuple[str, ExpandResponse]:
        anchor_label = _get_anchor_label(request, eid)
        result = await coordinator.get_expand_cached_async(
            eid, max_depth=6, max_branches=4
        )

        foundations = [
            FoundationItem(
                element_id=f["element_id"],
                label=f["label"],
                depth=f.get("depth", 0),
                dominant_type=f.get("dominant_type", "other"),
                support_weight=f.get("support_weight", 0.0),
                critique_weight=f.get("critique_weight", 0.0),
            )
            for f in result.get("final_foundations", [])
        ]
        tree_nodes = [
            FoundationTreeNode(
                element_id=n["element_id"], label=n["label"],
                node_type=n.get("node_type", "support"),
            )
            for n in result.get("foundation_tree", {}).get("nodes", [])
        ]
        tree_edges = [
            FoundationTreeEdge(
                source=e["source"], target=e["target"],
                edge_type=e.get("edge_type", "support"),
            )
            for e in result.get("foundation_tree", {}).get("edges", [])
        ]
        if not any(n.element_id == eid for n in tree_nodes):
            tree_nodes.append(FoundationTreeNode(element_id=eid, label=anchor_label))

        return eid, ExpandResponse(
            anchor_eid=eid,
            anchor_label=anchor_label,
            foundations=foundations,
            nodes=tree_nodes,
            edges=tree_edges,
        )

    pairs = await asyncio.gather(*[_one(e) for e in eids])
    return BatchExpandResponse(results=dict(pairs))


@router.get("/graph/rc-paths/{eid}", response_model=RCPathsResponse)
async def rc_paths(
    eid: str,
    request: Request,
    _uid: str = Depends(require_auth),
) -> RCPathsResponse:
    """取得指定 RC 的 downstream paths（DownstreamBFS）。

    結果快取由 BFSCoordinator 統一管理（取代 app.state.rcpath_cache）。
    """
    coordinator = request.app.state.coordinator
    rc_label    = _get_anchor_label(request, eid)

    # ── BFS（結果層 cache 命中則不執行）─────────────────────
    raw = await asyncio.to_thread(
        coordinator.get_rcpath_cached, eid, rc_label, max_depth=6
    )

    skill_loader = request.app.state.skill_loader
    nodes = await asyncio.to_thread(_attach_skill_flags, raw.get("nodes", []), skill_loader)
    
    edges = [
        PathEdge(
            id=e["id"],
            source=e["source"] if isinstance(e["source"], str) else e["source"]["id"],
            target=e["target"] if isinstance(e["target"], str) else e["target"]["id"],
            paper_title=e.get("paper_title", ""),
        )
        for e in raw.get("edges", [])
    ]
    paths = [
        RCPath(
            path_id=p["path_id"],
            node_ids=p["node_ids"],
            edge_ids=p["edge_ids"],
            final_label=p.get("final_label", ""),
        )
        for p in raw.get("paths", [])
    ]

    return RCPathsResponse(
        eid=eid, label=rc_label,
        nodes=nodes, edges=edges, paths=paths,
    )


@router.get("/brain/rc-semantic-scores", response_model=SemanticScoresResponse)
async def rc_semantic_scores(
    request:    Request,
    session_id: Optional[str] = None,
    query:      Optional[str] = None,
    uid:        str = Depends(require_auth),
) -> SemanticScoresResponse:
    """計算所有 Resulting Concept 對目前 SemanticContext 的匹配分數。

    優先使用 session_id 從 orchestrator checkpoint 取得完整 SemanticContext
    （query + summary + intent_types）；未提供 session_id 時退回到直接傳入
    的 query 字串（向後相容舊版前端呼叫方式，此時 intent_score 恆為 0）。
    """
    candidate_retriever = request.app.state.candidate_retriever

    ctx = (
        await _get_semantic_context(request, uid, session_id)
        if session_id
        else SemanticContext(query=query or "")
    )

    embedding_text = ctx.embedding_text
    if not embedding_text.strip():
        return SemanticScoresResponse(query=ctx.query, scores=[])

    query_emb = await asyncio.to_thread(candidate_retriever.embed, embedding_text)
    scored    = await asyncio.to_thread(candidate_retriever.score_all, query_emb)

    results: List[RCSemanticScore] = []
    for s in scored:
        embed_sc  = s["score"]
        intent_sc = type_overlap(s.get("analysis_types", []), ctx.intent_types)
        results.append(RCSemanticScore(
            eid=s["element_id"],
            label=s["node_name"],
            score=combine_score(embed_sc, intent_sc),
            embed_score=embed_sc,
            intent_score=intent_sc,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return SemanticScoresResponse(query=ctx.query, scores=results)

@router.get("/brain/semantic-context", response_model=SemanticContextResponse)
async def semantic_context(
    session_id: str,
    request:    Request,
    uid:        str = Depends(require_auth),
) -> SemanticContextResponse:
    ctx = await _get_semantic_context(request, uid, session_id)
    return SemanticContextResponse(
        session_id=session_id,
        query=ctx.query,
        intent_types=ctx.intent_types,
        summary=ctx.summary,
        thinking_process=ctx.thinking_process,
    )


@router.get("/admin/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(
    request: Request,
    _admin:  dict = Depends(require_admin),
) -> CacheStatsResponse:
    """回傳各層 cache 的統計資訊（命中率、大小等）。

    用於監控與調校 LRU maxsize。僅限管理者。
    """
    coordinator = request.app.state.coordinator
    stats       = coordinator.stats

    return CacheStatsResponse(
        adjacency_cache=stats["adjacency_cache"],
        rcpath_cache=stats["rcpath_cache"],
        expand_cache=stats["expand_cache"],
    )



@router.get("/graph/bertopic-layout", response_model=BertopicLayoutResponse)
async def bertopic_layout(
    request: Request,
    _uid: str = Depends(require_auth),
) -> BertopicLayoutResponse:
    """回傳 BERTopic 2D layout 資料，供 PersonalBrain 前端繪製。

    首次呼叫從 GCS 下載，快取在 app.state.bertopic_layout。
    """
    if request.app.state.bertopic_layout is not None:
        return request.app.state.bertopic_layout

    result = await asyncio.to_thread(_load_bertopic_layout, request)
    request.app.state.bertopic_layout = result
    return result


def _load_bertopic_layout(request: Request) -> BertopicLayoutResponse:
    s           = request.app.state.settings
    bucket_name = (getattr(s, "gcs_bucketforresult", "") or "").strip()
    gcs_object  = "bertopic_layout.json"
    raw: dict | None = None

    if bucket_name:
        try:
            from google.cloud import storage as gcs_storage
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)
            blob   = bucket.blob(gcs_object)
            if blob.exists(client):
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = tmp.name
                blob.download_to_filename(tmp_path)
                with open(tmp_path, encoding="utf-8") as f:
                    raw = json.load(f)
                Path(tmp_path).unlink(missing_ok=True)
                print(f"[BertopicLayout] loaded gs://{bucket_name}/{gcs_object}")
        except Exception as exc:
            print(f"[BertopicLayout] GCS load failed: {exc}")

    if raw is None:
        local_path = Path(__file__).resolve().parent.parent.parent / "bertopic_layout.json"
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                raw = json.load(f)
        else:
            print("[BertopicLayout] no layout found, returning empty")
            return BertopicLayoutResponse(nodes=[], topics=[])

    nodes  = [BertopicNode(**n)  for n in (raw.get("nodes")  or [])]
    topics = [BertopicTopic(**t) for t in (raw.get("topics") or [])]
    print(f"[BertopicLayout] nodes={len(nodes)}, topics={len(topics)}")
    return BertopicLayoutResponse(nodes=nodes, topics=topics)


@router.get("/brain/foundation-recommendations", response_model=FoundationRecommendationsResponse)
async def foundation_recommendations(
    session_id: str,
    request:    Request,
    uid:        str = Depends(require_auth),
) -> FoundationRecommendationsResponse:
    ctx = await _get_semantic_context(request, uid, session_id)
    if not ctx.embedding_text.strip():
        return FoundationRecommendationsResponse(session_id=session_id, recommendations=[])

    candidate_retriever = request.app.state.candidate_retriever
    recommender         = request.app.state.foundation_recommender
    settings            = request.app.state.settings
    pool_size           = getattr(settings, "rc_ranking_pool_size", 15)

    query_emb  = await asyncio.to_thread(candidate_retriever.embed, ctx.embedding_text)
    scored_rcs = await asyncio.to_thread(candidate_retriever.score_all, query_emb)
    top_rc_eids = [s["element_id"] for s in scored_rcs[:pool_size]]

    recs = await recommender.recommend(query_emb, top_rc_eids)

    return FoundationRecommendationsResponse(
        session_id=session_id,
        recommendations=[
            FoundationRecommendation(
                element_id=r["element_id"], label=r["label"], score=r["score"]
            )
            for r in recs
        ],
    )


async def _load_briefing_payload(
    cache: Any,
    key: tuple,
    uid: str,
    session_id: str,
    eid: str,
    ctx_hash: str,
) -> tuple[Optional[dict], bool]:
    """回傳 (payload, from_firestore)。from_firestore 時 caller 應補 display_messages。"""
    hit = cache.get(key)
    if hit:
        return normalize_briefing_payload(hit), False
    fs = await load_rc_briefing_for_context(uid, session_id, eid, ctx_hash)
    if not fs:
        return None, False
    payload = normalize_briefing_payload(fs)
    cache.put(key, payload)
    return payload, True


async def _ensure_briefing_display_row(
    uid: str,
    session_id: str,
    *,
    eid: str,
    rc_label: str,
    ctx_hash: str,
    payload: dict,
) -> None:
    b = payload["briefing"]
    briefing_dict = b.model_dump() if isinstance(b, RCBriefing) else dict(b)
    await append_briefing_display_message(
        uid, session_id,
        eid=eid, rc_label=rc_label, ctx_hash=ctx_hash,
        briefing=briefing_dict, labels=payload.get("labels") or {},
    )


@router.get("/brain/rc-briefing", response_model=RCBriefingResponse)
async def rc_briefing(
    session_id: str,
    eid:        str,
    request:    Request,
    uid:        str = Depends(require_auth),
    claims:     dict = Depends(require_auth_claims),
) -> RCBriefingResponse:
    """預覽用：RC 全局說明。無副作用（不改 selected_rc_eid / brain / sandbox）。"""
    ctx = await _get_semantic_context(request, uid, session_id)
    context_text = ctx.embedding_text
    if not context_text.strip():
        raise HTTPException(status_code=400, detail="尚無足夠的對話脈絡可產生說明。")

    cache        = request.app.state.briefing_cache
    generator    = request.app.state.briefing_generator
    walk_visited = await _get_walk_visited(request, uid, session_id)
    ctx_hash     = context_hash(context_text, walk_visited)
    key          = (uid, session_id, eid, ctx_hash)
    rc_label     = _get_anchor_label(request, eid)

    payload, from_fs = await _load_briefing_payload(cache, key, uid, session_id, eid, ctx_hash)
    if payload:
        if from_fs:
            await _ensure_briefing_display_row(
                uid, session_id, eid=eid, rc_label=rc_label, ctx_hash=ctx_hash, payload=payload,
            )
        return RCBriefingResponse(
            session_id=session_id, eid=eid, rc_label=rc_label, cached=True, **payload,
        )

    async with cache.lock_for(key):
        payload, from_fs = await _load_briefing_payload(cache, key, uid, session_id, eid, ctx_hash)
        if payload:
            if from_fs:
                await _ensure_briefing_display_row(
                    uid, session_id, eid=eid, rc_label=rc_label, ctx_hash=ctx_hash, payload=payload,
                )
            return RCBriefingResponse(
                session_id=session_id, eid=eid, rc_label=rc_label, cached=True, **payload,
            )

        quota_ctx = await compute_quota(claims)
        if quota_ctx.enforced and quota_ctx.exceeded and not quota_ctx.membership.is_admin:
            raise HTTPException(
                status_code=429,
                detail=quota_error_detail(quota_ctx),
            )

        coordinator = request.app.state.coordinator
        result = await coordinator.get_expand_cached_async(eid, max_depth=6, max_branches=4)
        tree   = result.get("foundation_tree", {}) or {}
        nodes  = tree.get("nodes", [])
        edges  = tree.get("edges", [])
        if not any(n["element_id"] == eid for n in nodes):
            nodes = nodes + [{"element_id": eid, "label": rc_label, "node_type": "support"}]

        from core.rc_briefing import prune_tree
        kept_nodes, _, _ = prune_tree(eid, nodes, edges)
        details = await asyncio.to_thread(
            request.app.state.concept_retriever.retrieve_batch,
            [n["element_id"] for n in kept_nodes],
        )

        async with usage_scope(uid, session_id, plan=quota_ctx.membership.plan):
            briefing, labels = await asyncio.to_thread(
                generator.generate,
                rc_eid=eid, rc_label=rc_label, context_text=context_text,
                nodes=nodes, edges=edges, details=details, walk_visited=walk_visited,
            )

        payload = {"briefing": briefing, "labels": labels}
        cache.put(key, payload)
        briefing_dict = briefing.model_dump()
        await save_rc_briefing(
            uid, session_id, eid, ctx_hash, rc_label, briefing_dict, labels,
        )
        await append_briefing_display_message(
            uid, session_id,
            eid=eid, rc_label=rc_label, ctx_hash=ctx_hash,
            briefing=briefing_dict, labels=labels,
        )
        return RCBriefingResponse(
            session_id=session_id, eid=eid, rc_label=rc_label, cached=False, **payload,
        )