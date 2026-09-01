# -*- coding: utf-8 -*-
"""api/routes/routes_brain.py — PersonalBrain 相關 endpoints（更新版）

與原版的差異：
  - /graph/expand/{eid}   → 改用 coordinator.get_expand_cached()
  - /graph/rc-paths/{eid} → 改用 coordinator.get_rcpath_cached()
  - 移除對 app.state.expand_cache / app.state.rcpath_cache 的直接存取
    （結果層 cache 統一由 BFSCoordinator 管理）
  - 新增 GET /admin/cache/stats → 回傳各層 cache 統計資訊

其餘邏輯（anchor label 查詢、Schema、semantic scores）維持不變。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Request,Depends
from pydantic import BaseModel
import json
import tempfile
from pathlib import Path

from core.semantic_context import SemanticContext
from graph_service.semantic_scorer import combine_score, type_overlap
from core.firebase_auth import require_auth
router = APIRouter()


# ── Schema ────────────────────────────────────────────────────────────

class FoundationTreeNode(BaseModel):
    element_id: str
    label:      str

class FoundationTreeEdge(BaseModel):
    source: str
    target: str

class ExpandResponse(BaseModel):
    anchor_eid:   str
    anchor_label: str
    foundations:  List[FoundationTreeNode]
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
    result_cache:    Dict[str, Any]
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

def _get_semantic_context(request: Request, uid: str, session_id: str) -> SemanticContext:
    orchestrator = request.app.state.orchestrator
    config = {"configurable": {"thread_id": f"{uid}:{session_id}"}}
    try:
        snapshot = orchestrator.get_state(config)
    except Exception:
        return SemanticContext()

    values = snapshot.values or {}
    socrates_state = values.get("socrates_state") or {}
    return SemanticContext(
        query=socrates_state.get("query") or "",
        intent_types=socrates_state.get("intent_types") or [],
        summary=socrates_state.get("context_summary") or "",
        thinking_process=values.get("thinking_process") or "",
    )

# ── Routes ────────────────────────────────────────────────────────────

@router.get("/graph/expand/{eid}", response_model=ExpandResponse)
async def expand_concept(eid: str, request: Request) -> ExpandResponse:
    coordinator  = request.app.state.coordinator
    anchor_label = _get_anchor_label(request, eid)

    result = await coordinator.get_expand_cached_async(eid, max_depth=6, max_branches=4)


    foundations = [
        FoundationTreeNode(element_id=f["element_id"], label=f["label"])
        for f in result.get("final_foundations", [])
    ]
    tree_nodes = [
        FoundationTreeNode(element_id=n["element_id"], label=n["label"])
        for n in result.get("foundation_tree", {}).get("nodes", [])
    ]
    tree_edges = [
        FoundationTreeEdge(source=e["source"], target=e["target"])
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
            FoundationTreeNode(element_id=f["element_id"], label=f["label"])
            for f in result.get("final_foundations", [])
        ]
        tree_nodes = [
            FoundationTreeNode(element_id=n["element_id"], label=n["label"])
            for n in result.get("foundation_tree", {}).get("nodes", [])
        ]
        tree_edges = [
            FoundationTreeEdge(source=e["source"], target=e["target"])
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
async def rc_paths(eid: str, request: Request) -> RCPathsResponse:
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
        _get_semantic_context(request, uid, session_id)
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
    ctx = _get_semantic_context(request, uid, session_id)
    return SemanticContextResponse(
        session_id=session_id,
        query=ctx.query,
        intent_types=ctx.intent_types,
        summary=ctx.summary,
        thinking_process=ctx.thinking_process,
    )


@router.get("/admin/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request) -> CacheStatsResponse:
    """回傳各層 cache 的統計資訊（命中率、大小等）。

    用於監控與調校 LRU maxsize。
    """
    coordinator = request.app.state.coordinator
    stats       = coordinator.stats

    return CacheStatsResponse(
        adjacency_cache=stats["adjacency_cache"],
        result_cache=stats["result_cache"],
        rcpath_cache=stats["rcpath_cache"],
        expand_cache=stats["expand_cache"],
    )



@router.get("/graph/bertopic-layout", response_model=BertopicLayoutResponse)
async def bertopic_layout(request: Request) -> BertopicLayoutResponse:
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
    ctx = _get_semantic_context(request, uid, session_id)
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
