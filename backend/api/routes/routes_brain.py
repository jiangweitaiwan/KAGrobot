# -*- coding: utf-8 -*-
"""api/routes/routes_brain.py — PersonalBrain 相關 endpoints。

GET  /graph/expand/{eid}          → 取得 concept 的 foundation tree（UpstreamBFS）
GET  /graph/rc-paths/{eid}        → 取得 RC 的 downstream paths（DownstreamBFS）
GET  /brain/rc-semantic-scores    → 取得所有 RC 對 query 的 SBERT 語意相似度
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel

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
    foundations:  List[FoundationTreeNode]  # leaf foundations
    nodes:        List[FoundationTreeNode]  # all tree nodes
    edges:        List[FoundationTreeEdge]


class PathNode(BaseModel):
    id:           str
    label:        str
    node_type:    str   # "concept" | "paper"
    concept_type: Optional[str]
    matched:      bool

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
    eid:   str
    label: str
    score: float

class SemanticScoresResponse(BaseModel):
    query:  str
    scores: List[RCSemanticScore]


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/graph/expand/{eid}", response_model=ExpandResponse)
async def expand_concept(eid: str, request: Request) -> ExpandResponse:
    """取得指定 concept 的 foundation tree（UpstreamBFS）。

    回傳供 DetailPanel 渲染 foundation tree 拓墣用的節點與邊。
    """
    upstream_bfs = request.app.state.upstream_bfs

    # 取得 anchor concept 名稱
    driver = request.app.state.driver
    with driver.session() as session:
        rec = session.run(
            "MATCH (c:Concept) WHERE elementId(c) = $eid RETURN c.name AS name",
            eid=eid,
        ).single()
    anchor_label = rec["name"] if rec and rec["name"] else eid

    result = await asyncio.to_thread(
        upstream_bfs.run, eid, max_depth=6, max_branches=4
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

    # 確保 anchor 也在 nodes 中
    anchor_in_nodes = any(n.element_id == eid for n in tree_nodes)
    if not anchor_in_nodes:
        tree_nodes.append(FoundationTreeNode(element_id=eid, label=anchor_label))

    return ExpandResponse(
        anchor_eid=eid,
        anchor_label=anchor_label,
        foundations=foundations,
        nodes=tree_nodes,
        edges=tree_edges,
    )


@router.get("/graph/rc-paths/{eid}", response_model=RCPathsResponse)
async def rc_paths(eid: str, request: Request) -> RCPathsResponse:
    """取得指定 RC 的 downstream paths（DownstreamBFS）。

    供 PersonalBrain 點選 RC 標籤後亮起對應 path 用。
    """
    downstream_bfs = request.app.state.downstream_bfs
    driver         = request.app.state.driver

    with driver.session() as session:
        rec = session.run(
            "MATCH (c:Concept) WHERE elementId(c) = $eid RETURN c.name AS name",
            eid=eid,
        ).single()
    rc_label = rec["name"] if rec and rec["name"] else eid

    raw = await asyncio.to_thread(
        downstream_bfs.run, eid, rc_label, max_depth_down=6
    )

    nodes = [
        PathNode(
            id=n["id"],
            label=n["label"],
            node_type=n["node_type"],
            concept_type=n.get("concept_type"),
            matched=n.get("matched", False),
        )
        for n in raw.get("nodes", [])
    ]
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
    query:   str,
    request: Request,
) -> SemanticScoresResponse:
    """計算所有 Resulting Concept 對 query 的 SBERT 語意相似度。

    在 concept_resolved 階段呼叫一次，結果存在 frontend state。
    α 係數（預設 0.5）在 frontend 做加權合併。
    """
    encoder      = request.app.state.encoder
    layout_cache = request.app.state.layout_cache

    if not layout_cache or not query.strip():
        return SemanticScoresResponse(query=query, scores=[])

    # 從 layout_cache 取所有 RC 節點
    rc_nodes = [
        n for n in layout_cache.get("nodes", [])
        if (n.get("type") or "").lower().find("resulting") >= 0
    ]
    if not rc_nodes:
        return SemanticScoresResponse(query=query, scores=[])

    names   = [n["name"] for n in rc_nodes]
    eids    = [n["id"]   for n in rc_nodes]

    # SBERT encode：分批，避免 OOM
    query_emb = await asyncio.to_thread(lambda: encoder.encode(query)[0])

    batch_size = 64
    emb_parts: List[np.ndarray] = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        part  = await asyncio.to_thread(lambda b=batch: encoder.encode(b))
        emb_parts.append(part)

    emb_matrix = np.vstack(emb_parts)
    raw_scores  = (emb_matrix @ query_emb).tolist()

    scores = [
        RCSemanticScore(eid=eids[i], label=names[i], score=float(raw_scores[i]))
        for i in range(len(rc_nodes))
    ]
    # 降序排列
    scores.sort(key=lambda s: s.score, reverse=True)

    return SemanticScoresResponse(query=query, scores=scores)
