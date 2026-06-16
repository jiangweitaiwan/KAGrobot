# -*- coding: utf-8 -*-
"""api/routes/concept.py — Concept 相關 endpoints。

GET  /concept/detail/{eid}  → 根據 elementId 查詢 concept 詳情
POST /concept/suggest       → LLM 非同步推薦相關 concept
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.google_llm import create_chat_google_generative_ai

router = APIRouter()


# ── Schema ────────────────────────────────────────────────────────────

class ConceptDetailResponse(BaseModel):
    found:          bool
    eid:            str
    rc_name:        str
    description:    Optional[str]
    function:       Optional[str]
    analysis_types: List[str]
    concept_type:   Optional[str]
    papers: List[Dict[str, Any]]


class SuggestRequest(BaseModel):
    text:         str                    # 最近幾輪對話文字
    anchor_eid:   Optional[str] = None  # 目前 Detail Panel 聚焦的 concept eid
    anchor_label: Optional[str] = None
    exclude_eids: List[str]     = []


class SuggestionItem(BaseModel):
    eid:        str
    label:      str
    reason:     str
    confidence: float


class SuggestResponse(BaseModel):
    suggestions: List[SuggestionItem]


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/concept/detail/{eid}", response_model=ConceptDetailResponse)
async def concept_detail(eid: str, request: Request) -> ConceptDetailResponse:
    """根據 elementId 查詢 concept 詳情。

    直接用 elementId(c) = $eid 精確匹配，比 name 查詢更快且不會有歧義。
    """
    driver = request.app.state.driver

    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            OPTIONAL MATCH (p:Paper)-[:PRODUCES]->(c)
            RETURN
                c.name             AS name,
                c.type             AS type,
                c.description      AS description,
                c.function         AS function,
                c.analysis_types   AS analysis_types,
                collect({
                    title:       p.title,
                    premises:    p.conceptual_premises,
                    constraints: p.contextual_model_constraints
                }) AS papers
            LIMIT 1
            """,
            eid=eid,
        ).data()

    if not rows:
        return ConceptDetailResponse(
            found=False, eid=eid, rc_name="", description=None,
            function=None, analysis_types=[], concept_type=None, papers=[],
        )

    row    = rows[0]
    papers = [
        {
            "title":       p["title"],
            "premises":    p["premises"],
            "constraints": p["constraints"],
        }
        for p in (row["papers"] or [])
        if p.get("title")
    ]

    return ConceptDetailResponse(
        found          = True,
        eid            = eid,
        rc_name        = row["name"] or "",
        description    = row["description"],
        function       = row["function"],
        analysis_types = row["analysis_types"] or [],
        concept_type   = row["type"],
        papers         = papers,
    )


@router.post("/concept/suggest", response_model=SuggestResponse)
async def concept_suggest(
    body:    SuggestRequest,
    request: Request,
) -> SuggestResponse:
    """LLM 推薦與對話相關的 concept，獨立於 chat API，非同步回傳。

    流程：
    1. 用 layout_cache 的 nodes 做前置過濾（排除 exclude_eids）
    2. 用 SBERT embedding 取 top-40 候選（語意相似度過濾，減少 LLM token）
    3. 呼叫輕量 Gemini Flash，從候選中精選並附上推薦理由
    """
    layout_cache = request.app.state.layout_cache
    settings     = request.app.state.settings

    if not layout_cache:
        return SuggestResponse(suggestions=[])

    all_nodes: List[Dict[str, Any]] = layout_cache.get("nodes", [])
    exclude_set = set(body.exclude_eids)

    # 前置過濾：排除已在 personal brain 的、優先 RC 類型
    candidates = [
        n for n in all_nodes
        if n["id"] not in exclude_set
        and n["id"] != body.anchor_eid
    ]

    if not candidates:
        return SuggestResponse(suggestions=[])

    # SBERT 前置過濾：取 top-40 語意相近候選，減少送給 LLM 的 token 數
    encoder = request.app.state.encoder  # SbertOnnxEncoder，從 app.state 取
    top_candidates = _sbert_prefilter(
        body.text, candidates, encoder, top_k=40
    )

    # 組裝 LLM prompt
    candidate_lines = "\n".join(
        f"- eid={n['id']} | label={n['name']} | type={n.get('type','')}"
        for n in top_candidates
    )
    anchor_hint = (
        f"目前聚焦的概念：{body.anchor_label}（eid={body.anchor_eid}）\n"
        if body.anchor_label else ""
    )

    prompt = f"""你是一位知識圖譜分析助手。
{anchor_hint}
以下是最近幾輪的對話內容：
---
{body.text[:800]}
---

從下方 concept 候選清單中，選出最多 4 個與對話主題高度相關的概念。
每個選出的概念需附上簡短理由（15 字以內的繁體中文）和信心分數（0.0–1.0）。

候選清單：
{candidate_lines}

只回傳合法 JSON，格式如下，不要任何前綴或 markdown：
[{{"eid":"...","label":"...","reason":"...","confidence":0.0}}]
"""

    try:
        llm = create_chat_google_generative_ai(
            model="gemini-2.0-flash",
            temperature=0.1,
            settings=settings,
        )
        resp = llm.invoke(prompt)
        raw  = resp.content if hasattr(resp, "content") else str(resp)

        # 清除可能的 markdown fence
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        items = json.loads(raw)
        suggestions = [
            SuggestionItem(
                eid        = item["eid"],
                label      = item["label"],
                reason     = item.get("reason", ""),
                confidence = float(item.get("confidence", 0.5)),
            )
            for item in items
            if isinstance(item, dict) and item.get("eid") and item.get("label")
        ]
    except Exception as exc:
        print(f"[concept/suggest] LLM error: {exc}")
        suggestions = []

    return SuggestResponse(suggestions=suggestions)


# ── Helpers ───────────────────────────────────────────────────────────

def _sbert_prefilter(
    query:      str,
    candidates: List[Dict[str, Any]],
    encoder,
    top_k:      int = 40,
) -> List[Dict[str, Any]]:
    """用 SBERT embedding 對候選做語意過濾，回傳 top_k 最相近的節點。"""
    import numpy as np

    try:
        query_emb = encoder.encode(query)[0]
        names     = [n.get("name", "") for n in candidates]

        # 分批 encode，避免大 batch 耗記憶體
        batch_size = 64
        embs = []
        for i in range(0, len(names), batch_size):
            embs.append(encoder.encode(names[i:i + batch_size]))
        emb_matrix = np.vstack(embs)

        scores = emb_matrix @ query_emb
        top_indices = scores.argsort()[::-1][:top_k]
        return [candidates[i] for i in top_indices]
    except Exception as exc:
        print(f"[_sbert_prefilter] error: {exc}, returning first {top_k}")
        return candidates[:top_k]
