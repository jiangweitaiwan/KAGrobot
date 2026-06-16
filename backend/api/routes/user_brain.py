# -*- coding: utf-8 -*-
"""api/routes/user_brain.py — Personal Brain 持久化 endpoints。

GET  /user/{uid}/brain              → 讀取 Personal Brain snapshot
POST /user/{uid}/brain/concepts     → 加入或更新一個 concept
DEL  /user/{uid}/brain/concepts/{eid} → 移除一個 concept

儲存後端：Google Cloud Firestore。
Collection 結構：
  users/{uid}/brain → {
      concepts: [{ eid, label, source, weight, addedAt }, ...],
      updatedAt: int (unix ms)
  }

uid 由瀏覽器自行產生（UUID），不需要 auth。
這是有意的設計決策：Personal Brain 是輕量個人偏好，
不涉及敏感資料，未來要加 auth 只需在此加 middleware。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

# Lazy Firestore client（避免 import 時就初始化影響啟動時間）
_db = None

def _get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.AsyncClient()
    return _db


# ── Schema ────────────────────────────────────────────────────────────

class BrainConcept(BaseModel):
    eid:     str
    label:   str
    source:  Literal["manual", "llm_suggest", "path_confirm"]
    weight:  float = Field(ge=0.0, le=1.0, default=0.5)
    addedAt: int   = Field(default_factory=lambda: int(time.time() * 1000))


class BrainSnapshot(BaseModel):
    concepts:  List[BrainConcept]
    updatedAt: int


class UpdateWeightRequest(BaseModel):
    weight: float = Field(ge=0.0, le=1.0)


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/user/{uid}/brain", response_model=BrainSnapshot)
async def get_brain(uid: str) -> BrainSnapshot:
    """讀取 Personal Brain 完整 snapshot。"""
    _validate_uid(uid)
    db  = _get_db()
    doc = await db.collection("users").document(uid).collection("brain").document("snapshot").get()

    if not doc.exists:
        return BrainSnapshot(concepts=[], updatedAt=0)

    data     = doc.to_dict() or {}
    concepts = [BrainConcept(**c) for c in (data.get("concepts") or [])]
    return BrainSnapshot(concepts=concepts, updatedAt=data.get("updatedAt", 0))


@router.post("/user/{uid}/brain/concepts", response_model=BrainSnapshot)
async def add_concept(uid: str, body: BrainConcept) -> BrainSnapshot:
    """加入或更新一個 concept（eid 已存在則更新 weight / source）。"""
    _validate_uid(uid)
    db      = _get_db()
    ref     = db.collection("users").document(uid).collection("brain").document("snapshot")
    doc     = await ref.get()
    data    = doc.to_dict() if doc.exists else {}
    concepts: List[Dict[str, Any]] = data.get("concepts") or []

    # 更新或追加
    exists = False
    for c in concepts:
        if c.get("eid") == body.eid:
            c["weight"]  = body.weight
            c["source"]  = body.source
            c["label"]   = body.label
            c["addedAt"] = body.addedAt
            exists = True
            break
    if not exists:
        concepts.append(body.model_dump())

    now = int(time.time() * 1000)
    await ref.set({"concepts": concepts, "updatedAt": now})

    return BrainSnapshot(
        concepts  = [BrainConcept(**c) for c in concepts],
        updatedAt = now,
    )


@router.patch("/user/{uid}/brain/concepts/{eid}", response_model=BrainSnapshot)
async def update_weight(uid: str, eid: str, body: UpdateWeightRequest) -> BrainSnapshot:
    """只更新 weight，不異動其他欄位。"""
    _validate_uid(uid)
    db  = _get_db()
    ref = db.collection("users").document(uid).collection("brain").document("snapshot")
    doc = await ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Brain snapshot not found")

    data     = doc.to_dict() or {}
    concepts = data.get("concepts") or []
    found    = False
    for c in concepts:
        if c.get("eid") == eid:
            c["weight"] = body.weight
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Concept {eid} not in brain")

    now = int(time.time() * 1000)
    await ref.set({"concepts": concepts, "updatedAt": now})
    return BrainSnapshot(concepts=[BrainConcept(**c) for c in concepts], updatedAt=now)


@router.delete("/user/{uid}/brain/concepts/{eid}", response_model=BrainSnapshot)
async def remove_concept(uid: str, eid: str) -> BrainSnapshot:
    """從 Personal Brain 移除一個 concept。"""
    _validate_uid(uid)
    db  = _get_db()
    ref = db.collection("users").document(uid).collection("brain").document("snapshot")
    doc = await ref.get()

    if not doc.exists:
        return BrainSnapshot(concepts=[], updatedAt=0)

    data     = doc.to_dict() or {}
    concepts = [c for c in (data.get("concepts") or []) if c.get("eid") != eid]

    now = int(time.time() * 1000)
    await ref.set({"concepts": concepts, "updatedAt": now})
    return BrainSnapshot(concepts=[BrainConcept(**c) for c in concepts], updatedAt=now)


# ── Helpers ───────────────────────────────────────────────────────────

def _validate_uid(uid: str) -> None:
    """基本格式驗證，防止 path traversal。uid 只允許 alphanumeric 和 -。"""
    if not uid or not all(c.isalnum() or c == "-" for c in uid) or len(uid) > 64:
        raise HTTPException(status_code=400, detail="Invalid uid format")
