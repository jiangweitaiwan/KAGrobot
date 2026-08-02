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

【更新】uid 擁有權驗證：
  舊版設計是「uid 由瀏覽器自行產生的 UUID，不需要 auth」——
  這代表任何人只要知道別人的 UUID，就能讀寫別人的 Personal Brain。
  現在改為：uid 必須與 Authorization header 帶的 Firebase ID token
  解出的 uid 一致，否則回傳 403（見 _verify_owner）。
  路徑仍保留 {uid}，是因為前端既有的匿名 UUID 使用者可以透過
  Firebase「匿名登入」升級成正式帳號、uid 不變，資料不需要遷移。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.firebase_auth import require_auth

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


# ── Auth dependency ───────────────────────────────────────────────────

async def _verify_owner(uid: str, caller_uid: str = Depends(require_auth)) -> str:
    """確保呼叫者已驗證的 uid 與路徑上的 uid 一致，否則 403。

    caller_uid 由 Firebase ID token 解出，無法偽造；
    uid 仍做基本格式檢查，防止 path traversal 之類的異常輸入。
    """
    _validate_uid(uid)
    if caller_uid != uid:
        raise HTTPException(status_code=403, detail="無權存取此使用者的資料")
    return uid


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/user/{uid}/brain", response_model=BrainSnapshot)
async def get_brain(uid: str, _owner: str = Depends(_verify_owner)) -> BrainSnapshot:
    """讀取 Personal Brain 完整 snapshot。"""
    db  = _get_db()
    doc = await db.collection("users").document(uid).collection("brain").document("snapshot").get()

    if not doc.exists:
        return BrainSnapshot(concepts=[], updatedAt=0)

    data     = doc.to_dict() or {}
    concepts = [BrainConcept(**c) for c in (data.get("concepts") or [])]
    return BrainSnapshot(concepts=concepts, updatedAt=data.get("updatedAt", 0))


@router.post("/user/{uid}/brain/concepts", response_model=BrainSnapshot)
async def add_concept(
    uid:    str,
    body:   BrainConcept,
    _owner: str = Depends(_verify_owner),
) -> BrainSnapshot:
    """加入或更新一個 concept（eid 已存在則更新 weight / source）。"""
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
async def update_weight(
    uid:    str,
    eid:    str,
    body:   UpdateWeightRequest,
    _owner: str = Depends(_verify_owner),
) -> BrainSnapshot:
    """只更新 weight，不異動其他欄位。"""
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
async def remove_concept(
    uid:    str,
    eid:    str,
    _owner: str = Depends(_verify_owner),
) -> BrainSnapshot:
    """從 Personal Brain 移除一個 concept。"""
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
