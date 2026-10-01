# -*- coding: utf-8 -*-
"""api/routes/user_brain.py — Personal Brain 持久化 endpoints。

GET    /user/{uid}/brain                → 讀取 Personal Brain snapshot
POST   /user/{uid}/brain/concepts       → 加入或更新一個 concept
PATCH  /user/{uid}/brain/concepts/{eid} → 只更新 weight
DELETE /user/{uid}/brain/concepts/{eid} → 移除一個 concept

儲存後端：Google Cloud Firestore。
Collection 結構：
  users/{uid}/brain → {
      concepts: [{ eid, label, source, weight, addedAt }, ...],
      updatedAt: int (unix ms)
  }

uid 擁有權驗證：
  uid 必須與 Authorization header 帶的 Firebase ID token 解出的 uid 一致，
  否則回傳 403（見 _verify_owner）。路徑仍保留 {uid}，是因為匿名使用者
  升級成正式帳號時 uid 不變，資料不需要遷移。

【新增 — 會員等級】
  Personal Brain 的「概念數上限」依等級而定（PlanLimits.brain_max_concepts）。
  只限制「新增一個全新的 eid」；更新既有 concept 的權重不受影響，
  也不會因為降級而刪除超出上限的既有資料（只是不能再新增）。
  超過上限回 409，detail.code = "brain_limit"。

【Phase 0】Firestore client 改由 core/firestore_db.py 統一提供。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.infra.firebase_auth import require_auth_claims
from core.infra.firestore_db import get_default_db as _get_db
from core.account.membership import resolve_membership

router = APIRouter()


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

async def _verify_owner(uid: str, claims: dict = Depends(require_auth_claims)) -> dict:
    """確保呼叫者已驗證的 uid 與路徑上的 uid 一致，否則 403。

    回傳完整 claims（供需要判斷等級的路由使用）。uid 仍做基本格式檢查，
    防止 path traversal 之類的異常輸入。
    """
    _validate_uid(uid)
    if claims.get("uid") != uid:
        raise HTTPException(status_code=403, detail="無權存取此使用者的資料")
    return claims


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/user/{uid}/brain", response_model=BrainSnapshot)
async def get_brain(uid: str, _owner: dict = Depends(_verify_owner)) -> BrainSnapshot:
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
    claims: dict = Depends(_verify_owner),
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
        membership = await resolve_membership(claims)
        limit = membership.limits.brain_max_concepts
        if len(concepts) >= limit and not membership.is_admin:
            raise HTTPException(
                status_code=409,
                detail={
                    "code":    "brain_limit",
                    "message": f"Personal Brain 已達上限（{limit} 個概念），請先移除部分概念"
                               + ("，或註冊帳號取得更高上限。" if membership.is_anonymous else "。"),
                    "plan":    membership.plan,
                    "limit":   limit,
                },
            )
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
    _owner: dict = Depends(_verify_owner),
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
    _owner: dict = Depends(_verify_owner),
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
