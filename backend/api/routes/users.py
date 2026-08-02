# -*- coding: utf-8 -*-
"""api/routes/users.py — 使用者 Profile 與 Session 列表 endpoints。

GET /me            → 目前登入者的 profile（首次登入自動建立文件）
GET /me/sessions   → 目前登入者的對話 session 列表

uid 一律來自 Depends(require_auth) 驗證過的 Firebase ID token，
不接受任何前端自報的 uid，杜絕冒用他人資料的問題
（對照 api/routes/user_brain.py 原本「uid 由前端自行生成、
後端完全不驗證」的舊設計）。
"""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.firebase_auth import require_auth
from core.session_store import list_sessions

router = APIRouter()

_db = None


def _get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.AsyncClient()
    return _db


# ── Schema ────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    uid:            str
    email:          Optional[str] = None
    display_name:   Optional[str] = None
    plan:           str = "free"
    created_at:     int
    last_login_at:  int


class SessionSummary(BaseModel):
    session_id: str
    phase:      Optional[str] = None
    summary:    str = ""
    updated_at: Optional[int] = None
    created_at: Optional[int] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(uid: str = Depends(require_auth)) -> ProfileResponse:
    """回傳目前登入者的 profile；首次呼叫自動建立文件並記錄 last_login_at。

    email / display_name 目前留空，之後可由前端在首次呼叫時，
    把 Firebase ID token 解出的 email / name 一併帶入補寫；
    也可以改在 firebase_auth.require_auth 內把 decoded token 整包
    傳出來，在這裡順手補寫，看之後前端資料流怎麼設計再調整。
    """
    db  = _get_db()
    ref = db.collection("users").document(uid)
    doc = await ref.get()

    now = int(time.time() * 1000)

    if not doc.exists:
        data = {
            "email":          None,
            "display_name":   None,
            "plan":           "free",
            "created_at":     now,
            "last_login_at":  now,
        }
        await ref.set(data)
    else:
        data = doc.to_dict() or {}
        data.setdefault("email", None)
        data.setdefault("display_name", None)
        data.setdefault("plan", "free")
        data.setdefault("created_at", now)
        await ref.set({"last_login_at": now}, merge=True)
        data["last_login_at"] = now

    return ProfileResponse(uid=uid, **data)


@router.get("/me/sessions", response_model=SessionListResponse)
async def get_my_sessions(uid: str = Depends(require_auth)) -> SessionListResponse:
    """回傳目前登入者的對話 session 列表（依最後更新時間新到舊）。

    資料來源為 core/session_store.py 在每輪 /chat 結束後寫入的備份，
    見 api/routes/chat.py。
    """
    sessions = await list_sessions(uid)
    return SessionListResponse(sessions=[SessionSummary(**s) for s in sessions])
