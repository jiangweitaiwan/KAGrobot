# -*- coding: utf-8 -*-
"""api/routes/users.py — 會員系統核心 endpoints。

GET    /me            → 目前登入者的 profile（首次登入自動建立文件）
PATCH  /me             → 更新 profile（目前僅 display_name）
GET    /me/sessions    → 目前登入者的對話 session 列表
DELETE /me              → 刪除帳號（Firebase Auth + Firestore 資料）

uid 一律來自 Depends(require_auth / require_auth_claims) 驗證過的
Firebase ID token，不接受任何前端自報的 uid。

【會員系統】匿名 → 正式帳號的「升級」流程完全發生在前端
（firebase/auth 的 linkWithPopup / linkWithCredential），uid 全程不變，
因此這裡不需要任何「合併資料」的邏輯——升級前後讀寫的都是同一份
users/{uid} 底下的資料。GET /me 只是如實反映 token 當下的
is_anonymous / provider 狀態。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional  # ← 加入 Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import auth as fb_auth
from pydantic import BaseModel

from core.firebase_auth import require_auth, require_auth_claims
from core.session_store import list_sessions, load_session_display  # ← 新增 import

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
    email_verified: bool = False
    display_name:   Optional[str] = None
    is_anonymous:   bool = False
    provider:       str = "anonymous"
    plan:           str = "free"
    created_at:     int
    last_login_at:  int


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    phase:      Optional[str] = None
    summary:    str = ""
    updated_at: Optional[int] = None
    created_at: Optional[int] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]


class SessionDisplayMessage(BaseModel):
    role:        str
    content:     str
    attachments: List[Dict[str, Any]] = []
    created_at:  int


class SessionDetailResponse(BaseModel):
    session_id:        str
    phase:             str
    display_messages:  List[SessionDisplayMessage]
    selected_rc_eid:   Optional[str] = None
    selected_rc_label: Optional[str] = None

class DeleteAccountResponse(BaseModel):
    ok: bool


# ── Helpers ───────────────────────────────────────────────────────────

def _provider_of(decoded: dict) -> str:
    firebase_info    = decoded.get("firebase", {}) or {}
    sign_in_provider = firebase_info.get("sign_in_provider", "")
    return {
        "anonymous": "anonymous",
        "password":  "password",
        "google.com": "google",
    }.get(sign_in_provider, sign_in_provider or "unknown")


async def _load_profile(claims: dict) -> ProfileResponse:
    """讀取（必要時建立）users/{uid} 文件，組成 ProfileResponse。

    is_anonymous / provider / email_verified 一律直接依「當前這次請求
    的 ID token」判斷，不快取進 Firestore、也不信任 Firestore 裡的舊值——
    使用者剛完成匿名升級時，token 是最新真相來源。
    """
    uid      = claims["uid"]
    provider = _provider_of(claims)
    now      = int(time.time() * 1000)

    db  = _get_db()
    ref = db.collection("users").document(uid)
    doc = await ref.get()

    if not doc.exists:
        data = {
            "email":         claims.get("email"),
            "display_name":  claims.get("name"),
            "plan":          "free",
            "created_at":    now,
            "last_login_at": now,
        }
        await ref.set(data)
    else:
        data = doc.to_dict() or {}
        data.setdefault("email", claims.get("email"))
        data.setdefault("display_name", claims.get("name"))
        data.setdefault("plan", "free")
        data.setdefault("created_at", now)

        updates: dict = {"last_login_at": now}
        # 升級為正式帳號後 token 上會多出 / 換掉 email，順手補寫回 Firestore，
        # 讓其他不會帶 token 的內部工具（例如未來的管理後台）也能讀到。
        if claims.get("email") and data.get("email") != claims.get("email"):
            updates["email"] = claims.get("email")
            data["email"] = claims.get("email")

        await ref.set(updates, merge=True)
        data["last_login_at"] = now

    return ProfileResponse(
        uid=uid,
        email=data.get("email"),
        email_verified=bool(claims.get("email_verified", False)),
        display_name=data.get("display_name"),
        is_anonymous=(provider == "anonymous"),
        provider=provider,
        plan=data.get("plan", "free"),
        created_at=data.get("created_at", now),
        last_login_at=data.get("last_login_at", now),
    )


async def _delete_collection(collection_ref, batch_size: int = 100) -> None:
    """遞迴刪除一個 collection 底下所有文件（Firestore 沒有原生的
    「刪除整個 collection」操作，需要自行分批刪）。"""
    docs  = collection_ref.limit(batch_size).stream()
    count = 0
    async for doc in docs:
        await doc.reference.delete()
        count += 1
    if count >= batch_size:
        await _delete_collection(collection_ref, batch_size)


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(claims: dict = Depends(require_auth_claims)) -> ProfileResponse:
    return await _load_profile(claims)


@router.patch("/me", response_model=ProfileResponse)
async def update_my_profile(
    body:   UpdateProfileRequest,
    claims: dict = Depends(require_auth_claims),
) -> ProfileResponse:
    """更新可編輯的 profile 欄位（目前僅 display_name）。

    email 不在這裡更新——email 的異動須透過 Firebase Auth SDK 本身
    （verifyBeforeUpdateEmail 等），完成後下次登入 token 會自然反映，
    _load_profile 會自動同步回 Firestore。
    """
    uid = claims["uid"]
    if body.display_name is not None:
        db  = _get_db()
        ref = db.collection("users").document(uid)
        await ref.set({"display_name": body.display_name}, merge=True)

    return await _load_profile(claims)


@router.get("/me/sessions", response_model=SessionListResponse)
async def get_my_sessions(uid: str = Depends(require_auth)) -> SessionListResponse:
    """回傳目前登入者的對話 session 列表（依最後更新時間新到舊）。

    資料來源為 core/session_store.py 在每輪 /chat 結束後寫入的備份，
    見 api/routes/chat.py。
    """
    sessions = await list_sessions(uid)
    return SessionListResponse(sessions=[SessionSummary(**s) for s in sessions])



@router.get("/me/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    uid:        str = Depends(require_auth),
) -> SessionDetailResponse:
    """回傳指定 session 的完整聊天紀錄，供前端「接續歷史對話」還原畫面。

    display_messages 已是前端 ChatMessage 可直接渲染的格式
    （role/content/attachments），不需要再額外轉換。
    """
    detail = await load_session_display(uid, session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="找不到此對話紀錄")

    return SessionDetailResponse(
        session_id=detail["session_id"],
        phase=detail["phase"],
        display_messages=[
            SessionDisplayMessage(**m) for m in detail["display_messages"]
        ],
        selected_rc_eid=detail.get("selected_rc_eid"),
        selected_rc_label=detail.get("selected_rc_label"),
    )


@router.delete("/me", response_model=DeleteAccountResponse)
async def delete_my_account(uid: str = Depends(require_auth)) -> DeleteAccountResponse:
    """刪除帳號：清空 Firestore 內所有相關資料，再刪除 Firebase Auth 使用者。

    順序刻意「先刪資料、後刪帳號」：若中途失敗，寧可留下一個帳號還在但
    資料已清光的狀態（可重試整個流程），也不要留下一個帳號已經無法登入、
    但資料還留著的狀態（使用者將永遠無法自行觸發清理）。

    fb_auth.delete_user 成功後，Firebase 會讓該 uid 之後所有 ID token
    驗證失敗，等同立即登出所有裝置。
    """
    db = _get_db()

    try:
        await _delete_collection(
            db.collection("users").document(uid).collection("sessions")
        )
        await (
            db.collection("users").document(uid)
              .collection("brain").document("snapshot").delete()
        )
        await db.collection("users").document(uid).delete()
    except Exception as exc:
        print(f"[users] Firestore cleanup failed for uid={uid}: {exc}")
        raise HTTPException(status_code=500, detail="清除使用者資料失敗，請稍後再試")

    try:
        fb_auth.delete_user(uid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"刪除登入帳號失敗：{exc}")

    return DeleteAccountResponse(ok=True)
