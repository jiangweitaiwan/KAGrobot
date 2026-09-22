# -*- coding: utf-8 -*-
"""api/routes/users.py — 會員系統核心 endpoints。

GET    /me            → 目前登入者的 profile（首次登入自動建立文件）
PATCH  /me             → 更新 profile（目前僅 display_name）
GET    /me/sessions    → 目前登入者的對話 session 列表（含每個對話的 token 用量）
DELETE /me              → 刪除帳號（Firebase Auth + Firestore 資料）

uid 一律來自 Depends(require_auth / require_auth_claims) 驗證過的
Firebase ID token，不接受任何前端自報的 uid。

【會員系統】匿名 → 正式帳號的「升級」流程完全發生在前端
（firebase/auth 的 linkWithPopup / linkWithCredential），uid 全程不變，
因此這裡不需要任何「合併資料」的邏輯——升級前後讀寫的都是同一份
users/{uid} 底下的資料。GET /me 只是如實反映 token 當下的
is_anonymous / provider 狀態。

【新增 — HTML Widget】
  SessionDisplayMessage 多一個選填的 widget 欄位，對應
  core/session_store.py 存進 display_messages 裡、某則 AI turn message
  可能帶的 "widget" 資料（見 api/routes/chat.py 的 ai_turn_message 組裝）。
  純粹透傳給前端重播用，這裡不做任何額外處理或驗證。

【新增 — 會員等級 / 用量】
  - ProfileResponse.plan 一律由 core/membership.resolve_membership 解析
    （匿名 → guest、VIP 過期 → normal、舊值 "free" → normal），
    與額度檢查實際套用的規則是同一份。
  - GET /me/sessions 附上每個對話的 title 與 total_tokens。
  - DELETE /me 的資料清理改為兩個資料庫都清：sessions 實際存放在
    kagrobot-state（見 core/firestore_db.py），原本只刪預設資料庫、
    刪不到那邊的 sessions；並一併清除 usage_* 集合。
    （usage_global_* 全站彙總不屬於任何使用者，不會也不應該被動到。）
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional  # ← 加入 Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from firebase_admin import auth as fb_auth
from pydantic import BaseModel

from core.firebase_auth import require_auth, require_auth_claims
from core.firestore_db import get_default_db
from core.membership import resolve_membership
from core.session_store import (  # ← 新增 import
    delete_all_sessions,
    list_sessions,
    load_session_display,
)
from core.usage import delete_user_usage, get_sessions_usage

router = APIRouter()


def _get_db():
    return get_default_db()


# ── Schema ────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    uid:             str
    email:           Optional[str] = None
    email_verified:  bool = False
    display_name:    Optional[str] = None
    is_anonymous:    bool = False
    provider:        str = "anonymous"
    plan:            str = "normal"
    plan_expires_at: Optional[int] = None
    created_at:      int
    last_login_at:   int


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None


class SessionSummary(BaseModel):
    session_id:   str
    title:        str = ""
    phase:        Optional[str] = None
    summary:      str = ""
    updated_at:   Optional[int] = None
    created_at:   Optional[int] = None
    total_tokens: int = 0


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]


class SessionDisplayMessage(BaseModel):
    role:        str
    content:     str
    attachments: List[Dict[str, Any]] = []
    created_at:  int
    # 新增 — HTML Widget：這則訊息若曾顯示過互動元件，帶
    # {widget_id, title, url, created_at, expires_in_hours}。
    # 前端依 created_at + expires_in_hours 判斷是否過期（比照
    # AttachmentCard.isAttachmentExpired），過期則顯示灰階、不渲染 iframe。
    widget:      Optional[Dict[str, Any]] = None


class SessionDetailResponse(BaseModel):
    session_id:             str
    phase:                  str
    display_messages:       List[SessionDisplayMessage]
    selected_rc_eid:        Optional[str] = None
    selected_rc_label:      Optional[str] = None
    walk_session:           Optional[Dict[str, Any]] = None
    walk_pending_question:  Optional[Dict[str, Any]] = None

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
            "plan":          "normal",
            "created_at":    now,
            "last_login_at": now,
        }
        await ref.set(data)
    else:
        data = doc.to_dict() or {}
        # admin 授予等級等流程可能比 GET /me 更早建立這份文件，此時沒有 created_at。
        missing_created = "created_at" not in data
        data.setdefault("email", claims.get("email"))
        data.setdefault("display_name", claims.get("name"))
        data.setdefault("created_at", now)

        updates: dict = {"last_login_at": now}
        if missing_created:
            updates["created_at"] = now
        # 升級為正式帳號後 token 上會多出 / 換掉 email，順手補寫回 Firestore，
        # 讓其他不會帶 token 的內部工具（例如未來的管理後台）也能讀到。
        if claims.get("email") and data.get("email") != claims.get("email"):
            updates["email"] = claims.get("email")
            data["email"] = claims.get("email")

        await ref.set(updates, merge=True)
        data["last_login_at"] = now

    # 等級一律由 membership 解析（匿名 → guest、VIP 過期 → normal、舊值 "free" → normal）
    membership = await resolve_membership(claims)

    return ProfileResponse(
        uid=uid,
        email=data.get("email"),
        email_verified=bool(claims.get("email_verified", False)),
        display_name=data.get("display_name"),
        is_anonymous=(provider == "anonymous"),
        provider=provider,
        plan=membership.plan,
        plan_expires_at=membership.plan_expires_at,
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
async def get_my_sessions(
    limit: int = Query(50, ge=1, le=100),
    uid:   str = Depends(require_auth),
) -> SessionListResponse:
    """回傳目前登入者的對話 session 列表（依最後更新時間新到舊），附上每個對話的 token 用量。

    資料來源為 core/session_store.py 在每輪 /chat 結束後寫入的備份，
    見 api/routes/chat.py。用量來自 core/usage.py 的 usage_sessions 集合。
    """
    sessions  = await list_sessions(uid, limit=limit)
    usage_map = await get_sessions_usage(uid, [s["session_id"] for s in sessions])
    return SessionListResponse(sessions=[
        SessionSummary(
            **s,
            total_tokens=(usage_map.get(s["session_id"]) or {}).get("total_tokens", 0),
        )
        for s in sessions
    ])



@router.get("/me/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    uid:        str = Depends(require_auth),
) -> SessionDetailResponse:
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
        walk_session=detail.get("walk_session"),
        walk_pending_question=detail.get("walk_pending_question"),
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
        await delete_all_sessions(uid)        # kagrobot-state 資料庫（sessions 實際存放位置）
        await delete_user_usage(uid)          # usage_monthly / daily / sessions / meta
        await _delete_collection(             # 舊版本可能留在預設 DB 的 sessions；空集合刪除是 no-op
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
