# -*- coding: utf-8 -*-
"""core/admin_directory.py — 給 admin 介面用的 Firebase Auth 使用者目錄。

Firebase Auth 才是「有哪些使用者、Email 是什麼」的真相來源（Firestore 的 users/{uid}
只有在 GET /me 或授予等級時才會被建立，不完整）。

這一層全是同步呼叫（firebase_admin 沒有 async API），呼叫端一律用 asyncio.to_thread 包住。
獨立成模組的另一個原因：測試時只需要替換這一層，不必模擬 firebase_admin。

限制（Firebase 本身的限制，不是這裡偷懶）：
  - 沒有「Email 模糊搜尋」，只能用完整 Email 或 UID 精確查找。
  - list_users 依內部順序分頁，無法指定排序或篩選條件；因此「不含訪客」的篩選
    是在拿到頁面後才做，一頁可能只剩下少數人（見 api/routes/admin.py 的補頁迴圈）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from firebase_admin import auth as fb_auth

from core.firebase_auth import _ensure_initialized


@dataclass
class UserIdentity:
    uid: str
    email: Optional[str]
    display_name: Optional[str]
    is_anonymous: bool
    disabled: bool
    created_at: Optional[int]        # unix ms
    last_sign_in_at: Optional[int]   # unix ms
    custom_claims: Dict[str, Any] = field(default_factory=dict)


def _to_identity(rec: Any) -> UserIdentity:
    meta = getattr(rec, "user_metadata", None)
    return UserIdentity(
        uid=rec.uid,
        email=getattr(rec, "email", None),
        display_name=getattr(rec, "display_name", None),
        # 沒有任何登入方式綁定 = 匿名帳號（與 token 的 sign_in_provider="anonymous" 對應）
        is_anonymous=not getattr(rec, "provider_data", None),
        disabled=bool(getattr(rec, "disabled", False)),
        created_at=getattr(meta, "creation_timestamp", None) if meta else None,
        last_sign_in_at=getattr(meta, "last_sign_in_timestamp", None) if meta else None,
        custom_claims=dict(getattr(rec, "custom_claims", None) or {}),
    )


def get_user(uid: str) -> Optional[UserIdentity]:
    _ensure_initialized()
    try:
        return _to_identity(fb_auth.get_user(uid))
    except fb_auth.UserNotFoundError:
        return None


def find_user_by_email(email: str) -> Optional[UserIdentity]:
    _ensure_initialized()
    try:
        return _to_identity(fb_auth.get_user_by_email(email))
    except (fb_auth.UserNotFoundError, ValueError):
        return None


def get_users(uids: List[str]) -> Dict[str, UserIdentity]:
    """批次查詢（Firebase 單次上限 100 個 identifier）。查不到的 uid 不會出現在結果裡。"""
    _ensure_initialized()
    out: Dict[str, UserIdentity] = {}
    unique = list(dict.fromkeys(uids))
    for i in range(0, len(unique), 100):
        chunk = unique[i:i + 100]
        result = fb_auth.get_users([fb_auth.UidIdentifier(u) for u in chunk])
        for rec in result.users:
            out[rec.uid] = _to_identity(rec)
    return out


def list_users_page(page_token: Optional[str], max_results: int) -> Tuple[List[UserIdentity], Optional[str]]:
    _ensure_initialized()
    page = fb_auth.list_users(page_token=page_token, max_results=max(1, min(max_results, 1000)))
    return [_to_identity(u) for u in page.users], (page.next_page_token or None)
