# -*- coding: utf-8 -*-
"""core/firebase_auth.py — Firebase Authentication 驗證層。

所有需要登入的 API 路由透過 FastAPI Depends(require_auth) 取得
已驗證的 uid，取代舊版由前端自行生成、完全未經驗證的 uid
（見 api/routes/user_brain.py 原本的設計）。

初始化策略：
  - lazy + 全域單例：第一次呼叫 require_auth / optional_auth 時才
    initialize_app()，避免測試環境或憑證尚未設定時，import 階段就噴錯。
  - 憑證來源交給 firebase_admin 預設行為：
      本地開發 → 環境變數 GOOGLE_APPLICATION_CREDENTIALS 指向
                 service account json
      Cloud Run / GCE 部署 → Application Default Credentials，
                 使用附掛的服務帳號，容器內不需要放任何金鑰檔案
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import firebase_admin
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as fb_auth

_init_lock = Lock()
_initialized = False

# auto_error=False：憑證缺失時交由 require_auth / optional_auth 自行決定
# 回傳 401 或視為未登入，而不是讓 HTTPBearer 直接丟出通用錯誤訊息。
_bearer_scheme = HTTPBearer(auto_error=False)


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app()
        _initialized = True


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency：驗證 Firebase ID token，回傳已驗證的 uid。

    前端需在每個受保護請求的 Authorization header 帶入：
        Authorization: Bearer <firebase_id_token>

    驗證失敗（token 缺失／過期／偽造／簽章不符）一律回傳 401，
    不透露詳細錯誤原因給用戶端，避免資訊洩漏。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="缺少登入憑證（Authorization header）")

    _ensure_initialized()

    try:
        decoded = fb_auth.verify_id_token(credentials.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="登入憑證無效或已過期")

    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="登入憑證缺少 uid")

    return uid


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[str]:
    """不強制登入，但若帶了合法 token 則回傳 uid，否則回傳 None。

    供尚未全面要求登入的路由過渡使用（例如舊版前端尚未更新完畢期間）。
    目前專案內尚未使用，保留供未來漸進式上線使用。
    """
    if credentials is None or not credentials.credentials:
        return None
    _ensure_initialized()
    try:
        decoded = fb_auth.verify_id_token(credentials.credentials)
        return decoded.get("uid")
    except Exception:
        return None
