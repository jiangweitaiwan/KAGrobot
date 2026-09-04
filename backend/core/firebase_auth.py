# -*- coding: utf-8 -*-
"""core/firebase_auth.py — Firebase Authentication 驗證層。

所有需要登入的 API 路由透過 FastAPI Depends(require_auth) 取得
已驗證的 uid，取代舊版由前端自行生成、完全未經驗證的 uid
（見 api/routes/user_brain.py 原本的設計）。

初始化策略：
  - lazy + 全域單例：第一次呼叫 require_auth 時才 initialize_app()，
    避免測試環境或憑證尚未設定時，import 階段就噴錯。
  - 憑證來源交給 firebase_admin 預設行為：
      本地開發 → 環境變數 GOOGLE_APPLICATION_CREDENTIALS 指向
                 service account json
      Cloud Run / GCE 部署 → Application Default Credentials，
                 使用附掛的服務帳號，容器內不需要放任何金鑰檔案

【會員系統擴充】
  require_auth_claims：與 require_auth 相同的驗證邏輯，但回傳完整
  decoded token（uid / email / email_verified / firebase.sign_in_provider
  等），供需要判斷「使用者是不是匿名帳號」「用哪個 provider 登入」的
  路由使用（例如 GET /me）。刻意獨立一個 dependency，而不是把 require_auth
  的回傳型別從 str 改成 dict——避免動到現有一大票只需要 uid 字串的路由。

  匿名使用者（signInAnonymously）與一般使用者共用同一套 verify_id_token
  驗證路徑：Firebase 對匿名使用者一樣會核發合法 ID token，
  decoded["firebase"]["sign_in_provider"] == "anonymous"。
  也因此，前端「先匿名使用、之後升級為正式帳號」的流程完全不需要
  後端另外處理——升級（linkWithCredential / linkWithPopup）不會改變
  uid，所有既有以 uid 為 key 的資料（brain / sessions）自動延續。
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

# auto_error=False：憑證缺失時交由 require_auth 自行決定
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


async def require_auth_claims(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """與 require_auth 相同的驗證，但回傳完整 decoded token dict。

    供需要 email / email_verified / sign_in_provider 等額外資訊的路由
    使用（目前為 GET/PATCH /me）。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="缺少登入憑證（Authorization header）")

    _ensure_initialized()

    try:
        decoded = fb_auth.verify_id_token(credentials.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="登入憑證無效或已過期")

    if not decoded.get("uid"):
        raise HTTPException(status_code=401, detail="登入憑證缺少 uid")

    return decoded
