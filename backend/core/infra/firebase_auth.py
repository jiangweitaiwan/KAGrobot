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

【新增 — 會員等級 / 管理者】
  本檔案仍然只回答「你是誰」。「你能做什麼、用了多少」在
  core/membership.py 與 core/quota.py。

  is_anonymous_claims : token 是否為匿名登入（= 訪客等級）。
  is_admin_claims     : 管理者判斷，兩種來源任一成立即可——
      (a) Firebase custom claim  admin=true
          （用 firebase_admin.auth.set_custom_user_claims(uid, {"admin": True}) 設定，
            使用者需重新取得 token 後生效）
      (b) 環境變數 ADMIN_UIDS（逗號分隔 uid）——用來 bootstrap 第一位管理者，
          不需要等 token 刷新。
  require_admin       : 非管理者一律 403。
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


def _verify(credentials: Optional[HTTPAuthorizationCredentials]) -> dict:
    """驗證 Bearer token，回傳 decoded token；失敗一律 401（不透露原因）。"""
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


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency：驗證 Firebase ID token，回傳已驗證的 uid。

    前端需在每個受保護請求的 Authorization header 帶入：
        Authorization: Bearer <firebase_id_token>

    驗證失敗（token 缺失／過期／偽造／簽章不符）一律回傳 401，
    不透露詳細錯誤原因給用戶端，避免資訊洩漏。
    """
    return _verify(credentials)["uid"]


async def require_auth_claims(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """與 require_auth 相同的驗證，但回傳完整 decoded token dict。

    供需要 email / email_verified / sign_in_provider 等額外資訊的路由
    使用（GET/PATCH /me、額度檢查、Personal Brain 上限檢查等）。
    """
    return _verify(credentials)


# ── 身分屬性判斷（純函式，不做 I/O）────────────────────────────

def is_anonymous_claims(claims: dict) -> bool:
    return ((claims.get("firebase") or {}).get("sign_in_provider")) == "anonymous"


def _admin_uids() -> set:
    from settings import get_settings

    raw = get_settings().admin_uids or ""
    return {u.strip() for u in raw.split(",") if u.strip()}


def is_admin_claims(claims: dict) -> bool:
    if claims.get("admin") is True:
        return True
    return claims.get("uid") in _admin_uids()


async def require_admin(claims: dict = Depends(require_auth_claims)) -> dict:
    if not is_admin_claims(claims):
        raise HTTPException(status_code=403, detail="需要管理者權限")
    return claims
