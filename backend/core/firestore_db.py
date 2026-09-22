# -*- coding: utf-8 -*-
"""core/firestore_db.py — Firestore client 的唯一建立點。

背景（Phase 0 清理）：
  原本三個地方各自建 client，而且指到不同的資料庫：
    - session_store.py         → AsyncClient(project=..., database="kagrobot-state")
    - users.py / user_brain.py → AsyncClient()            （預設資料庫）
    - app.py::_load_brain_weights → 每次呼叫都新建一個 AsyncClient
  結果是 profile / brain 在預設 DB、sessions 在 kagrobot-state，
  DELETE /me 用預設 DB 去刪 sessions 會刪不到。

現在統一從這裡取：
    get_default_db() → users/{uid}、brain、usage_*（跟著使用者本人的資料）
    get_state_db()   → users/{uid}/sessions/{session_id}（對話狀態備份）

【資料庫位置不變】既有資料不需要搬遷：sessions 仍在 kagrobot-state，
其餘仍在預設 DB。改變的只是「誰負責建 client」，以及刪帳號時兩邊都要清。

client 為 lazy 單例（不要在 import 階段建立）。AsyncClient 綁定建立當下的
event loop，uvicorn 單一 loop 下沒有問題。
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from settings import get_settings

_lock = Lock()
_default_db: Optional[Any] = None
_state_db: Optional[Any] = None


def get_default_db():
    global _default_db
    if _default_db is None:
        with _lock:
            if _default_db is None:
                from google.cloud import firestore

                s = get_settings()
                kwargs = {"project": s.firestore_project} if s.firestore_project else {}
                _default_db = firestore.AsyncClient(**kwargs)
    return _default_db


def get_state_db():
    global _state_db
    if _state_db is None:
        with _lock:
            if _state_db is None:
                from google.cloud import firestore

                s = get_settings()
                kwargs = {"database": s.firestore_state_database}
                if s.firestore_state_project:
                    kwargs["project"] = s.firestore_state_project
                _state_db = firestore.AsyncClient(**kwargs)
    return _state_db
