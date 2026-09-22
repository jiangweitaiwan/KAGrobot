# -*- coding: utf-8 -*-
"""core/firestore_db.py — Firestore client 的唯一建立點。

【統一資料庫】所有 Firestore 資料（users/{uid} profile、brain、
usage_*、admin_audit、sessions）現在統一存在 kagrobot-state
（見 settings.firestore_state_project / firestore_state_database）。

get_default_db() / get_state_db() 現在回傳同一個 client；保留兩個
名稱只是避免一次改動所有呼叫端，語意上完全等價。新程式碼建議
一律呼叫 get_default_db()。

client 為 lazy 單例；AsyncClient 綁定建立當下的 event loop，
uvicorn 單一 loop 下沒有問題。
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from settings import get_settings

_lock: Lock = Lock()
_db: Optional[Any] = None


def _build_client():
    from google.cloud import firestore

    s = get_settings()
    kwargs: dict = {"database": s.firestore_state_database}
    if s.firestore_state_project:
        kwargs["project"] = s.firestore_state_project
    return firestore.AsyncClient(**kwargs)


def get_default_db():
    global _db
    if _db is None:
        with _lock:
            if _db is None:
                _db = _build_client()
    return _db


def get_state_db():
    return get_default_db()