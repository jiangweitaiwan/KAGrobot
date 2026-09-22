# -*- coding: utf-8 -*-
"""core/admin_audit.py — 管理者操作的稽核紀錄。

    admin_audit/{auto-id} → {
        admin_uid, admin_email, action, target_uid, before, after, at (unix ms)
    }

只記「會改變使用者狀態」的操作（目前是 plan.set）。順序取捨：先執行變更、再寫紀錄——
若寫紀錄失敗只 log、不讓已完成的變更回報成失敗（否則管理者會誤以為沒生效而重複操作）。
代價是極端情況下可能有一筆變更沒有紀錄，log 裡會有 error 可追。

target_uid 查詢刻意不加 order_by（避免需要複合索引），改為取回後在記憶體排序。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.firestore_db import get_default_db

log = logging.getLogger("admin_audit")

COLLECTION = "admin_audit"


async def record(
    admin_claims: Dict[str, Any],
    action: str,
    target_uid: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> None:
    try:
        await get_default_db().collection(COLLECTION).document().set({
            "admin_uid":   admin_claims.get("uid"),
            "admin_email": admin_claims.get("email"),
            "action":      action,
            "target_uid":  target_uid,
            "before":      before,
            "after":       after,
            "at":          int(time.time() * 1000),
        })
    except Exception as exc:
        log.error("audit write failed (action=%s target=%s admin=%s): %s",
                  action, target_uid, admin_claims.get("uid"), exc)


async def list_entries(limit: int = 50, target_uid: Optional[str] = None) -> List[Dict[str, Any]]:
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    col = get_default_db().collection(COLLECTION)
    out: List[Dict[str, Any]] = []
    if target_uid:
        async for snap in col.where(filter=FieldFilter("target_uid", "==", target_uid)).limit(200).stream():
            out.append({"id": snap.id, **(snap.to_dict() or {})})
        out.sort(key=lambda e: e.get("at", 0), reverse=True)
        return out[:limit]

    async for snap in col.order_by("at", direction=firestore.Query.DESCENDING).limit(limit).stream():
        out.append({"id": snap.id, **(snap.to_dict() or {})})
    return out
