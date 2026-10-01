# -*- coding: utf-8 -*-
"""core/admin_data.py — admin 介面用的 Firestore 讀取。

全站彙總（usage_global_*）由 core/usage.py 在每次 flush 時寫入，這裡只負責讀；
所以總覽是 O(1) 讀取，不會隨使用者數量變慢。

需要 Firestore 索引的只有一處：top_users()（跨所有使用者依用量排序）。
  collection group: usage_monthly   period ASC, total_tokens DESC
  建立方式見 backend/firestore.indexes.json 與 README。索引缺少時 Firestore 會丟
  FailedPrecondition，路由層把它轉成帶說明的 503，不會讓整個總覽壞掉。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.infra.firestore_db import get_default_db
from core.account.usage import (
    USAGE_GLOBAL_MONTHLY,
    USAGE_META,
    USAGE_MONTHLY,
    _as_int,
    normalize_usage,
)

_CHUNK = 300


async def _get_all(refs: List[Any]) -> List[Any]:
    db = get_default_db()
    snaps: List[Any] = []
    for i in range(0, len(refs), _CHUNK):
        async for snap in db.get_all(refs[i:i + _CHUNK]):
            snaps.append(snap)
    return snaps


async def get_plan_records(uids: List[str]) -> Dict[str, Tuple[Any, Optional[int]]]:
    """uid → (儲存的 plan 原始值, plan_expires_at)。沒有文件的 uid 回 (None, None)。"""
    db = get_default_db()
    snaps = await _get_all([db.collection("users").document(u) for u in uids])
    out: Dict[str, Tuple[Any, Optional[int]]] = {u: (None, None) for u in uids}
    for snap in snaps:
        if snap.exists:
            d = snap.to_dict() or {}
            exp = d.get("plan_expires_at")
            out[snap.id] = (d.get("plan"), int(exp) if exp else None)
    return out


async def get_month_tokens(uids: List[str], period_key: str) -> Dict[str, int]:
    db = get_default_db()
    snaps = await _get_all([
        db.collection("users").document(u).collection(USAGE_MONTHLY).document(period_key) for u in uids
    ])
    out = {u: 0 for u in uids}
    for snap in snaps:
        if snap.exists:
            out[snap.reference.parent.parent.id] = _as_int((snap.to_dict() or {}).get("total_tokens"))
    return out


async def get_lifetime_tokens(uids: List[str]) -> Dict[str, int]:
    db = get_default_db()
    snaps = await _get_all([
        db.collection("users").document(u).collection(USAGE_META).document("lifetime") for u in uids
    ])
    out = {u: 0 for u in uids}
    for snap in snaps:
        if snap.exists:
            out[snap.reference.parent.parent.id] = _as_int((snap.to_dict() or {}).get("total_tokens"))
    return out


async def get_global_month(period_key: str) -> Dict[str, Any]:
    """全站月度彙總（含 by_feature 與 by_plan）。該月沒有資料時回全 0。"""
    snap = await get_default_db().collection(USAGE_GLOBAL_MONTHLY).document(period_key).get()
    raw = (snap.to_dict() if snap.exists else None) or {}
    out = normalize_usage(raw)
    out["by_plan"] = {
        plan: {"total_tokens": _as_int((v or {}).get("total_tokens")), "turns": _as_int((v or {}).get("turns"))}
        for plan, v in (raw.get("by_plan") or {}).items()
    }
    return out


async def top_users(period_key: str, limit: int) -> List[Tuple[str, Dict[str, Any]]]:
    """該月用量最高的使用者。需要 usage_monthly 的 collection-group 複合索引（見檔頭）。"""
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    query = (
        get_default_db().collection_group(USAGE_MONTHLY)
        .where(filter=FieldFilter("period", "==", period_key))
        .order_by("total_tokens", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    out: List[Tuple[str, Dict[str, Any]]] = []
    async for snap in query.stream():
        out.append((snap.reference.parent.parent.id, normalize_usage(snap.to_dict())))
    return out


async def list_vip_records(limit: int = 200) -> List[Tuple[str, Optional[int]]]:
    """所有 plan == "vip" 的使用者（含已到期，是否到期由呼叫端用 plan_from_record 判斷）。
    單欄位等值查詢，使用 Firestore 自動建立的索引。"""
    from google.cloud.firestore_v1.base_query import FieldFilter

    query = get_default_db().collection("users").where(filter=FieldFilter("plan", "==", "vip")).limit(limit)
    out: List[Tuple[str, Optional[int]]] = []
    async for snap in query.stream():
        exp = (snap.to_dict() or {}).get("plan_expires_at")
        out.append((snap.id, int(exp) if exp else None))
    return out
