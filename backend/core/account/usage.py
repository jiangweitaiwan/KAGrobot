# -*- coding: utf-8 -*-
"""core/usage.py — LLM token 計量。

流程：
  1. 路由（/chat、/walk/*、/concept/suggest）用 `async with usage_scope(uid, session_id)`
     包住整輪執行；進入時建立一個 UsageAccumulator 放進 ContextVar。
  2. 所有 LLM 都由 core/google_llm.py 建立，並掛上 UsageCallbackHandler(feature)。
     每次 LLM 呼叫結束（on_llm_end）就把 usage_metadata 累加進「目前 scope 的
     accumulator」。handler 本身無狀態，因此 LLM 實例可以跨請求共用。
  3. 離開 scope 時（不論正常結束、例外、或用戶端中途斷線）一次性寫入 Firestore：
     用 Increment 做原子累加，一個 batch 同時寫四份文件：

       users/{uid}/usage_monthly/{YYYYMM}        月度（額度檢查用）
       users/{uid}/usage_daily/{YYYYMMDD}        每日（圖表用）
       users/{uid}/usage_sessions/{session_id}   單一對話（歷史對話顯示用）
       users/{uid}/usage_meta/lifetime           終身累計（訪客一次性額度用）

     每輪只有一次寫入，中斷時已消耗的 token 也不會漏記。

為什麼用 ContextVar + 可變 accumulator：
  - asyncio.to_thread / LangGraph 建立子 task 時會「複製」context，但複製的只是
    指向同一個 accumulator 物件的參考，所以在 thread 內同步呼叫 LLM
    （例如 MainAgent.chat）也累加得到同一份。
  - accumulator 用 threading.Lock 保護，callback 在 thread 或 event loop 皆安全。
  - handler 設 run_inline=True，讓 LangChain 直接在呼叫端 context 執行 callback，
    不經 executor 轉手。

【Phase 3 — 全站彙總（給 admin 總覽用）】
  同一次 flush 另外用「獨立的第二個 batch」寫兩份全站彙總文件：

       usage_global_monthly/{YYYYMM}     本月全站總量 + by_feature + by_plan
       usage_global_daily/{YYYYMMDD}     當日全站總量（趨勢圖用）

  為什麼獨立 batch：這兩份是「所有使用者共寫」的熱點文件（Firestore 單一文件
  持續寫入約 1 次/秒的上限），若與使用者本人的記帳放同一個 batch，熱點爭用
  失敗會連帶讓使用者自己的用量沒被記到。獨立之後，全站彙總失敗只會讓總覽
  略少一筆，使用者本人的額度記帳不受影響。
  by_plan 用 usage_scope(plan=...) 傳入的「當下等級」歸類。

時區：月／日的切分一律用 settings.usage_timezone（預設 Asia/Taipei）。

Firestore 欄位（每份文件皆同）：
  total_tokens / input_tokens / output_tokens / reasoning_tokens / calls / turns
  by_feature.{socratic|walk|main_agent|suggest|other}.{同上，不含 turns}
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from core.infra.firestore_db import get_default_db
from settings import get_settings

log = logging.getLogger("usage")

_TOKEN_KEYS = ("total_tokens", "input_tokens", "output_tokens", "reasoning_tokens", "calls")

USAGE_MONTHLY = "usage_monthly"
USAGE_DAILY = "usage_daily"
USAGE_SESSIONS = "usage_sessions"
USAGE_META = "usage_meta"
_ALL_USAGE_COLLECTIONS = (USAGE_MONTHLY, USAGE_DAILY, USAGE_SESSIONS, USAGE_META)

# 全站彙總（頂層集合，不屬於任何使用者；刪帳號時不會、也不應該動它們）
USAGE_GLOBAL_MONTHLY = "usage_global_monthly"
USAGE_GLOBAL_DAILY = "usage_global_daily"


# ══════════════════════════════════════════════════════════════
# 時間 helper
# ══════════════════════════════════════════════════════════════

def _tz():
    name = get_settings().usage_timezone
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        # 容器缺 tzdata 時的保底（預設時區 Asia/Taipei = UTC+8，無夏令時間）。
        # 部署請在 requirements.txt 加上 tzdata。
        return timezone(timedelta(hours=8))


def now_local() -> datetime:
    return datetime.now(_tz())


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y%m")


def day_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def next_month_start_ms(dt: datetime) -> int:
    y, m = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
    start = dt.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


# ══════════════════════════════════════════════════════════════
# Accumulator + ContextVar
# ══════════════════════════════════════════════════════════════

class UsageAccumulator:
    """單一 scope（一輪對話 / 一段 SSE）內累加的 token 用量。"""

    def __init__(self, uid: str, session_id: Optional[str] = None, plan: Optional[str] = None) -> None:
        self.uid = uid
        self.session_id = session_id
        self.plan = plan            # 僅供全站彙總的 by_plan 歸類，不影響使用者本人的記帳
        self._lock = threading.Lock()
        self._by_feature: Dict[str, Dict[str, int]] = {}

    def add(
        self,
        feature: str,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        reasoning_tokens: int = 0,
    ) -> None:
        with self._lock:
            b = self._by_feature.setdefault(feature, dict.fromkeys(_TOKEN_KEYS, 0))
            b["input_tokens"] += input_tokens
            b["output_tokens"] += output_tokens
            b["reasoning_tokens"] += reasoning_tokens
            b["total_tokens"] += total_tokens
            b["calls"] += 1

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {f: dict(c) for f, c in self._by_feature.items()}

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(c["total_tokens"] for c in self._by_feature.values())

    @property
    def calls(self) -> int:
        with self._lock:
            return sum(c["calls"] for c in self._by_feature.values())


_current: contextvars.ContextVar[Optional[UsageAccumulator]] = contextvars.ContextVar(
    "usage_accumulator", default=None
)


def current_usage() -> Optional[UsageAccumulator]:
    return _current.get()


# ══════════════════════════════════════════════════════════════
# LangChain callback
# ══════════════════════════════════════════════════════════════

def _as_int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def extract_usage(response: Any) -> Dict[str, int]:
    """從 LLMResult 取出 token 用量。

    優先讀 generation.message.usage_metadata（langchain-core 標準欄位：
    input_tokens / output_tokens / total_tokens / output_token_details.reasoning）。
    total_tokens 優先採用供應商回報值（Gemini 的 total 可能包含 tool-use / cached
    等 input+output 之外的項目，以它為帳務依據最貼近實際計費）；缺值才用 input+output。
    找不到時退回 llm_output 的 token_usage / usage_metadata。
    """
    i = o = t = r = 0
    found = False

    for gens in getattr(response, "generations", None) or []:
        for g in gens:
            msg = getattr(g, "message", None)
            um = getattr(msg, "usage_metadata", None)
            if not um:
                continue
            found = True
            gi, go = _as_int(um.get("input_tokens")), _as_int(um.get("output_tokens"))
            i += gi
            o += go
            t += _as_int(um.get("total_tokens")) or (gi + go)
            r += _as_int((um.get("output_token_details") or {}).get("reasoning"))

    if not found:
        lo = getattr(response, "llm_output", None) or {}
        um = lo.get("token_usage") or lo.get("usage_metadata") or {}
        if um:
            i = _as_int(um.get("input_tokens") or um.get("prompt_tokens") or um.get("prompt_token_count"))
            o = _as_int(um.get("output_tokens") or um.get("completion_tokens") or um.get("candidates_token_count"))
            t = _as_int(um.get("total_tokens") or um.get("total_token_count")) or (i + o)

    return {"input": i, "output": o, "total": t, "reasoning": r}


class UsageCallbackHandler(BaseCallbackHandler):
    """把每次 LLM 呼叫的用量累加進目前 scope 的 accumulator。

    無狀態（只記 feature 名稱），可安全掛在跨請求共用的 LLM 實例上。
    """

    run_inline = True  # 直接在呼叫端 context 執行，確保讀得到 ContextVar

    def __init__(self, feature: str = "other") -> None:
        self.feature = feature

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        acc = _current.get()
        if acc is None:
            return  # 不在任何 usage_scope 內（啟動流程、腳本等），不計量
        u = extract_usage(response)
        if u["total"] == 0:
            # Phase 0 驗證重點：供應商沒回 usage_metadata 時這裡會出現，
            # 代表該 feature 的計量是漏的，額度數字不可信。
            log.warning("LLM response without usage_metadata (feature=%s)", self.feature)
        acc.add(
            self.feature,
            input_tokens=u["input"],
            output_tokens=u["output"],
            total_tokens=u["total"],
            reasoning_tokens=u["reasoning"],
        )


# ══════════════════════════════════════════════════════════════
# 寫入（scope 結束時）
# ══════════════════════════════════════════════════════════════

_pending_flushes: set = set()


def build_increment_body(snapshot: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """把 accumulator 快照轉成 Firestore 的 Increment 文件片段（純函式，便於測試）。"""
    from google.cloud.firestore import Increment

    totals = dict.fromkeys(_TOKEN_KEYS, 0)
    for c in snapshot.values():
        for k in _TOKEN_KEYS:
            totals[k] += c.get(k, 0)

    def inc(counts: Dict[str, int]) -> Dict[str, Any]:
        return {k: Increment(counts.get(k, 0)) for k in _TOKEN_KEYS}

    return {
        **inc(totals),
        "turns": Increment(1),
        "by_feature": {f: inc(c) for f, c in snapshot.items()},
    }


def _totals_of(snapshot: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    totals = dict.fromkeys(_TOKEN_KEYS, 0)
    for c in snapshot.values():
        for k in _TOKEN_KEYS:
            totals[k] += c.get(k, 0)
    return totals


async def _flush_user(acc: UsageAccumulator, snapshot: Dict[str, Dict[str, int]], now: datetime) -> None:
    from google.cloud import firestore

    db = get_default_db()
    user = db.collection("users").document(acc.uid)
    body = build_increment_body(snapshot)
    stamp = {"updated_at": int(now.timestamp() * 1000)}

    batch = db.batch()
    batch.set(
        user.collection(USAGE_MONTHLY).document(month_key(now)),
        {**body, "period": month_key(now), **stamp},
        merge=True,
    )
    batch.set(
        user.collection(USAGE_DAILY).document(day_key(now)),
        {**body, "date": day_key(now), **stamp},
        merge=True,
    )
    if acc.session_id:
        batch.set(
            user.collection(USAGE_SESSIONS).document(acc.session_id),
            {**body, **stamp},
            merge=True,
        )
    totals = {k: firestore.Increment(v) for k, v in _totals_of(snapshot).items()}
    batch.set(user.collection(USAGE_META).document("lifetime"), {**totals, **stamp}, merge=True)
    await batch.commit()


async def _flush_global(acc: UsageAccumulator, snapshot: Dict[str, Dict[str, int]], now: datetime) -> None:
    from google.cloud.firestore import Increment

    db = get_default_db()
    t = _totals_of(snapshot)
    body = build_increment_body(snapshot)
    plan = acc.plan or "unknown"
    body["by_plan"] = {plan: {"total_tokens": Increment(t["total_tokens"]), "turns": Increment(1)}}
    stamp = {"updated_at": int(now.timestamp() * 1000)}

    batch = db.batch()
    batch.set(
        db.collection(USAGE_GLOBAL_MONTHLY).document(month_key(now)),
        {**body, "period": month_key(now), **stamp},
        merge=True,
    )
    batch.set(
        db.collection(USAGE_GLOBAL_DAILY).document(day_key(now)),
        {**body, "date": day_key(now), **stamp},
        merge=True,
    )
    await batch.commit()


async def _flush(acc: UsageAccumulator) -> None:
    snapshot = acc.snapshot()
    if not snapshot or not acc.uid:
        return
    now = now_local()

    # 使用者本人的記帳（額度依據）：最重要，先做，且獨立於全站彙總。
    try:
        await _flush_user(acc, snapshot, now)
    except Exception as exc:  # 計量失敗絕不能讓使用者的對話失敗
        log.error("usage flush failed (uid=%s, session=%s): %s", acc.uid, acc.session_id, exc)

    # 全站彙總（admin 總覽用）：熱點文件，失敗只影響總覽，不影響上面的記帳。
    try:
        await _flush_global(acc, snapshot, now)
    except Exception as exc:
        log.warning("global usage flush failed (uid=%s): %s", acc.uid, exc)


@asynccontextmanager
async def usage_scope(
    uid: str, session_id: Optional[str] = None, plan: Optional[str] = None
) -> AsyncIterator[UsageAccumulator]:
    """包住一輪 LLM 工作；結束時寫入用量。可巢狀，內層會遮蔽外層（不會重複計）。

    plan：呼叫端當下解析出的會員等級（ctx.membership.plan），只用於全站彙總的 by_plan 分解。
    """
    acc = UsageAccumulator(uid, session_id, plan)
    token = _current.set(acc)
    try:
        yield acc
    finally:
        try:
            _current.reset(token)
        except ValueError:
            # async generator（SSE）在不同 context 收尾時 reset 會失敗，
            # 此時直接清掉即可，accumulator 本身已經拿在手上。
            _current.set(None)
        # 用 task + shield：用戶端斷線造成 CancelledError 時，寫入仍會完成。
        task = asyncio.ensure_future(_flush(acc))
        _pending_flushes.add(task)
        task.add_done_callback(_pending_flushes.discard)
        await asyncio.shield(task)


# ══════════════════════════════════════════════════════════════
# 讀取
# ══════════════════════════════════════════════════════════════

def normalize_usage(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = d or {}
    out: Dict[str, Any] = {k: _as_int(d.get(k)) for k in _TOKEN_KEYS}
    out["turns"] = _as_int(d.get("turns"))
    out["by_feature"] = {
        name: {k: _as_int((v or {}).get(k)) for k in _TOKEN_KEYS}
        for name, v in (d.get("by_feature") or {}).items()
    }
    return out


def _user_ref(uid: str):
    return get_default_db().collection("users").document(uid)


async def get_period_usage(uid: str, period_key: str) -> Dict[str, Any]:
    snap = await _user_ref(uid).collection(USAGE_MONTHLY).document(period_key).get()
    return normalize_usage(snap.to_dict() if snap.exists else None)


async def get_lifetime_usage(uid: str) -> Dict[str, Any]:
    snap = await _user_ref(uid).collection(USAGE_META).document("lifetime").get()
    return normalize_usage(snap.to_dict() if snap.exists else None)


async def _daily_series(collection_ref: Any, days: int) -> List[Dict[str, Any]]:
    """從一個「文件含 date 欄位（YYYYMMDD）」的集合讀最近 days 天，缺的日子補 0，由舊到新。
    使用者的 usage_daily 與全站的 usage_global_daily 共用。"""
    from google.cloud.firestore_v1.base_query import FieldFilter

    days = max(1, min(int(days), 92))
    today = now_local()
    start = today - timedelta(days=days - 1)

    query = collection_ref.where(filter=FieldFilter("date", ">=", day_key(start)))
    found: Dict[str, Dict[str, Any]] = {}
    async for snap in query.stream():
        d = snap.to_dict() or {}
        found[d.get("date") or snap.id] = normalize_usage(d)

    result: List[Dict[str, Any]] = []
    for n in range(days):
        day = start + timedelta(days=n)
        u = found.get(day_key(day)) or normalize_usage(None)
        result.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "total_tokens": u["total_tokens"],
                "input_tokens": u["input_tokens"],
                "output_tokens": u["output_tokens"],
                "reasoning_tokens": u["reasoning_tokens"],
                "calls": u["calls"],
                "turns": u["turns"],
            }
        )
    return result


async def get_daily_usage(uid: str, days: int) -> List[Dict[str, Any]]:
    """最近 days 天（含今天）的每日用量，缺的日子補 0，由舊到新。"""
    return await _daily_series(_user_ref(uid).collection(USAGE_DAILY), days)


async def get_global_daily_usage(days: int) -> List[Dict[str, Any]]:
    """全站每日用量（admin 總覽趨勢圖）。"""
    return await _daily_series(get_default_db().collection(USAGE_GLOBAL_DAILY), days)


async def get_sessions_usage(uid: str, session_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not session_ids:
        return {}
    col = _user_ref(uid).collection(USAGE_SESSIONS)
    out: Dict[str, Dict[str, Any]] = {}
    async for snap in get_default_db().get_all([col.document(s) for s in session_ids]):
        if snap.exists:
            out[snap.id] = normalize_usage(snap.to_dict())
    return out


async def delete_user_usage(uid: str) -> None:
    """刪帳號時清掉所有 usage 集合。"""
    user = _user_ref(uid)
    for name in _ALL_USAGE_COLLECTIONS:
        while True:
            count = 0
            async for snap in user.collection(name).limit(100).stream():
                await snap.reference.delete()
                count += 1
            if count < 100:
                break
