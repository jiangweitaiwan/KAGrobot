# -*- coding: utf-8 -*-
"""core/membership.py — 會員等級與各等級權限。

三種身分：
  guest  訪客   Firebase 匿名登入。一次性試用額度（終身累計，不重置），不能用慢想。
  normal 普通   已註冊帳號的預設等級。月額度（時區見 settings.usage_timezone）。
  vip    VIP    月額度較高、慢想較深、Personal Brain 上限較高。

儲存：
  users/{uid}.plan            "normal" | "vip"   （舊資料的 "free" 視為 normal）
  users/{uid}.plan_expires_at int (unix ms) 或 None（None = 不過期）
  訪客不需要儲存——由 token 的 sign_in_provider 判斷。

「有效等級」在讀取時計算：VIP 過期就視為 normal，不需要排程降級。
Firestore 是唯一真相來源；為了不讓每個請求都多讀一次，加一個
per-process 的短 TTL cache（settings.plan_cache_ttl_s，預設 60 秒）。
多個 Cloud Run instance 之間最多有 TTL 長度的不一致，可接受；
admin 改等級時會清本機 cache。

QUOTA_ENFORCE=false（預設，Phase 0 只計量）時：
  effective_limits() 會放寬「功能限制」（慢想、Brain 上限），
  同時 core/quota.py 不會回 429。plan_table() 仍回報「名義上的」限額，
  讓前端可以顯示未來的方案內容。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.infra.firebase_auth import is_admin_claims, is_anonymous_claims
from core.infra.firestore_db import get_default_db
from settings import get_settings

PLAN_GUEST = "guest"
PLAN_NORMAL = "normal"
PLAN_VIP = "vip"
ASSIGNABLE_PLANS = (PLAN_NORMAL, PLAN_VIP)

_RELAXED_BRAIN_MAX = 10_000


@dataclass(frozen=True)
class PlanLimits:
    plan: str
    period: str              # "lifetime"（訪客試用）| "monthly"
    token_limit: int
    walk_allowed: bool
    walk_max_depth: int      # 單條慢想路徑最多走幾步；也就是單次慢想的成本上限
    brain_max_concepts: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Membership:
    uid: str
    plan: str
    plan_expires_at: Optional[int]
    is_anonymous: bool
    is_admin: bool
    limits: PlanLimits


def normalize_plan(raw: Any) -> str:
    return PLAN_VIP if raw == PLAN_VIP else PLAN_NORMAL   # 舊值 "free"、缺值 → normal


def plan_from_record(
    raw_plan: Any,
    expires_ms: Optional[int],
    *,
    anonymous: bool = False,
    now_ms: Optional[int] = None,
) -> Tuple[str, Optional[int]]:
    """由「儲存的欄位」算出有效等級與（有效的）到期日。純函式，無 I/O。

    resolve_membership（額度檢查）與 admin 列表共用這一份，確保管理者看到的等級
    與系統實際套用的等級永遠一致。
    """
    if anonymous:
        return PLAN_GUEST, None
    plan = normalize_plan(raw_plan)
    if plan == PLAN_VIP and expires_ms is not None and expires_ms <= (now_ms or int(time.time() * 1000)):
        plan = PLAN_NORMAL   # VIP 已到期：讀取時降級，不需排程
    return plan, (expires_ms if plan == PLAN_VIP else None)


def nominal_limits(plan: str) -> PlanLimits:
    """該等級「名義上」的限額（讀 settings，可用環境變數調整）。"""
    s = get_settings()
    if plan == PLAN_GUEST:
        return PlanLimits(PLAN_GUEST, "lifetime", s.quota_guest_tokens, False, 0, s.brain_max_guest)
    if plan == PLAN_VIP:
        return PlanLimits(
            PLAN_VIP, "monthly", s.quota_vip_monthly_tokens, True, s.walk_max_depth_vip, s.brain_max_vip
        )
    return PlanLimits(
        PLAN_NORMAL, "monthly", s.quota_normal_monthly_tokens, True, s.walk_max_depth_normal, s.brain_max_normal
    )


def effective_limits(plan: str) -> PlanLimits:
    """實際執行用的限額。未啟用限額（QUOTA_ENFORCE=false）時放寬功能限制，只保留計量。"""
    lim = nominal_limits(plan)
    if get_settings().quota_enforce:
        return lim
    return PlanLimits(
        plan=lim.plan,
        period=lim.period,
        token_limit=lim.token_limit,
        walk_allowed=True,
        walk_max_depth=get_settings().walk_max_depth_vip,
        brain_max_concepts=_RELAXED_BRAIN_MAX,
    )


def plan_table() -> List[PlanLimits]:
    return [nominal_limits(p) for p in (PLAN_GUEST, PLAN_NORMAL, PLAN_VIP)]


# ── 讀取（含 TTL cache）──────────────────────────────────────

_plan_cache: Dict[str, Tuple[float, str, Optional[int]]] = {}


def invalidate_membership_cache(uid: Optional[str] = None) -> None:
    if uid is None:
        _plan_cache.clear()
    else:
        _plan_cache.pop(uid, None)


async def _load_plan_record(uid: str) -> Tuple[str, Optional[int]]:
    ttl = get_settings().plan_cache_ttl_s
    hit = _plan_cache.get(uid)
    if hit and hit[0] > time.time():
        return hit[1], hit[2]

    snap = await get_default_db().collection("users").document(uid).get()
    data = snap.to_dict() if snap.exists else {}
    plan = normalize_plan((data or {}).get("plan"))
    expires = (data or {}).get("plan_expires_at")
    expires = int(expires) if expires else None

    _plan_cache[uid] = (time.time() + ttl, plan, expires)
    return plan, expires


async def resolve_membership(claims: dict) -> Membership:
    uid = claims["uid"]
    admin = is_admin_claims(claims)

    if is_anonymous_claims(claims):
        return Membership(uid, PLAN_GUEST, None, True, admin, effective_limits(PLAN_GUEST))

    raw_plan, raw_expires = await _load_plan_record(uid)
    plan, expires = plan_from_record(raw_plan, raw_expires)
    return Membership(uid, plan, expires, False, admin, effective_limits(plan))


# ── 寫入（admin 用）──────────────────────────────────────────

async def set_user_plan(uid: str, plan: str, expires_at_ms: Optional[int] = None) -> None:
    if plan not in ASSIGNABLE_PLANS:
        raise ValueError(f"plan 必須是 {ASSIGNABLE_PLANS} 之一")
    payload: Dict[str, Any] = {
        "plan": plan,
        "plan_expires_at": expires_at_ms if plan == PLAN_VIP else None,
        "plan_updated_at": int(time.time() * 1000),
    }
    await get_default_db().collection("users").document(uid).set(payload, merge=True)
    invalidate_membership_cache(uid)
