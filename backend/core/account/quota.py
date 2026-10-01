# -*- coding: utf-8 -*-
"""core/quota.py — token 額度計算與 require_quota dependency。

用法（只掛在會消耗 LLM 的路由）：

    @router.post("/chat")
    async def chat_endpoint(..., ctx: QuotaContext = Depends(require_quota)):
        uid = ctx.uid
        async with usage_scope(uid, session_id):
            ...

檢查方式是「事前檢查、事後扣款」：呼叫前無法預知一輪會用多少 token，
所以只在「已用量 >= 上限」時擋下，最多會超額一輪（MainAgent 的 tool loop
可能較大，可接受）。慢想的單次成本由 PlanLimits.walk_max_depth 限制，
不在走到一半時強制中止（避免留下 walk_pending 的半截 checkpoint）。

超額回 HTTP 429，detail 為結構化物件（前端依 code 決定顯示什麼）：
    {
      "code": "guest_trial_exhausted" | "quota_exceeded",
      "message": "...給使用者看的說明...",
      "plan", "used", "limit", "period", "resets_at"(ms | null),
      "upgrade_hint": "register" | "vip" | null
    }

QUOTA_ENFORCE=false 時只計算、不擋（Phase 0）。管理者不受限。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException

from core.infra.firebase_auth import is_anonymous_claims, require_auth_claims
from core.account.membership import (
    PLAN_GUEST,
    PLAN_VIP,
    Membership,
    PlanLimits,
    resolve_membership,
)
from core.account.usage import (
    get_lifetime_usage,
    get_period_usage,
    month_key,
    next_month_start_ms,
    now_local,
    _tz,
)
from settings import get_settings


@dataclass
class QuotaContext:
    uid: str
    membership: Membership
    used: int
    limit: int
    period: str                       # "lifetime" | "monthly"
    period_key: Optional[str]         # "202609" | None
    resets_at_ms: Optional[int]       # 下次重置時間（訪客為 None）
    breakdown: Dict[str, Any] = field(default_factory=dict)
    enforced: bool = False

    @property
    def limits(self) -> PlanLimits:
        return self.membership.limits

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def ratio(self) -> float:
        return (self.used / self.limit) if self.limit > 0 else 1.0

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    @property
    def warn(self) -> bool:
        return self.ratio >= get_settings().quota_warn_ratio


async def compute_quota(claims: dict) -> QuotaContext:
    """讀取等級與用量，組成 QuotaContext（不做任何擋下）。"""
    uid = claims["uid"]
    now = now_local()

    if is_anonymous_claims(claims):
        usage_coro = get_lifetime_usage(uid)
        key: Optional[str] = None
        resets: Optional[int] = None
    else:
        key = month_key(now)
        usage_coro = get_period_usage(uid, key)
        resets = next_month_start_ms(now)

    membership, usage = await asyncio.gather(resolve_membership(claims), usage_coro)

    return QuotaContext(
        uid=uid,
        membership=membership,
        used=usage["total_tokens"],
        limit=membership.limits.token_limit,
        period=membership.limits.period,
        period_key=key,
        resets_at_ms=resets,
        breakdown=usage,
        enforced=get_settings().quota_enforce,
    )


def _fmt_reset(ms: Optional[int]) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=_tz()).strftime("%m 月 %d 日")


def quota_error_detail(ctx: QuotaContext) -> Dict[str, Any]:
    plan = ctx.membership.plan
    if plan == PLAN_GUEST:
        code = "guest_trial_exhausted"
        message = "訪客試用額度已用完。註冊帳號即可繼續使用，並取得每月額度，目前的對話與 Personal Brain 都會保留。"
        hint: Optional[str] = "register"
    else:
        code = "quota_exceeded"
        reset = _fmt_reset(ctx.resets_at_ms)
        message = f"本月 token 額度已用完，額度將於 {reset} 重置。" if reset else "本月 token 額度已用完。"
        hint = None if plan == PLAN_VIP else "vip"
        if hint == "vip":
            message += "升級 VIP 可取得更高額度。"
    return {
        "code": code,
        "message": message,
        "plan": plan,
        "used": ctx.used,
        "limit": ctx.limit,
        "period": ctx.period,
        "resets_at": ctx.resets_at_ms,
        "upgrade_hint": hint,
    }


async def require_quota(claims: dict = Depends(require_auth_claims)) -> QuotaContext:
    ctx = await compute_quota(claims)

    if ctx.enforced and ctx.exceeded and not ctx.membership.is_admin:
        headers = {}
        if ctx.resets_at_ms:
            import time

            headers["Retry-After"] = str(max(1, int(ctx.resets_at_ms / 1000 - time.time())))
        raise HTTPException(status_code=429, detail=quota_error_detail(ctx), headers=headers)

    return ctx
