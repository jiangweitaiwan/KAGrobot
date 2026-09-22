# -*- coding: utf-8 -*-
"""api/routes/account.py — 會員中心用的 endpoints。

GET    /me/usage                    → 等級、額度、本期用量與功能分解、各等級方案表
GET    /me/usage/daily?days=30      → 每日用量（缺的日子補 0）
PATCH  /me/sessions/{session_id}    → 重新命名歷史對話
DELETE /me/sessions/{session_id}    → 刪除歷史對話

（GET /me、PATCH /me、GET /me/sessions、GET /me/sessions/{id}、DELETE /me
  仍在 api/routes/users.py。）

刪除對話只刪對話內容（Firestore sessions 文件與記憶體 checkpoint），
不動 usage_* 累計——已經花掉的 token 不會因為刪除對話而「退還」。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.firebase_auth import require_auth, require_auth_claims
from core.membership import plan_table
from core.quota import QuotaContext, compute_quota
from core.session_store import delete_session, rename_session
from core.usage import get_daily_usage

router = APIRouter()

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# ── Schema ────────────────────────────────────────────────────────────

class UsageBucket(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0


class UsageBreakdown(UsageBucket):
    turns: int = 0
    by_feature: Dict[str, UsageBucket] = {}


class PlanLimitsModel(BaseModel):
    plan: str
    period: str
    token_limit: int
    walk_allowed: bool
    walk_max_depth: int
    brain_max_concepts: int


class MyUsageResponse(BaseModel):
    plan: str                          # guest | normal | vip
    plan_expires_at: Optional[int] = None
    is_anonymous: bool
    is_admin: bool = False
    enforced: bool                     # False = 目前只計量、尚未限額（Phase 0）
    period: str                        # lifetime | monthly
    period_key: Optional[str] = None
    resets_at: Optional[int] = None
    used: int
    limit: int
    remaining: int
    ratio: float
    warn: bool
    exceeded: bool
    breakdown: UsageBreakdown
    limits: PlanLimitsModel            # 目前實際生效的限制
    plans: List[PlanLimitsModel]       # 各等級名義方案（供比較表）


class DailyUsageItem(UsageBucket):
    date: str
    turns: int = 0


class DailyUsageResponse(BaseModel):
    days: List[DailyUsageItem]


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=0, max_length=80)


class OkResponse(BaseModel):
    ok: bool


# ── Helpers ───────────────────────────────────────────────────────────

def build_usage_response(ctx: QuotaContext) -> MyUsageResponse:
    m = ctx.membership
    return MyUsageResponse(
        plan=m.plan,
        plan_expires_at=m.plan_expires_at,
        is_anonymous=m.is_anonymous,
        is_admin=m.is_admin,
        enforced=ctx.enforced,
        period=ctx.period,
        period_key=ctx.period_key,
        resets_at=ctx.resets_at_ms,
        used=ctx.used,
        limit=ctx.limit,
        remaining=ctx.remaining,
        ratio=round(ctx.ratio, 4),
        warn=ctx.warn,
        exceeded=ctx.exceeded,
        breakdown=UsageBreakdown(**ctx.breakdown),
        limits=PlanLimitsModel(**m.limits.to_dict()),
        plans=[PlanLimitsModel(**p.to_dict()) for p in plan_table()],
    )


def _check_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/me/usage", response_model=MyUsageResponse)
async def get_my_usage(claims: dict = Depends(require_auth_claims)) -> MyUsageResponse:
    return build_usage_response(await compute_quota(claims))


@router.get("/me/usage/daily", response_model=DailyUsageResponse)
async def get_my_daily_usage(
    days: int = Query(30, ge=1, le=92),
    uid: str = Depends(require_auth),
) -> DailyUsageResponse:
    return DailyUsageResponse(days=[DailyUsageItem(**d) for d in await get_daily_usage(uid, days)])


@router.patch("/me/sessions/{session_id}", response_model=OkResponse)
async def rename_my_session(
    session_id: str,
    body: RenameSessionRequest,
    uid: str = Depends(require_auth),
) -> OkResponse:
    _check_session_id(session_id)
    ok = await rename_session(uid, session_id, body.title.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="找不到此對話紀錄")
    return OkResponse(ok=True)


@router.delete("/me/sessions/{session_id}", response_model=OkResponse)
async def delete_my_session(
    session_id: str,
    request: Request,
    uid: str = Depends(require_auth),
) -> OkResponse:
    _check_session_id(session_id)
    await delete_session(uid, session_id)

    # 同步清掉記憶體內的 LangGraph checkpoint（若此版本的 saver 支援）。
    checkpointer = getattr(request.app.state.orchestrator, "checkpointer", None)
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if delete_thread is not None:
        try:
            delete_thread(f"{uid}:{session_id}")
        except Exception as exc:
            print(f"[account] delete_thread failed (uid={uid}, session={session_id}): {exc}")

    return OkResponse(ok=True)
