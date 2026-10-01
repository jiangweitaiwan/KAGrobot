# -*- coding: utf-8 -*-
"""api/routes/admin.py — 管理者端點（全部需要 require_admin）。

總覽
  GET  /admin/overview?period=YYYYMM         本月全站用量、依功能 / 依等級分解、有效 VIP 人數
  GET  /admin/overview/daily?days=30         全站每日趨勢
  GET  /admin/top-users?period=&limit=10     用量最高的使用者（需要 Firestore 索引，缺少時回 503 index_required）

使用者
  GET  /admin/users?q=&include_guests=&page_token=&limit=
                                              q 為完整 Email 或 UID；空白 = 分頁列出已註冊使用者
  GET  /admin/users/{uid}                     身分 + 有效等級 + 額度用量
  GET  /admin/users/{uid}/usage               （相容舊版）同上的 usage 部分
  GET  /admin/users/{uid}/usage/daily?days=   該使用者的每日用量
  PUT  /admin/users/{uid}/plan                設定等級（會寫稽核紀錄）

VIP / 稽核
  GET  /admin/vips                            VIP 名單（含已到期者，已到期排最前）
  GET  /admin/audit?limit=&target_uid=        操作紀錄

等級規則一律走 core.account.membership.plan_from_record，與額度檢查實際套用的規則是同一份，
所以管理者看到的「有效等級」不會與系統行為不一致。

PUT plan 使用範例（curl，token 為管理者的 Firebase ID token）：
    curl -X PUT $API/admin/users/<uid>/plan \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"plan":"vip","expires_at":1798761600000}'
金流串接後，webhook 只需呼叫 core.account.membership.set_user_plan()，不必動額度邏輯。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from google.api_core import exceptions as gexc
from pydantic import BaseModel

from api.routes.account import (
    DailyUsageItem,
    DailyUsageResponse,
    MyUsageResponse,
    UsageBucket,
    build_usage_response,
)
from core.admin import admin_audit, admin_data, admin_directory
from core.admin.admin_directory import UserIdentity
from core.infra.firebase_auth import _admin_uids, require_admin
from core.account.membership import plan_from_record, set_user_plan
from core.account.quota import compute_quota
from core.account.usage import get_daily_usage, get_global_daily_usage, month_key, now_local

router = APIRouter()

_UID_RE = re.compile(r"^[A-Za-z0-9\-_]{1,128}$")
_PERIOD_PATTERN = r"^\d{6}$"


# ── Schema ────────────────────────────────────────────────────────────

class AdminIdentity(BaseModel):
    uid: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    is_anonymous: bool
    is_admin: bool = False
    disabled: bool = False
    created_at: Optional[int] = None
    last_sign_in_at: Optional[int] = None


class AdminUserRow(AdminIdentity):
    plan: str
    plan_expires_at: Optional[int] = None
    month_tokens: int = 0
    lifetime_tokens: int = 0


class AdminUserListResponse(BaseModel):
    users: List[AdminUserRow]
    next_page_token: Optional[str] = None


class AdminUserDetail(BaseModel):
    identity: AdminIdentity
    plan: str
    plan_expires_at: Optional[int] = None
    usage: MyUsageResponse


class PlanBucket(BaseModel):
    total_tokens: int = 0
    turns: int = 0


class AdminOverviewResponse(BaseModel):
    period: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    calls: int
    turns: int
    by_feature: Dict[str, UsageBucket]
    by_plan: Dict[str, PlanBucket]
    vip_active: int


class TopUser(BaseModel):
    uid: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    plan: str
    total_tokens: int
    turns: int


class TopUsersResponse(BaseModel):
    period: str
    users: List[TopUser]


class VipRow(BaseModel):
    uid: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    plan_expires_at: Optional[int] = None
    expired: bool
    month_tokens: int = 0


class VipListResponse(BaseModel):
    vips: List[VipRow]


class AuditEntry(BaseModel):
    id: str
    admin_uid: Optional[str] = None
    admin_email: Optional[str] = None
    action: str
    target_uid: str
    before: Dict[str, Any] = {}
    after: Dict[str, Any] = {}
    at: int


class AuditResponse(BaseModel):
    entries: List[AuditEntry]


class SetPlanRequest(BaseModel):
    plan: Literal["normal", "vip"]
    expires_at: Optional[int] = None   # unix ms；僅 vip 有意義，省略 = 不過期


class SetPlanResponse(BaseModel):
    uid: str
    plan: str
    plan_expires_at: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────

def _check_uid(uid: str) -> None:
    if not _UID_RE.match(uid):
        raise HTTPException(status_code=400, detail="Invalid uid format")


def _is_admin_identity(u: UserIdentity) -> bool:
    return u.custom_claims.get("admin") is True or u.uid in _admin_uids()


def _identity_model(u: UserIdentity) -> AdminIdentity:
    return AdminIdentity(
        uid=u.uid, email=u.email, display_name=u.display_name, is_anonymous=u.is_anonymous,
        is_admin=_is_admin_identity(u), disabled=u.disabled,
        created_at=u.created_at, last_sign_in_at=u.last_sign_in_at,
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _rows_for(identities: List[UserIdentity], period: str) -> List[AdminUserRow]:
    """把 Firebase 身分與 Firestore 的等級 / 用量合併成列表列。三份讀取並行。"""
    uids = [u.uid for u in identities]
    if not uids:
        return []
    plan_recs, month, life = await asyncio.gather(
        admin_data.get_plan_records(uids),
        admin_data.get_month_tokens(uids, period),
        admin_data.get_lifetime_tokens(uids),
    )
    rows: List[AdminUserRow] = []
    for u in identities:
        raw_plan, raw_exp = plan_recs.get(u.uid, (None, None))
        plan, exp = plan_from_record(raw_plan, raw_exp, anonymous=u.is_anonymous)
        rows.append(AdminUserRow(
            **_identity_model(u).model_dump(),
            plan=plan, plan_expires_at=exp,
            month_tokens=month.get(u.uid, 0), lifetime_tokens=life.get(u.uid, 0),
        ))
    return rows


async def _identity_or_404(uid: str) -> UserIdentity:
    _check_uid(uid)
    ident = await asyncio.to_thread(admin_directory.get_user, uid)
    if ident is None:
        raise HTTPException(status_code=404, detail="找不到此使用者")
    return ident


def _claims_for(ident: UserIdentity) -> Dict[str, Any]:
    """組出「假裝是這位使用者」的 claims，讓 compute_quota 能沿用同一套額度計算。"""
    return {
        **ident.custom_claims,
        "uid": ident.uid,
        "firebase": {"sign_in_provider": "anonymous" if ident.is_anonymous else "password"},
    }


# ── 總覽 ──────────────────────────────────────────────────────────────

@router.get("/admin/overview", response_model=AdminOverviewResponse)
async def admin_overview(
    period: Optional[str] = Query(None, pattern=_PERIOD_PATTERN),
    _admin: dict = Depends(require_admin),
) -> AdminOverviewResponse:
    key = period or month_key(now_local())
    g, vip_records = await asyncio.gather(admin_data.get_global_month(key), admin_data.list_vip_records())
    now = _now_ms()
    vip_active = sum(1 for _uid, exp in vip_records if exp is None or exp > now)

    return AdminOverviewResponse(
        period=key,
        total_tokens=g["total_tokens"], input_tokens=g["input_tokens"],
        output_tokens=g["output_tokens"], reasoning_tokens=g["reasoning_tokens"],
        calls=g["calls"], turns=g["turns"],
        by_feature={k: UsageBucket(**v) for k, v in g["by_feature"].items()},
        by_plan={k: PlanBucket(**v) for k, v in g["by_plan"].items()},
        vip_active=vip_active,
    )


@router.get("/admin/overview/daily", response_model=DailyUsageResponse)
async def admin_overview_daily(
    days: int = Query(30, ge=1, le=92),
    _admin: dict = Depends(require_admin),
) -> DailyUsageResponse:
    return DailyUsageResponse(days=[DailyUsageItem(**d) for d in await get_global_daily_usage(days)])


@router.get("/admin/top-users", response_model=TopUsersResponse)
async def admin_top_users(
    period: Optional[str] = Query(None, pattern=_PERIOD_PATTERN),
    limit: int = Query(10, ge=1, le=50),
    _admin: dict = Depends(require_admin),
) -> TopUsersResponse:
    key = period or month_key(now_local())
    try:
        ranked = await admin_data.top_users(key, limit)
    except gexc.FailedPrecondition as exc:
        raise HTTPException(status_code=503, detail={
            "code": "index_required",
            "message": "需要先建立 Firestore 索引才能排序使用者用量："
                       "collection group「usage_monthly」，欄位 period 遞增、total_tokens 遞減。"
                       "建立方式見 backend/firestore.indexes.json。",
            "firestore_error": str(exc),
        })

    uids = [uid for uid, _ in ranked]
    identities = await asyncio.to_thread(admin_directory.get_users, uids) if uids else {}
    plan_recs = await admin_data.get_plan_records(uids) if uids else {}

    users: List[TopUser] = []
    for uid, usage in ranked:
        ident = identities.get(uid)
        raw_plan, raw_exp = plan_recs.get(uid, (None, None))
        plan, _ = plan_from_record(raw_plan, raw_exp, anonymous=bool(ident and ident.is_anonymous))
        users.append(TopUser(
            uid=uid, email=ident.email if ident else None,
            display_name=ident.display_name if ident else None,
            plan=plan, total_tokens=usage["total_tokens"], turns=usage["turns"],
        ))
    return TopUsersResponse(period=key, users=users)


# ── 使用者 ────────────────────────────────────────────────────────────

@router.get("/admin/users", response_model=AdminUserListResponse)
async def admin_list_users(
    q: Optional[str] = Query(None, max_length=200),
    include_guests: bool = False,
    page_token: Optional[str] = None,
    limit: int = Query(30, ge=1, le=100),
    _admin: dict = Depends(require_admin),
) -> AdminUserListResponse:
    period = month_key(now_local())
    query = (q or "").strip()

    if query:
        # 精確查找：Firebase 不支援模糊搜尋。搜尋結果一律顯示（含訪客）。
        if "@" in query:
            found = await asyncio.to_thread(admin_directory.find_user_by_email, query)
        elif _UID_RE.match(query):
            found = await asyncio.to_thread(admin_directory.get_user, query)
        else:
            raise HTTPException(status_code=400, detail="請輸入完整的 Email 或 UID（Firebase 不支援模糊搜尋）")
        return AdminUserListResponse(users=await _rows_for([found] if found else [], period))

    # 分頁列出。「不含訪客」是拿到頁面後才過濾（Firebase 無法在查詢時篩選），
    # 所以補頁直到湊滿 limit 或用完；每次都以 limit 為頁大小、且不截斷，
    # 避免 page_token 已前進卻丟掉還沒顯示的人。最多補 5 頁以免訪客極多時卡住。
    collected: List[UserIdentity] = []
    token = page_token
    for _ in range(5):
        batch, token = await asyncio.to_thread(admin_directory.list_users_page, token, limit)
        collected.extend(u for u in batch if include_guests or not u.is_anonymous)
        if len(collected) >= limit or not token:
            break

    return AdminUserListResponse(users=await _rows_for(collected, period), next_page_token=token)


@router.get("/admin/users/{uid}", response_model=AdminUserDetail)
async def admin_user_detail(uid: str, _admin: dict = Depends(require_admin)) -> AdminUserDetail:
    ident = await _identity_or_404(uid)
    ctx = await compute_quota(_claims_for(ident))
    return AdminUserDetail(
        identity=_identity_model(ident),
        plan=ctx.membership.plan,
        plan_expires_at=ctx.membership.plan_expires_at,
        usage=build_usage_response(ctx),
    )


@router.get("/admin/users/{uid}/usage", response_model=MyUsageResponse)
async def admin_get_usage(uid: str, _admin: dict = Depends(require_admin)) -> MyUsageResponse:
    ident = await _identity_or_404(uid)
    return build_usage_response(await compute_quota(_claims_for(ident)))


@router.get("/admin/users/{uid}/usage/daily", response_model=DailyUsageResponse)
async def admin_user_daily(
    uid: str,
    days: int = Query(30, ge=1, le=92),
    _admin: dict = Depends(require_admin),
) -> DailyUsageResponse:
    _check_uid(uid)
    return DailyUsageResponse(days=[DailyUsageItem(**d) for d in await get_daily_usage(uid, days)])


@router.put("/admin/users/{uid}/plan", response_model=SetPlanResponse)
async def admin_set_plan(
    uid: str,
    body: SetPlanRequest,
    admin: dict = Depends(require_admin),
) -> SetPlanResponse:
    ident = await _identity_or_404(uid)
    if ident.is_anonymous:
        raise HTTPException(status_code=400, detail="訪客帳號不能設定等級，請對方先註冊。")
    if body.plan == "vip" and body.expires_at is not None and body.expires_at <= _now_ms():
        raise HTTPException(status_code=400, detail="到期日必須在未來。")

    raw_plan, raw_exp = (await admin_data.get_plan_records([uid]))[uid]
    before_plan, before_exp = plan_from_record(raw_plan, raw_exp)

    await set_user_plan(uid, body.plan, body.expires_at)

    after_exp = body.expires_at if body.plan == "vip" else None
    await admin_audit.record(
        admin, "plan.set", uid,
        before={"plan": before_plan, "plan_expires_at": before_exp},
        after={"plan": body.plan, "plan_expires_at": after_exp},
    )
    return SetPlanResponse(uid=uid, plan=body.plan, plan_expires_at=after_exp)


# ── VIP 名單 / 稽核 ───────────────────────────────────────────────────

@router.get("/admin/vips", response_model=VipListResponse)
async def admin_vips(_admin: dict = Depends(require_admin)) -> VipListResponse:
    records = await admin_data.list_vip_records()
    # 永久 VIP（None）排最後；已到期者到期日最小，自然排最前，方便管理者清理。
    records.sort(key=lambda r: (r[1] is None, r[1] or 0))
    uids = [uid for uid, _ in records]
    if not uids:
        return VipListResponse(vips=[])

    identities, month = await asyncio.gather(
        asyncio.to_thread(admin_directory.get_users, uids),
        admin_data.get_month_tokens(uids, month_key(now_local())),
    )
    now = _now_ms()
    return VipListResponse(vips=[
        VipRow(
            uid=uid,
            email=identities[uid].email if uid in identities else None,
            display_name=identities[uid].display_name if uid in identities else None,
            plan_expires_at=exp,
            expired=exp is not None and exp <= now,
            month_tokens=month.get(uid, 0),
        )
        for uid, exp in records
    ])


@router.get("/admin/audit", response_model=AuditResponse)
async def admin_audit_log(
    limit: int = Query(50, ge=1, le=200),
    target_uid: Optional[str] = None,
    _admin: dict = Depends(require_admin),
) -> AuditResponse:
    if target_uid:
        _check_uid(target_uid)
    return AuditResponse(entries=[AuditEntry(**e) for e in await admin_audit.list_entries(limit, target_uid)])
