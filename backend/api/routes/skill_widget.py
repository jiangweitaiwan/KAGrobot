# -*- coding: utf-8 -*-
"""api/routes/skill_widget.py — HTML Widget 即時試算 endpoint。

POST /skill/widget/dispatch

刻意不掛 require_auth：這支 endpoint 是給 sandboxed iframe（
`<iframe sandbox="allow-scripts">`，見 frontend/app/components/
SkillWidgetCard.tsx）呼叫的，iframe 內的 JS 拿不到 parent 頁面的
Firebase ID token（也不應該拿到——widget HTML 本身是不受信任內容，
不能給它 allow-same-origin，否則它就能讀 parent 的 cookie/localStorage）。

改用獨立的 widget_token（core/widget_token.py 簽發的 HMAC 簽章 token）
做身分與授權驗證：
  - token 由 show_interactive_html 工具在顯示 widget 當下簽發，
    payload 內已經編碼好 uid / rc_name / 這個 widget 被允許呼叫的
    compute_functions 白名單。
  - dispatch 時只驗簽章與過期時間，不查資料庫，符合「即時試算」
    要求的低延遲。
  - action 必須存在於 token payload 的白名單內，不接受任意路徑——
    即使請求者拿到別的合法 token，也只能呼叫該 token 綁定的
    compute_functions，不能越權呼叫其他 widget 或其他 skill 的函式。

實際執行走 core/skill_compute.py 的 subprocess 隔離（見該檔案檔頭
關於信任邊界的說明）。skill_loader.load() 本身是同步 I/O（GCS 下載 +
zip 解壓，見 core/skill_loader.py），比照 orchestrator/nodes.py 的既有
慣例用 asyncio.to_thread 包住，避免卡住 event loop；正常情況下命中
SkillLoader 內部快取，幾乎零延遲。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.skills.skill_compute import run_compute_function
from core.infra.widget_token import verify_token

router = APIRouter()


class DispatchRequest(BaseModel):
    widget_token: str
    action:       str
    parameters:   Dict[str, Any] = {}


class DispatchResponse(BaseModel):
    ok:     bool
    result: Any = None
    error:  Optional[str] = None


@router.post("/skill/widget/dispatch", response_model=DispatchResponse)
async def dispatch(body: DispatchRequest, request: Request) -> DispatchResponse:
    settings = request.app.state.settings

    payload = verify_token(settings.widget_token_secret, body.widget_token)
    if payload is None:
        raise HTTPException(status_code=401, detail="widget_token 無效或已過期")

    allowed_fns: Dict[str, str] = payload.get("fns") or {}
    fn_ref = allowed_fns.get(body.action)
    if fn_ref is None:
        raise HTTPException(
            status_code=403,
            detail=f"action '{body.action}' 不在此 widget 允許呼叫的清單內",
        )

    rc_name = payload.get("rc_name", "")
    skill_loader = request.app.state.skill_loader
    skill = await asyncio.to_thread(skill_loader.load, rc_name)

    if skill is None or skill.skill_dir is None:
        raise HTTPException(status_code=404, detail=f"找不到 skill：{rc_name}")

    result = await run_compute_function(
        skill.skill_dir,
        fn_ref,
        body.parameters,
        timeout_s=settings.widget_compute_timeout_s,
        mem_limit_mb=settings.widget_compute_mem_limit_mb,
    )

    return DispatchResponse(**result)
