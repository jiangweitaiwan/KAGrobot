# -*- coding: utf-8 -*-
"""api/routes/graph.py — GET /graph/global-layout endpoint。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from core.infra.firebase_auth import require_auth
from graph_service.layout_cache import ensure_layout_cache

router = APIRouter()


class GlobalLayoutResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@router.get("/graph/global-layout", response_model=GlobalLayoutResponse)
async def global_layout_endpoint(
    request: Request,
    _uid: str = Depends(require_auth),
) -> GlobalLayoutResponse:
    layout = await ensure_layout_cache(request)
    return GlobalLayoutResponse(nodes=layout["nodes"], edges=layout["edges"])
