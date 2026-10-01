# -*- coding: utf-8 -*-
"""api/routes/graph.py — GET /graph/global-layout endpoint。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
import asyncio, os
from pathlib import Path

from core.infra.firebase_auth import require_auth

router = APIRouter()


class GlobalLayoutResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


def _get_rc_category_map(request: Request) -> dict:
    if request.app.state.rc_category_map is None:
        request.app.state.rc_category_map = request.app.state.rc_category_loader.load()
    return request.app.state.rc_category_map


def _enrich_nodes_with_category(nodes: list[dict], category_map: dict) -> None:
    for n in nodes:
        norm = (n.get("name") or "").strip().lower()
        info = category_map.get(norm)
        n["category"] = info["category"] if info else None
        n["category_color"] = info["color"] if info else None


@router.get("/graph/global-layout", response_model=GlobalLayoutResponse)
async def global_layout_endpoint(
    request: Request,
    _uid: str = Depends(require_auth),
) -> GlobalLayoutResponse:
    if request.app.state.layout_cache is not None:
        layout = request.app.state.layout_cache
    else:
        s = request.app.state.settings
        base_dir = Path(__file__).resolve().parent.parent.parent

        pkl_path = os.environ.get("RESULTS_PKL_PATH", str(base_dir / "results_dict.pkl"))
        chapter_pkl_path = os.environ.get("CHAPTER_PKL_PATH", str(base_dir / "chapter_results_dict.pkl"))

        layout = await asyncio.to_thread(
            request.app.state.layout_builder.build,
            pkl_path,
            chapter_pkl_path=chapter_pkl_path,
            gcs_bucket_name=(getattr(s, "gcs_bucketforresult", "") or "").strip(),
        )

        category_map = await asyncio.to_thread(_get_rc_category_map, request)
        _enrich_nodes_with_category(layout["nodes"], category_map)

        request.app.state.layout_cache = layout

    return GlobalLayoutResponse(nodes=layout["nodes"], edges=layout["edges"])
