# -*- coding: utf-8 -*-
"""api/routes/graph.py — GET /graph/global-layout endpoint。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class GlobalLayoutResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@router.get("/graph/global-layout", response_model=GlobalLayoutResponse)
async def global_layout_endpoint(request: Request) -> GlobalLayoutResponse:
    # 快取掛在 app.state.layout_cache
    if request.app.state.layout_cache is not None:
        layout = request.app.state.layout_cache
    else:
        s        = request.app.state.settings
        base_dir = Path(__file__).resolve().parent.parent.parent

        pkl_path = os.environ.get(
            "RESULTS_PKL_PATH",
            str(base_dir / "results_dict.pkl"),
        )
        chapter_pkl_path = os.environ.get(
            "CHAPTER_PKL_PATH",
            str(base_dir / "chapter_results_dict.pkl"),
        )

        layout = await asyncio.to_thread(
            request.app.state.layout_builder.build,
            pkl_path,
            chapter_pkl_path=chapter_pkl_path,
            gcs_bucket_name=(getattr(s, "gcs_bucketforresult", "") or "").strip(),
        )
        request.app.state.layout_cache = layout

    return GlobalLayoutResponse(
        nodes=layout["nodes"],
        edges=layout["edges"],
    )
