# -*- coding: utf-8 -*-
"""Global layout 快取（lazy build，供 expand label 補齊與 /graph/global-layout 共用）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Request


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


async def ensure_layout_cache(request: Request) -> Dict[str, Any]:
    """若 layout_cache 尚未建立，與 GET /graph/global-layout 相同邏輯建一次。"""
    cached = request.app.state.layout_cache
    if cached is not None:
        return cached

    s = request.app.state.settings
    base_dir = Path(__file__).resolve().parent.parent

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
    return layout
