# -*- coding: utf-8 -*-
"""graph_service/expander.py — 並行展開候選概念的知識圖譜（更新版）

與原版的差異：
  - 不再直接依賴 DownstreamBFS / UpstreamBFS
  - 改由 BFSCoordinator 統一排程，BFS 過程走鄰接層 cache
  - expand_one_async 已在 Coordinator 層做結果快取，
    相同 eid 第二次呼叫直接回傳，不重跑 BFS
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from graph_service.bfs_coordinator import BFSCoordinator
from graph_service.scorer          import PathScorer


_TOP_N_PATHS = 3


class GraphExpander:
    """並行展開所有候選概念的知識圖譜。

    Parameters
    ----------
    coordinator : BFSCoordinator  統一 BFS 排程器（取代原本的 downstream/upstream 注入）
    scorer      : PathScorer
    """

    def __init__(
        self,
        coordinator: BFSCoordinator,
        scorer:      PathScorer,
    ) -> None:
        self._coordinator = coordinator
        self._scorer      = scorer

    async def expand_all(
        self,
        candidates:   List[dict],
        intent_types: Optional[List[str]] = None,
    ) -> Dict[int, Any]:
        """並行展開前三個候選，回傳 {candidate_index: graph_data}。"""
        results = await asyncio.gather(
            *[self._expand_one(cand) for cand in candidates[:3]]
        )
        all_graphs = dict(results)

        if intent_types:
            all_graphs = {
                idx: self._scorer.score(g, intent_types)
                for idx, g in all_graphs.items()
                if g is not None
            }

        return all_graphs

    # ── Private ───────────────────────────────────────────────

    async def _expand_one(self, cand: dict) -> tuple[int, Any]:
        idx     = cand["index"]
        eid     = cand.get("eid") or ""
        rc_name = cand.get("rc_name") or ""

        if not eid:
            return idx, None

        try:
            # 透過 Coordinator 取得兩方向結果（鄰接層 cache 共享）
            raw_graph, foundations_data = await self._coordinator.expand_one_async(
                eid, rc_name
            )

            raw_graph["foundations"]     = foundations_data["final_foundations"]
            raw_graph["foundation_tree"] = foundations_data["foundation_tree"]

            trimmed = self._trim_to_top_paths(raw_graph)
            return idx, trimmed

        except Exception as exc:
            print(f"[GraphExpander] failed to expand '{rc_name}': {exc}")
            return idx, None

    @staticmethod
    def _trim_to_top_paths(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """只保留 top-N 條路徑及其節點/邊，重設 path_id。"""
        all_paths = graph_data.get("paths") or []
        if not all_paths:
            return graph_data

        sorted_paths = sorted(
            all_paths,
            key=lambda p: (p.get("intent_score", 0.0), len(p.get("node_ids", []))),
            reverse=True,
        )
        top_paths = sorted_paths[:_TOP_N_PATHS]
        for new_id, path in enumerate(top_paths):
            path["path_id"] = new_id

        used_node_ids: set = set()
        used_edge_ids: set = set()
        for path in top_paths:
            used_node_ids.update(path.get("node_ids") or [])
            used_edge_ids.update(path.get("edge_ids") or [])

        return {
            **graph_data,
            "paths":            top_paths,
            "nodes":            [
                n for n in (graph_data.get("nodes") or [])
                if n["id"] in used_node_ids
            ],
            "edges":            [
                e for e in (graph_data.get("edges") or [])
                if e["id"] in used_edge_ids
            ],
            "selected_path_id": 0,
        }
