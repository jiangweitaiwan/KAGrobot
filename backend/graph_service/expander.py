# -*- coding: utf-8 -*-
"""graph_service/expander.py — 並行展開候選概念的知識圖譜。

GraphExpander 負責：
  1. 並行呼叫 DownstreamBFS 與 UpstreamBFS 展開各候選
  2. 修剪每個圖譜只保留 top-N 路徑
  3. 若有 intent_types，呼叫 PathScorer 重新評分
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from graph_service.scorer    import PathScorer
from retrieval.bfs            import DownstreamBFS, UpstreamBFS


_TOP_N_PATHS = 3


class GraphExpander:
    """並行展開所有候選概念的知識圖譜。

    Parameters
    ----------
    downstream_bfs : DownstreamBFS
    upstream_bfs   : UpstreamBFS
    scorer         : PathScorer
    """

    def __init__(
        self,
        downstream_bfs: DownstreamBFS,
        upstream_bfs:   UpstreamBFS,
        scorer:         PathScorer,
    ) -> None:
        self._downstream = downstream_bfs
        self._upstream   = upstream_bfs
        self._scorer     = scorer

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _expand_one(self, cand: dict) -> tuple[int, Any]:
        idx     = cand["index"]
        eid     = cand.get("eid") or ""
        rc_name = cand.get("rc_name") or ""

        if not eid:
            return idx, None

        try:
            raw_graph, foundations_data = await asyncio.gather(
                asyncio.to_thread(
                    self._downstream.run, eid, rc_name
                ),
                asyncio.to_thread(
                    self._upstream.run, eid
                ),
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
