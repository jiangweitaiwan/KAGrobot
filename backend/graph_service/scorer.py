# -*- coding: utf-8 -*-
"""graph_service/scorer.py — 根據 intent_types 對路徑評分。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PathScorer:
    """依 intent_types 對 graph_data 內的路徑重新評分並排序。"""

    def score(
        self,
        graph_data:   Dict[str, Any],
        intent_types: List[str],
    ) -> Dict[str, Any]:
        """回傳評分後的新 graph_data（不修改原物件）。"""
        if not intent_types or not graph_data:
            return graph_data

        nodes_by_id: Dict[str, Any] = {
            n["id"]: n for n in graph_data.get("nodes", [])
        }

        scored_paths = []
        for path in graph_data.get("paths", []):
            final_id   = path.get("final_concept")
            final_node = nodes_by_id.get(final_id or "", {})
            type_score = self._type_overlap(
                final_node.get("analysis_types", []), intent_types
            )
            scored_paths.append({**path, "intent_score": type_score})

        scored_paths.sort(
            key=lambda p: (p["intent_score"], len(p.get("node_ids", []))),
            reverse=True,
        )
        for new_id, path in enumerate(scored_paths):
            path["path_id"] = new_id

        return {**graph_data, "paths": scored_paths, "selected_path_id": 0}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _type_overlap(
        node_analysis_types: List[str],
        intent_types:        List[str],
    ) -> float:
        if not node_analysis_types or not intent_types:
            return 0.0
        matched = len(set(node_analysis_types) & set(intent_types))
        return matched / len(intent_types)
