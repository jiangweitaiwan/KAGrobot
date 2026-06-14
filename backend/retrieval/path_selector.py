# -*- coding: utf-8 -*-
"""retrieval/path_selector.py — Foundation 路徑選取策略。

替換 select() 的實作即可切換選路策略，不影響呼叫端。
"""

from __future__ import annotations

from typing import Any, Dict, List


class PathSelector:
    """從 foundation raw_paths 中選出兼顧深度與多樣性的路徑。

    演算法（貪婪 Jaccard 多樣性選取）：
      1. 以步數中位數作為深度門檻，過濾太短的路徑。
      2. 第一條取最深路徑。
      3. 後續每輪選與已選集合 Jaccard distance 最大的路徑。
    """

    def select(
        self,
        paths:        List[Dict[str, Any]],
        max_branches: int = 3,
    ) -> List[Dict[str, Any]]:
        """回傳至多 max_branches 條精選路徑。"""
        if not paths or max_branches <= 0:
            return []

        decorated  = self._decorate(paths)
        candidates = self._filter_by_depth(decorated)

        if not candidates:
            candidates = list(decorated)

        return self._greedy_select(candidates, max_branches)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decorate(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "path":     p,
                "depth":    len(p.get("steps", [])),
                "node_set": PathSelector._path_node_set(p),
            }
            for p in paths
        ]

    @staticmethod
    def _filter_by_depth(
        decorated: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        depths = sorted(d["depth"] for d in decorated)
        median = depths[len(depths) // 2]
        return [d for d in decorated if d["depth"] >= median]

    @staticmethod
    def _greedy_select(
        candidates:   List[Dict[str, Any]],
        max_branches: int,
    ) -> List[Dict[str, Any]]:
        candidates = sorted(candidates, key=lambda d: d["depth"], reverse=True)

        selected:           List[Dict[str, Any]] = [candidates[0]]
        selected_node_sets: List[set]            = [candidates[0]["node_set"]]
        remaining = candidates[1:]

        while len(selected) < max_branches and remaining:
            remaining.sort(
                key=lambda d: min(
                    PathSelector._jaccard_distance(d["node_set"], s)
                    for s in selected_node_sets
                ),
                reverse=True,
            )
            best = remaining.pop(0)
            selected.append(best)
            selected_node_sets.append(best["node_set"])

        return [d["path"] for d in selected]

    @staticmethod
    def _path_node_set(path: Dict[str, Any]) -> set:
        nodes: set = set()
        for step in path.get("steps", []):
            for key in ("from_concept", "to_concept"):
                c = step.get(key)
                if c:
                    nodes.add(c["element_id"])
        return nodes

    @staticmethod
    def _jaccard_distance(set_a: set, set_b: set) -> float:
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return 1.0 - len(set_a & set_b) / union
