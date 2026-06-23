# -*- coding: utf-8 -*-
"""graph_service/bfs_engine.py — 純 Python BFS 引擎

職責：
  - 不直接打 Neo4j，所有鄰接查詢都透過 AdjacencyCache
  - 提供 downstream_bfs / upstream_bfs 兩個方向的展開
  - 輸出格式與原 retrieval/bfs.py 完全相容，讓上層呼叫端不需要改動

與原 bfs.py 的差異：
  - 移除所有 _driver.session() / tx.run() 呼叫
  - 每個節點的鄰接關係改由 AdjacencyCache 提供
  - BFS 邏輯本身（visited set, deque, path assembly）維持不變
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from graph_service.bfs_cache import (
    AdjacencyCache,
    ConceptInfo,
    DownstreamNeighbor,
    UpstreamNeighbor,
)
from retrieval.bfs import (
    _integrate_path,
    _upsert_concept_node,
)
from retrieval.path_selector import PathSelector


# ══════════════════════════════════════════════════════════════
# 內部 helper：把 cache dataclass 轉成 bfs.py 慣用的 dict 格式
# ══════════════════════════════════════════════════════════════

def _concept_to_dict(c: ConceptInfo) -> dict:
    return {"element_id": c.element_id, "name": c.name, "type": c.node_type}


def _paper_to_dict(p) -> Optional[dict]:
    if p is None:
        return None
    return {
        "element_id":                   p.element_id,
        "title":                        p.title,
        "conceptual_premises":          p.conceptual_premises,
        "contextual_model_constraints": p.contextual_model_constraints,
    }


# ══════════════════════════════════════════════════════════════
# CachedDownstreamBFS
# ══════════════════════════════════════════════════════════════

class CachedDownstreamBFS:
    """從 Resulting Concept 向下游展開 BFS。

    所有 Neo4j 查詢皆透過 AdjacencyCache，重複節點的鄰接關係
    在同一個 request 與跨 request 之間都可命中快取。

    輸出格式與 retrieval.bfs.DownstreamBFS 完全相容。
    """

    def __init__(self, cache: AdjacencyCache) -> None:
        self._cache = cache

    def run(
        self,
        start_eid:      str,
        rc_name:        str = "",
        max_depth_down: int = 8,
    ) -> Dict[str, Any]:
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        all_paths: List[dict]      = []

        _upsert_concept_node(
            nodes_map, start_eid, rc_name, "Resulting_Concept", matched=True
        )

        # 判斷節點類型，決定實際展開起點
        kind = self._cache.get_concept_kind(start_eid)

        if kind == "resulting_concept":
            effective_eid = start_eid
        elif kind == "sub_concept":
            parent = self._cache.get_parent_rc(start_eid)
            if parent:
                _upsert_concept_node(
                    nodes_map, parent.element_id, parent.name, parent.node_type
                )
                effective_eid = parent.element_id
            else:
                effective_eid = start_eid
        else:
            effective_eid = None

        if effective_eid:
            raw_paths = self._walk(effective_eid, max_depth_down)
        else:
            raw_paths = self._walk_from_other(start_eid, max_depth_down)

        for raw in raw_paths:
            _integrate_path(raw, nodes_map, edges_map, all_paths, direction="downstream")

        # 去重
        seen:         set        = set()
        unique_paths: List[dict] = []
        for p in all_paths:
            sig = tuple(p["node_ids"])
            if sig not in seen:
                seen.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)

        return {
            "query_concepts": [{
                "id":    start_eid,
                "name":  rc_name,
                "type":  "Resulting_Concept",
                "score": 1.0,
            }],
            "nodes":           list(nodes_map.values()),
            "edges":           list(edges_map.values()),
            "paths":           unique_paths,
            "foundations":     [],
            "foundation_tree": {"nodes": [], "edges": []},
        }

    # ── BFS 核心 ─────────────────────────────────────────────

    def _walk(
        self,
        start_eid: str,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        """從 RC 向下展開，透過 cache 取鄰接關係。"""
        start_concept = {"element_id": start_eid, "name": "", "type": "Resulting_Concept"}
        queue         = deque([(start_concept, [], frozenset([start_eid]))])
        completed:    List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth:
                fallback_paper = _paper_to_dict(self._cache.get_fallback_paper(cur_eid))
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   fallback_paper,
                })
                continue

            # 透過 cache 取鄰居（不打 Neo4j）
            neighbors: List[DownstreamNeighbor] = self._cache.get_downstream_neighbors(cur_eid)
            new_neighbors = [n for n in neighbors if n.to_concept.element_id not in visited]

            if not new_neighbors:
                fallback_paper = _paper_to_dict(self._cache.get_fallback_paper(cur_eid))
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   fallback_paper,
                })
                continue

            for neighbor in new_neighbors:
                next_c   = _concept_to_dict(neighbor.to_concept)
                new_step = {
                    "from_concept": cur_c,
                    "via_paper":    _paper_to_dict(neighbor.via_paper),
                    "to_concept":   next_c,
                }
                queue.append((
                    next_c,
                    steps + [new_step],
                    visited | {neighbor.to_concept.element_id},
                ))

        return completed

    def _walk_from_other(
        self,
        eid:       str,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        """非 RC 節點：先找所有可達的 RC，再各自展開。"""
        # 取下游鄰居中屬於 RC 的節點
        neighbors = self._cache.get_downstream_neighbors(eid)
        rc_eids   = [
            n.to_concept.element_id
            for n in neighbors
            if (n.to_concept.node_type or "").lower() == "resulting_concept"
        ]

        all_paths: List[Dict[str, Any]] = []
        for rc_eid in rc_eids:
            all_paths.extend(self._walk(rc_eid, max_depth))
        return all_paths


# ══════════════════════════════════════════════════════════════
# CachedUpstreamBFS
# ══════════════════════════════════════════════════════════════

class CachedUpstreamBFS:
    """從 Resulting Concept 向上游追溯基礎概念。

    輸出格式與 retrieval.bfs.UpstreamBFS 完全相容。
    """

    def __init__(self, cache: AdjacencyCache, path_selector: PathSelector) -> None:
        self._cache    = cache
        self._selector = path_selector

    def run(
        self,
        start_eid:    str,
        max_depth:    int = 10,
        max_branches: int = 4,
    ) -> Dict[str, Any]:
        raw_paths      = self._walk(start_eid, max_depth)
        selected_paths = self._selector.select(raw_paths, max_branches)

        seen_leaf_eids:   set                  = set()
        final_foundations: List[Dict[str, Any]] = []
        tree_nodes:        Dict[str, dict]      = {}
        tree_edges:        Dict[str, dict]      = {}

        for path in selected_paths:
            steps = path.get("steps") or []
            if not steps:
                continue

            leaf     = steps[0]["from_concept"]
            leaf_eid = leaf["element_id"]
            if leaf_eid not in seen_leaf_eids:
                seen_leaf_eids.add(leaf_eid)
                final_foundations.append({
                    "element_id": leaf_eid,
                    "label":      leaf.get("name") or leaf_eid,
                })

            for step in steps:
                for role in ("from_concept", "to_concept"):
                    c   = step[role]
                    eid = c["element_id"]
                    if eid not in tree_nodes:
                        tree_nodes[eid] = {
                            "element_id": eid,
                            "label":      c.get("name") or eid,
                        }

                fc       = step["from_concept"]
                tc       = step["to_concept"]
                edge_key = f"{fc['element_id']}__{tc['element_id']}"
                if edge_key not in tree_edges:
                    tree_edges[edge_key] = {
                        "source": fc["element_id"],
                        "target": tc["element_id"],
                    }

        return {
            "final_foundations": final_foundations,
            "foundation_tree": {
                "nodes": list(tree_nodes.values()),
                "edges": list(tree_edges.values()),
            },
        }

    # ── BFS 核心 ─────────────────────────────────────────────

    def _walk(
        self,
        start_eid: str,
        max_depth: int = 10,
    ) -> List[Dict[str, Any]]:
        """向上游追溯，透過 cache 取鄰接關係。"""
        start_concept = {"element_id": start_eid, "name": "", "type": None}
        queue         = deque([(start_concept, [], frozenset([start_eid]))])
        completed:    List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth:
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   None,
                })
                continue

            # 透過 cache 取上游鄰居（不打 Neo4j）
            neighbors: List[UpstreamNeighbor] = self._cache.get_upstream_neighbors(cur_eid)
            new_neighbors = [n for n in neighbors if n.from_concept.element_id not in visited]

            if not new_neighbors:
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   None,
                })
                continue

            for neighbor in new_neighbors:
                prev_c   = _concept_to_dict(neighbor.from_concept)
                new_step = {
                    "from_concept": prev_c,
                    "via_paper":    _paper_to_dict(neighbor.via_paper),
                    "to_concept":   cur_c,
                }
                queue.append((
                    prev_c,
                    [new_step] + steps,
                    visited | {neighbor.from_concept.element_id},
                ))

        return [r for r in completed if r["steps"]]
