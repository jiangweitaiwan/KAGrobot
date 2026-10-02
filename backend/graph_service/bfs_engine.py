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
# Sign propagation（upstream foundation 的 support/critique 判定）
# ══════════════════════════════════════════════════════════════

_GAMMA = 0.6  # decay 係數：離 RC 越遠（depth 越大），該路徑對 score 的貢獻越小


def _decay(depth: int) -> float:
    return _GAMMA ** depth


def _combine(parent_type: str, edge_type: str) -> str:
    """sign propagation 規則：
        parent=other                    → other   （吸收態，一旦變 other 就不再回頭）
        edge=critique, parent=critique   → other   （c×c：批評的批評，不支持也不反對 RC）
        edge=critique, parent=support    → critique（s×c：支持「被批評者」的論述 → RC 的批評）
        edge=support                     → parent_type（s×s=s, c×s=c：不改變既有立場）
    """
    if parent_type == "other":
        return "other"
    if edge_type == "critique":
        return "other" if parent_type == "critique" else "critique"
    return parent_type

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
            effective_eid, effective_name = start_eid, rc_name
        elif kind == "sub_concept":
            parent = self._cache.get_parent_rc(start_eid)
            if parent:
                _upsert_concept_node(
                    nodes_map, parent.element_id, parent.name, parent.node_type
                )
                effective_eid, effective_name = parent.element_id, parent.name
            else:
                effective_eid, effective_name = start_eid, rc_name
        else:
            effective_eid, effective_name = None, ""

        if effective_eid:
            raw_paths = self._walk(effective_eid, effective_name, max_depth_down)
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
        start_eid:  str,
        start_name: str,
        max_depth:  int = 8,
    ) -> List[Dict[str, Any]]:
        """從 RC 向下展開，透過 cache 取鄰接關係。"""
        start_concept = {"element_id": start_eid, "name": start_name, "type": "Resulting_Concept"}
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
        # 取下游鄰居中屬於 RC 的節點（連同 name 一併保留，避免 leaf RC 遺失 label）
        neighbors = self._cache.get_downstream_neighbors(eid)
        rc_infos  = [
            n.to_concept
            for n in neighbors
            if (n.to_concept.node_type or "").lower() == "resulting_concept"
        ]

        all_paths: List[Dict[str, Any]] = []
        for rc_info in rc_infos:
            all_paths.extend(self._walk(rc_info.element_id, rc_info.name, max_depth))
        return all_paths


# ══════════════════════════════════════════════════════════════
# CachedUpstreamBFS
# ══════════════════════════════════════════════════════════════

class CachedUpstreamBFS:
    def __init__(self, cache: AdjacencyCache, path_selector: PathSelector) -> None:
        self._cache    = cache
        self._selector = path_selector

    def run(
        self,
        start_eid:    str,
        max_depth:    int = 10,
        max_branches: int = 4,
        start_name:   str = "",
    ) -> Dict[str, Any]:
        raw_paths = self._walk(start_eid, max_depth, start_name=start_name)

        # ── Score 用：對「全部」raw_paths 彙總（不受下方 diversity selection
        #    影響，確保同一 leaf 的分數不會因為多樣性演算法選中哪條代表路徑
        #    而飄動）。依 leaf 分別累加 support / critique 兩軸的 decay 加權。
        leaf_support:   Dict[str, float] = {}
        leaf_critique:  Dict[str, float] = {}
        leaf_min_depth: Dict[str, int]   = {}

        for raw in raw_paths:
            leaf_eid = raw["final_concept"]["element_id"]
            decay    = _decay(raw["depth"])
            if raw["path_type"] == "support":
                leaf_support[leaf_eid] = leaf_support.get(leaf_eid, 0.0) + decay
            elif raw["path_type"] == "critique":
                leaf_critique[leaf_eid] = leaf_critique.get(leaf_eid, 0.0) + decay
            # "other" 兩軸皆不貢獻
            leaf_min_depth[leaf_eid] = min(
                leaf_min_depth.get(leaf_eid, raw["depth"]), raw["depth"]
            )

        # ── Display 用：既有多樣性選取，只決定畫面上呈現哪幾條路徑
        selected_paths = self._selector.select(raw_paths, max_branches)

        seen_leaf_eids:    set                  = set()
        final_foundations: List[Dict[str, Any]] = []
        tree_nodes:        Dict[str, dict]       = {}
        tree_edges:        Dict[str, dict]       = {}

        for path in selected_paths:
            steps = path.get("steps") or []
            if not steps:
                continue

            leaf     = steps[0]["from_concept"]
            leaf_eid = leaf["element_id"]
            if leaf_eid not in seen_leaf_eids:
                seen_leaf_eids.add(leaf_eid)
                s_val = leaf_support.get(leaf_eid, 0.0)
                c_val = leaf_critique.get(leaf_eid, 0.0)
                dominant_type = (
                    "support" if s_val > c_val
                    else "critique" if c_val > s_val
                    else "other"
                )
                final_foundations.append({
                    "element_id":      leaf_eid,
                    "label":           leaf.get("name") or leaf_eid,
                    "depth":           leaf_min_depth.get(leaf_eid, len(steps)),
                    "dominant_type":   dominant_type,
                    "support_weight":  round(s_val, 6),
                    "critique_weight": round(c_val, 6),
                })

            for step in steps:
                for role, type_key in (("to_concept", "to_type"), ("from_concept", "from_type")):
                    c   = step[role]
                    eid = c["element_id"]
                    if eid not in tree_nodes:
                        tree_nodes[eid] = {
                            "element_id": eid,
                            "label":      c.get("name") or eid,
                            "node_type":  step[type_key],  # first-seen 為準（純展示）
                        }

                fc, tc   = step["from_concept"], step["to_concept"]
                edge_key = f"{fc['element_id']}__{tc['element_id']}"
                if edge_key not in tree_edges:
                    tree_edges[edge_key] = {
                        "source":    fc["element_id"],
                        "target":    tc["element_id"],
                        "edge_type": step.get("edge_type", "support"),
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
        start_name: str = "",
    ) -> List[Dict[str, Any]]:
        """向上游追溯，透過 cache 取鄰接關係，沿路徑做 sign propagation。

        queue tuple 多帶 cur_type：RC 自身 base case = "support"。
        一旦 cur_type == "other"，代表這條路徑之後子孫必為 other
        （對 score 無貢獻），直接標記為 completed、不再展開，
        避免白打 Neo4j。
        """
        start_concept = {"element_id": start_eid, "name": start_name or "", "type": None}
        queue         = deque([(start_concept, [], frozenset([start_eid]), "support")])
        completed:    List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited, cur_type = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth or cur_type == "other":
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   None,
                    "path_type":     cur_type,
                    "depth":         depth,
                })
                continue

            neighbors: List[UpstreamNeighbor] = self._cache.get_upstream_neighbors(cur_eid)
            new_neighbors = [n for n in neighbors if n.from_concept.element_id not in visited]

            if not new_neighbors:
                completed.append({
                    "steps":         steps,
                    "final_concept": cur_c,
                    "final_paper":   None,
                    "path_type":     cur_type,
                    "depth":         depth,
                })
                continue

            for neighbor in new_neighbors:
                prev_c    = _concept_to_dict(neighbor.from_concept)
                prev_type = _combine(cur_type, neighbor.edge_type)
                new_step = {
                    "from_concept": prev_c,
                    "via_paper":    _paper_to_dict(neighbor.via_paper),
                    "to_concept":   cur_c,
                    "edge_type":    neighbor.edge_type,
                    "to_type":      cur_type,    # to_concept 在此步之前已確定的 type
                    "from_type":    prev_type,   # 沿此邊推出 from_concept 的 type
                }
                queue.append((
                    prev_c,
                    [new_step] + steps,
                    visited | {neighbor.from_concept.element_id},
                    prev_type,
                ))

        return [r for r in completed if r["steps"]]
