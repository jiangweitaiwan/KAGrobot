# -*- coding: utf-8 -*-
"""graph_service/projector.py — 圖譜投影工具。

將含 Paper 節點的完整 graph_data 投影為只含
Concept 節點與 Concept-to-Concept 邊的精簡結構，
供前端 KnowledgeGraph 渲染與 graph_context 組裝使用。
"""

from __future__ import annotations

from typing import Any, Dict, List


class GraphProjector:
    """將完整圖譜投影為純 Concept 結構（無 Paper 節點）。"""

    def project(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """投影並回傳精簡圖譜。foundations 與 foundation_tree 直接透傳。"""
        nodes_map   = {n["id"]: n for n in graph_data.get("nodes", [])}
        concept_ids = {
            nid for nid, n in nodes_map.items()
            if n.get("node_type") == "concept"
        }

        proj_nodes = [
            n for n in graph_data.get("nodes", [])
            if n.get("node_type") == "concept"
        ]

        proj_edges = self._build_projected_edges(
            graph_data, nodes_map, concept_ids
        )

        proj_paths = self._build_projected_paths(
            graph_data, concept_ids, proj_edges
        )

        return {
            "query_concepts":   graph_data.get("query_concepts", []),
            "nodes":            proj_nodes,
            "edges":            list(proj_edges.values()),
            "paths":            proj_paths,
            "selected_path_id": graph_data.get("selected_path_id"),
            "foundations":      graph_data.get("foundations", []),
            "foundation_tree":  graph_data.get(
                "foundation_tree", {"nodes": [], "edges": []}
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_projected_edges(
        graph_data:  Dict[str, Any],
        nodes_map:   Dict[str, Any],
        concept_ids: set,
    ) -> Dict[str, dict]:
        edges_by_source: Dict[str, List[dict]] = {}
        for e in graph_data.get("edges", []):
            edges_by_source.setdefault(e["source"], []).append(e)

        proj_edges: Dict[str, dict] = {}

        for e in graph_data.get("edges", []):
            src, tgt = e["source"], e["target"]

            # Concept → Concept 直接邊
            if src in concept_ids and tgt in concept_ids:
                key = f"{src}__{tgt}"
                if key not in proj_edges:
                    proj_edges[key] = {
                        "id":              key,
                        "source":          src,
                        "target":          tgt,
                        "paper_title":     e.get("paper_title", ""),
                        "via_paper_title": None,
                    }

            # Concept → Paper → Concept（跨 Paper 橋接）
            elif src in concept_ids and tgt not in concept_ids:
                paper_eid = tgt
                for e2 in edges_by_source.get(paper_eid, []):
                    next_eid = e2["target"]
                    if next_eid in concept_ids:
                        key = f"{src}__{next_eid}"
                        if key not in proj_edges:
                            proj_edges[key] = {
                                "id":              key,
                                "source":          src,
                                "target":          next_eid,
                                "paper_title":     nodes_map.get(
                                    paper_eid, {}
                                ).get("label", ""),
                                "via_paper_title": nodes_map.get(
                                    paper_eid, {}
                                ).get("label"),
                            }

        return proj_edges

    @staticmethod
    def _build_projected_paths(
        graph_data:  Dict[str, Any],
        concept_ids: set,
        proj_edges:  Dict[str, dict],
    ) -> List[dict]:
        proj_paths: List[dict] = []

        for p in graph_data.get("paths", []):
            concept_node_ids = [
                nid for nid in p.get("node_ids", [])
                if nid in concept_ids
            ]
            if not concept_node_ids:
                continue

            proj_edge_ids: List[str] = []
            for i in range(len(concept_node_ids) - 1):
                eid = f"{concept_node_ids[i]}__{concept_node_ids[i + 1]}"
                if eid in proj_edges:
                    proj_edge_ids.append(eid)

            proj_paths.append({
                "path_id":       p["path_id"],
                "direction":     p.get("direction", "downstream"),
                "node_ids":      concept_node_ids,
                "edge_ids":      proj_edge_ids,
                "final_concept": p.get("final_concept"),
                "final_label":   p.get("final_label", ""),
                "intent_score":  p.get("intent_score", 0.0),
            })

        return proj_paths
