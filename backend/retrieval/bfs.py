# -*- coding: utf-8 -*-
"""retrieval/bfs.py — 知識圖譜 BFS 共用 helper。

提供 graph-assembly helper 函式（_upsert_concept_node, _upsert_paper_node,
_upsert_edge, _integrate_path），供 graph_service.bfs_engine 使用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# Shared graph-assembly helpers（module-level，供兩個 BFS 共用）
# ═══════════════════════════════════════════════════════════════

def _upsert_concept_node(
    nodes_map: Dict[str, dict],
    eid:       str,
    name:      str,
    ctype:     Optional[str],
    matched:   bool = False,
) -> None:
    if eid not in nodes_map:
        nodes_map[eid] = {
            "id":             eid,
            "label":          name or eid,
            "node_type":      "concept",
            "concept_type":   ctype,
            "matched":        matched,
            "analysis_types": [],
            "meta":           None,
        }
    elif matched:
        nodes_map[eid]["matched"] = True


def _upsert_paper_node(
    nodes_map:   Dict[str, dict],
    eid:         str,
    title:       str,
    premises:    Optional[str] = None,
    constraints: Optional[str] = None,
) -> None:
    if eid not in nodes_map:
        nodes_map[eid] = {
            "id":           eid,
            "label":        title or eid,
            "node_type":    "paper",
            "concept_type": None,
            "matched":      False,
            "meta": {
                "conceptual_premises":          premises,
                "contextual_model_constraints": constraints,
            },
        }


def _upsert_edge(
    edges_map:   Dict[str, dict],
    source:      str,
    target:      str,
    paper_title: str = "",
) -> str:
    eid = f"{source}__{target}"
    if eid not in edges_map:
        edges_map[eid] = {
            "id":          eid,
            "source":      source,
            "target":      target,
            "paper_title": paper_title,
        }
    return eid


def _integrate_path(
    raw:        dict,
    nodes_map:  Dict[str, dict],
    edges_map:  Dict[str, dict],
    all_paths:  List[dict],
    direction:  str = "downstream",
) -> None:
    steps   = raw.get("steps") or []
    final_c = raw.get("final_concept")
    final_p = raw.get("final_paper")

    if not final_c:
        return

    node_ids: List[str] = []
    edge_ids: List[str] = []

    if steps:
        first_c = steps[0]["from_concept"]
        _upsert_concept_node(
            nodes_map,
            first_c["element_id"], first_c["name"], first_c.get("type"),
        )
        node_ids.append(first_c["element_id"])

        for step in steps:
            paper  = step.get("via_paper")
            to_c   = step["to_concept"]
            from_c = step["from_concept"]

            if paper:
                _upsert_paper_node(
                    nodes_map,
                    paper["element_id"], paper["title"],
                    paper.get("conceptual_premises"),
                    paper.get("contextual_model_constraints"),
                )
                _upsert_concept_node(
                    nodes_map,
                    to_c["element_id"], to_c["name"], to_c.get("type"),
                )
                e1 = _upsert_edge(
                    edges_map, from_c["element_id"],
                    paper["element_id"], paper["title"],
                )
                e2 = _upsert_edge(
                    edges_map, paper["element_id"],
                    to_c["element_id"], paper["title"],
                )
                node_ids += [paper["element_id"], to_c["element_id"]]
                edge_ids += [e1, e2]
            else:
                _upsert_concept_node(
                    nodes_map,
                    to_c["element_id"], to_c["name"], to_c.get("type"),
                )
                e1 = _upsert_edge(
                    edges_map, from_c["element_id"],
                    to_c["element_id"], "",
                )
                node_ids.append(to_c["element_id"])
                edge_ids.append(e1)
    else:
        _upsert_concept_node(
            nodes_map,
            final_c["element_id"], final_c["name"], final_c.get("type"),
        )
        node_ids.append(final_c["element_id"])

    # trailing paper（若最後一步沒有 via_paper，額外掛上）
    if final_p:
        last_paper_eid = None
        if steps:
            last_via = steps[-1].get("via_paper")
            if last_via:
                last_paper_eid = last_via["element_id"]

        if final_p["element_id"] != last_paper_eid:
            _upsert_paper_node(
                nodes_map,
                final_p["element_id"], final_p["title"],
                final_p.get("conceptual_premises"),
                final_p.get("contextual_model_constraints"),
            )
            efp = _upsert_edge(
                edges_map,
                final_c["element_id"], final_p["element_id"],
                final_p["title"],
            )
            node_ids.append(final_p["element_id"])
            edge_ids.append(efp)

    final_label = final_c.get("name") or nodes_map.get(
        final_c["element_id"], {}
    ).get("label", "")

    all_paths.append({
        "path_id":       len(all_paths),
        "direction":     direction,
        "node_ids":      node_ids,
        "edge_ids":      edge_ids,
        "final_concept": final_c["element_id"],
        "final_label":   final_label,
        "intent_score":  0.0,
    })
