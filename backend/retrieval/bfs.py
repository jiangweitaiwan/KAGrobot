# -*- coding: utf-8 -*-
"""retrieval/bfs.py — 知識圖譜 BFS 遍歷。

DownstreamBFS : 從 Resulting Concept 向下游展開推理鏈。
UpstreamBFS   : 從 Resulting Concept 向上游追溯基礎概念。

兩個類別皆依賴 Neo4j driver，不持有其他狀態。
BFS 的 Neo4j transaction 方法以 @staticmethod 定義，
方便在 session.execute_read() 中直接傳入。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from retrieval.path_selector import PathSelector


# ═══════════════════════════════════════════════════════════════
# Shared data classes
# ═══════════════════════════════════════════════════════════════

ConceptKind = Literal["resulting_concept", "sub_concept", "other"]


@dataclass
class ConceptNode:
    element_id: str
    name:       str
    type:       Optional[str]


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


# ═══════════════════════════════════════════════════════════════
# DownstreamBFS
# ═══════════════════════════════════════════════════════════════

class DownstreamBFS:
    """從 Resulting Concept 向下游展開 BFS，回傳 D3 渲染用結構。"""

    def __init__(self, driver) -> None:
        self._driver = driver

    def run(
        self,
        start_eid:      str,
        rc_name:        str = "",
        max_depth_down: int = 8,
    ) -> Dict[str, Any]:
        """主要入口，對應原 retrieve_graph()。"""
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        all_paths: List[dict]      = []

        _upsert_concept_node(
            nodes_map, start_eid, rc_name, "Resulting_Concept", matched=True
        )

        with self._driver.session() as session:
            kind = session.execute_read(self._detect_kind, start_eid)

            if kind == "resulting_concept":
                effective_eid = start_eid
            elif kind == "sub_concept":
                parent_rc = session.execute_read(
                    self._find_parent_rc, start_eid
                )
                if parent_rc:
                    _upsert_concept_node(
                        nodes_map,
                        parent_rc.element_id,
                        parent_rc.name,
                        parent_rc.type,
                    )
                    effective_eid = parent_rc.element_id
                else:
                    effective_eid = start_eid
            else:
                effective_eid = None

            if effective_eid:
                raw_paths = session.execute_read(
                    self._walk, effective_eid, max_depth_down
                )
            else:
                raw_paths = session.execute_read(
                    self._walk_from_other, start_eid, max_depth_down
                )

            for raw in raw_paths:
                _integrate_path(raw, nodes_map, edges_map, all_paths,
                                direction="downstream")

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

    # ------------------------------------------------------------------
    # Neo4j transaction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_kind(tx, eid: str) -> ConceptKind:
        rec = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            OPTIONAL MATCH (rc:Concept {type:'Resulting_Concept'})-[:CONTAINS]->(c)
            WITH c, collect(rc) AS parents
            RETURN
              CASE
                WHEN c.type = 'Resulting_Concept' THEN 'resulting_concept'
                WHEN size(parents) > 0            THEN 'sub_concept'
                ELSE 'other'
              END AS kind
            """,
            eid=eid,
        ).single()
        return rec["kind"] if rec else "other"

    @staticmethod
    def _find_parent_rc(tx, eid: str) -> Optional[ConceptNode]:
        rec = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (rc:Concept {type:'Resulting_Concept'})-[:CONTAINS]->(c)
            RETURN rc, elementId(rc) AS rc_eid LIMIT 1
            """,
            eid=eid,
        ).single()
        if not rec:
            return None
        rc = rec["rc"]
        return ConceptNode(
            element_id=rec["rc_eid"],
            name=rc.get("name"),
            type=rc.get("type"),
        )

    @staticmethod
    def _walk(
        tx,
        start_eid: str,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        rec0 = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
                                AND c.type = 'Resulting_Concept'
            RETURN c, elementId(c) AS eid
            """,
            eid=start_eid,
        ).single()
        if not rec0:
            return []

        def _c(eid, node):
            return {"element_id": eid,
                    "name": node.get("name"), "type": node.get("type")}

        def _p(eid, node):
            return {
                "element_id":                   eid,
                "title":                        node.get("title"),
                "conceptual_premises":          node.get("conceptual_premises"),
                "contextual_model_constraints": node.get("contextual_model_constraints"),
            }

        start_concept = _c(rec0["eid"], rec0["c"])
        queue         = deque([(start_concept, [], frozenset([start_eid]))])
        completed:    List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth:
                fp = DownstreamBFS._fallback_paper(tx, cur_eid)
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": fp})
                continue

            a_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                MATCH (c)-[:COMPOSES]->(next:Concept)
                WHERE elementId(next) <> $eid
                RETURN next, elementId(next) AS next_eid
                """,
                eid=cur_eid,
            ).data()

            b_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                                    AND c.type = 'Resulting_Concept'
                MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->(p:Paper)
                MATCH (p)-[:PRODUCES]->(next:Concept {type:'Resulting_Concept'})
                WHERE elementId(next) <> $eid
                RETURN p, next,
                       elementId(p)    AS p_eid,
                       elementId(next) AS next_eid
                """,
                eid=cur_eid,
            ).data()

            new_a = [b for b in a_branches if b["next_eid"] not in visited]
            new_b = [b for b in b_branches if b["next_eid"] not in visited]

            if not new_a and not new_b:
                fp = DownstreamBFS._fallback_paper(tx, cur_eid)
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": fp})
                continue

            for branch in new_a:
                next_c   = _c(branch["next_eid"], branch["next"])
                new_step = {"from_concept": cur_c, "via_paper": None,
                            "to_concept": next_c}
                queue.append(
                    (next_c, steps + [new_step], visited | {branch["next_eid"]}))

            for branch in new_b:
                paper    = _p(branch["p_eid"], branch["p"])
                next_c   = _c(branch["next_eid"], branch["next"])
                new_step = {"from_concept": cur_c, "via_paper": paper,
                            "to_concept": next_c}
                queue.append(
                    (next_c, steps + [new_step], visited | {branch["next_eid"]}))

        return completed

    @staticmethod
    def _walk_from_other(
        tx,
        eid:       str,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        rows = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|APPLY_TO|COMPOSES]->(p:Paper)
            MATCH (p)-[:PRODUCES]->(rc:Concept {type:'Resulting_Concept'})
            RETURN DISTINCT elementId(rc) AS rc_eid
            """,
            eid=eid,
        ).data()
        all_paths: List[Dict[str, Any]] = []
        for row in rows:
            sub = DownstreamBFS._walk(tx, row["rc_eid"], max_depth)
            all_paths.extend(sub)
        return all_paths

    @staticmethod
    def _fallback_paper(tx, eid: str) -> Optional[dict]:
        rec = tx.run(
            """
            MATCH (p:Paper)-[:PRODUCES]->(rc:Concept) WHERE elementId(rc) = $eid
            RETURN p, elementId(p) AS p_eid LIMIT 1
            """,
            eid=eid,
        ).single()
        if not rec:
            return None
        p = rec["p"]
        return {
            "element_id":                   rec["p_eid"],
            "title":                        p.get("title"),
            "conceptual_premises":          p.get("conceptual_premises"),
            "contextual_model_constraints": p.get("contextual_model_constraints"),
        }


# ═══════════════════════════════════════════════════════════════
# UpstreamBFS
# ═══════════════════════════════════════════════════════════════

class UpstreamBFS:
    """從 Resulting Concept 向上游追溯基礎概念。"""

    def __init__(self, driver, path_selector: PathSelector) -> None:
        self._driver   = driver
        self._selector = path_selector

    def run(
        self,
        start_eid:    str,
        max_depth:    int = 10,
        max_branches: int = 4,
    ) -> Dict[str, Any]:
        """主要入口，對應原 retrieve_foundations()。

        Returns
        -------
        {
            "final_foundations": [{"element_id": str, "label": str}, ...],
            "foundation_tree": {
                "nodes": [...],
                "edges": [...]
            }
        }
        """
        with self._driver.session() as session:
            raw_paths = session.execute_read(
                self._walk, start_eid, max_depth
            )

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

                fc      = step["from_concept"]
                tc      = step["to_concept"]
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

    # ------------------------------------------------------------------
    # Neo4j transaction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(
        tx,
        start_eid: str,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        rec0 = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            RETURN c, elementId(c) AS eid
            """,
            eid=start_eid,
        ).single()
        if not rec0:
            return []

        def _c(eid, node):
            return {"element_id": eid,
                    "name": node.get("name"), "type": node.get("type")}

        def _p(eid, node):
            return {
                "element_id":                   eid,
                "title":                        node.get("title"),
                "conceptual_premises":          node.get("conceptual_premises"),
                "contextual_model_constraints": node.get("contextual_model_constraints"),
            }

        start_concept = _c(rec0["eid"], rec0["c"])
        queue         = deque([(start_concept, [], frozenset([start_eid]))])
        completed:    List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth:
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": None})
                continue

            a_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                MATCH (prev:Concept)-[:COMPOSES]->(c)
                WHERE elementId(prev) <> $eid
                RETURN prev, elementId(prev) AS prev_eid
                """,
                eid=cur_eid,
            ).data()

            b_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                MATCH (p:Paper)-[:PRODUCES]->(c)
                MATCH (prev:Concept)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->(p)
                WHERE elementId(prev) <> $eid
                RETURN prev, p,
                       elementId(prev) AS prev_eid,
                       elementId(p)    AS p_eid
                """,
                eid=cur_eid,
            ).data()

            new_a = [b for b in a_branches if b["prev_eid"] not in visited]
            new_b = [b for b in b_branches if b["prev_eid"] not in visited]

            if not new_a and not new_b:
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": None})
                continue

            for branch in new_a:
                prev_c   = _c(branch["prev_eid"], branch["prev"])
                new_step = {"from_concept": prev_c, "via_paper": None,
                            "to_concept": cur_c}
                queue.append(
                    (prev_c, [new_step] + steps, visited | {branch["prev_eid"]}))

            for branch in new_b:
                prev_c   = _c(branch["prev_eid"], branch["prev"])
                paper    = _p(branch["p_eid"], branch["p"])
                new_step = {"from_concept": prev_c, "via_paper": paper,
                            "to_concept": cur_c}
                queue.append(
                    (prev_c, [new_step] + steps, visited | {branch["prev_eid"]}))

        return [r for r in completed if r["steps"]]
