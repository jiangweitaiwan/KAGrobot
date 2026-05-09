# -*- coding: utf-8 -*-
"""Neo4jReasoningRetriever

Public API
----------
retrieve_candidates(query, top_k)
    SBERT 語意搜尋，回傳 top-k matched concept 摘要。
    僅在 socratic 階段使用；不走 Neo4j chain。
    使用 name + function embedding 做匹配。

retrieve_graph(start_eid, rc_name, max_depth)
    以已知 element_id 展開雙向 BFS，回傳 {nodes, edges, paths}。
    同時支援：
      A 類圖：(s)-[:REASONING]->(t)  直接 concept→concept
      B 類圖：s -[:INSPIRES|INDIRECT_INSPIRES|REASONING]-> Paper -[:PRODUCES]-> t
    paths 含 direction="upstream" | "downstream"。

retrieve_concept_detail(rc_name)
    查詢指定 concept 的 description 以及所屬 paper 的
    conceptual_premises / contextual_model_constraints。
    供 SocratesChatBot concept_tool 使用（fallback，優先從 all_graphs lookup）。

project_concepts_only(graph_data)
    靜態方法。將完整 graph_data 投影為只含 concept 節點與
    concept-to-concept 邊的精簡結構，供前端與 graph_context 使用。

Notes
-----
_embedding_index 每筆 variant 同時儲存兩種向量：
  retrieval_embedding  — name + function，供 _match_entities (retrieve_candidates) 使用
  similarity_embedding — name + description，供 _calc_pair (processing_pipeline) 使用
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from transformers import AutoTokenizer
import onnxruntime as ort
from neo4j import GraphDatabase


# ============================================================
# SBERT ONNX Encoder
# ============================================================

class SbertOnnxEncoder:
    def __init__(self, model_path: str, tokenizer_dir: str):
        opts = ort.SessionOptions()
        opts.enable_mem_pattern   = False
        opts.enable_cpu_mem_arena = False
        self.session   = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def tokenize(self, text: str, max_length: int = 256):
        enc = self.tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="np",
        )
        return enc["input_ids"], enc["attention_mask"]

    def mean_pooling(self, token_embeddings: np.ndarray,
                     attention_mask: np.ndarray) -> np.ndarray:
        mask     = attention_mask[..., None].astype(np.float32)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return (token_embeddings * mask).sum(axis=1) / sum_mask

    def encode(self, sentences: Union[str, List[str]],
               normalize: bool = True) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        ids_list, mask_list = [], []
        for s in sentences:
            ids, mask = self.tokenize(s)
            ids_list.append(ids)
            mask_list.append(mask)
        input_ids      = np.vstack(ids_list).astype(np.int64)
        attention_mask = np.vstack(mask_list).astype(np.int64)
        outputs        = self.session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask,
        })
        embeddings = self.mean_pooling(outputs[0], attention_mask)
        if normalize:
            norms      = np.clip(
                np.linalg.norm(embeddings, axis=1, keepdims=True),
                a_min=1e-9, a_max=None)
            embeddings = embeddings / norms
        return embeddings


# ============================================================
# Data Classes（僅供內部 BFS 使用）
# ============================================================

ConceptKind = Literal["resulting_concept", "sub_concept", "other"]


@dataclass
class ConceptNode:
    element_id: str
    name: str
    type: Optional[str]


@dataclass
class PaperNode:
    element_id: str
    title: str
    conceptual_premises: Optional[str]
    contextual_model_constraints: Optional[str]


# ============================================================
# Neo4jReasoningRetriever
# ============================================================

class Neo4jReasoningRetriever:

    def __init__(self, uri: str, password: str,
                 model_path: str, tokenizer_dir: str):
        self.driver  = GraphDatabase.driver(uri, auth=("neo4j", password))
        self.encoder = SbertOnnxEncoder(model_path, tokenizer_dir)
        self._embedding_index: List[Dict[str, Any]] = []
        self._embeddings_loaded = False

    def close(self):
        self.driver.close()

    # ----------------------------------------------------------
    # Embedding loading
    # ----------------------------------------------------------

    def load_embeddings(self):
        self._load_concept_embeddings()

    def _load_concept_embeddings(self):
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE c.type = 'Resulting_Concept'
                RETURN elementId(c)           AS eid,
                       c.name                 AS name,
                       c.type                 AS type,
                       c.description          AS description,
                       c.function             AS function,
                       c.analysis_types       AS analysis_types,
                       c.merged_names         AS merged_names,
                       c.merged_descriptions  AS merged_descriptions,
                       c.merged_functions     AS merged_functions
                """
            ).data()

        variants: List[Dict[str, Any]] = []
        for row in rows:
            eid            = row["eid"]
            node_name      = row["name"]
            node_type      = row["type"]
            node_func      = row["function"] or ""
            node_atypes    = row["analysis_types"] or []
            m_names        = row["merged_names"]
            m_descs        = row["merged_descriptions"]
            m_funcs        = row["merged_functions"]

            if m_names and len(m_names) > 0:
                for i, mn in enumerate(m_names):
                    md = m_descs[i] if (m_descs and i < len(m_descs)) else ""
                    mf = m_funcs[i] if (m_funcs and i < len(m_funcs)) else node_func
                    variants.append({
                        "element_id":          eid,
                        "node_name":           node_name,
                        "node_type":           node_type,
                        "analysis_types":      node_atypes,
                        "variant_name":        mn,
                        "variant_description": md or "",
                        "variant_function":    mf or "",
                    })
            else:
                variants.append({
                    "element_id":          eid,
                    "node_name":           node_name,
                    "node_type":           node_type,
                    "analysis_types":      node_atypes,
                    "variant_name":        node_name or "",
                    "variant_description": row["description"] or "",
                    "variant_function":    node_func,
                })

        BATCH = 64

        # retrieval 用：name + function（供 _match_entities 使用）
        retrieval_texts = [
            f"{v['variant_name']} {v['variant_function']}".strip()
            for v in variants
        ]
        # similarity 用：name + description（供 _calc_pair / processing_pipeline 使用）
        similarity_texts = [
            f"{v['variant_name']} {v['variant_description']}".strip()
            for v in variants
        ]

        retrieval_embs:  List[np.ndarray] = []
        similarity_embs: List[np.ndarray] = []
        for i in range(0, len(variants), BATCH):
            retrieval_embs.append(self.encoder.encode(retrieval_texts[i:i + BATCH]))
            similarity_embs.append(self.encoder.encode(similarity_texts[i:i + BATCH]))

        if retrieval_embs:
            ret_matrix = np.vstack(retrieval_embs)
            sim_matrix = np.vstack(similarity_embs)
            for idx, v in enumerate(variants):
                v["retrieval_embedding"]  = ret_matrix[idx]
                v["similarity_embedding"] = sim_matrix[idx]

        self._embedding_index   = variants
        self._embeddings_loaded = True
        print(f"Loaded {len(variants)} concept variants from {len(rows)} nodes.")

    # ----------------------------------------------------------
    # Entity matching（僅供 retrieve_candidates 使用）
    # Uses retrieval_embedding (name + function)
    # ----------------------------------------------------------

    def _match_entities(self, query: str,
                        top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._embeddings_loaded:
            self._load_concept_embeddings()
        query_emb = self.encoder.encode(query)[0]
        scored: List[Dict[str, Any]] = []
        for v in self._embedding_index:
            if "retrieval_embedding" not in v:
                continue
            score = float(np.dot(query_emb, v["retrieval_embedding"]))
            scored.append({
                "element_id":     v["element_id"],
                "node_name":      v["node_name"],
                "node_type":      v["node_type"],
                "analysis_types": v.get("analysis_types", []),
                "score":          score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)

        seen: Dict[str, Dict[str, Any]] = {}
        for s in scored:
            eid = s["element_id"]
            if eid not in seen:
                seen[eid] = s
        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return deduped[:top_k]

    # ----------------------------------------------------------
    # Public: retrieve_candidates()
    # ----------------------------------------------------------

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """語意搜尋，回傳 top-k concept 摘要（使用 name+function embedding）。"""
        matches = self._match_entities(query, top_k=top_k)
        return [
            {
                "index":          i + 1,
                "eid":            m["element_id"],
                "rc_name":        m["node_name"],
                "type":           m["node_type"],
                "analysis_types": m.get("analysis_types", []),
                "score":          round(m["score"], 4),
            }
            for i, m in enumerate(matches)
        ]

    # ----------------------------------------------------------
    # Public: retrieve_concept_detail()
    # ----------------------------------------------------------

    def retrieve_concept_detail(
        self,
        rc_name: str,
    ) -> Dict[str, Any]:
        """查詢指定 concept 的描述與所屬論文的理論背景。

        優先由 SocratesChatBot 從 all_graphs 直接 lookup；
        此方法作為 fallback（all_graphs 尚未展開時）。
        """
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($name)
                OPTIONAL MATCH (p:Paper)-[:PRODUCES]->(c)
                RETURN
                    c.name                          AS name,
                    c.description                   AS description,
                    c.function                      AS function,
                    c.analysis_types                AS analysis_types,
                    collect({
                        title:       p.title,
                        premises:    p.conceptual_premises,
                        constraints: p.contextual_model_constraints
                    }) AS papers
                LIMIT 1
                """,
                name=rc_name,
            ).data()

        if not rows:
            return {
                "found":   False,
                "rc_name": rc_name,
                "message": f"找不到與「{rc_name}」相符的 concept。",
                "papers":  [],
            }

        row = rows[0]
        papers = [
            {
                "title":       p["title"],
                "premises":    p["premises"],
                "constraints": p["constraints"],
            }
            for p in (row["papers"] or [])
            if p.get("title")
        ]

        return {
            "found":          True,
            "rc_name":        row["name"],
            "description":    row["description"],
            "function":       row["function"],
            "analysis_types": row["analysis_types"] or [],
            "papers":         papers,
        }

    # ----------------------------------------------------------
    # Public: retrieve_graph()
    # ----------------------------------------------------------

    def retrieve_graph(
        self,
        start_eid:          str,
        rc_name:            str = "",
        max_depth_down:     int = 8,
        max_depth_up:       int = 3,
    ) -> Dict[str, Any]:
        """以 element_id 展開雙向 BFS，回傳 D3 渲染用結構。

        同時支援：
          A 類圖：(s)-[:REASONING]->(t)
          B 類圖：s -[:INSPIRES|INDIRECT_INSPIRES|REASONING]-> Paper -[:PRODUCES]-> t

        Parameters
        ----------
        start_eid       : Neo4j elementId，由 retrieve_candidates 取得。
        rc_name         : 僅作 label 用。
        max_depth_down  : 下游 BFS 最大深度（預設 8）。
        max_depth_up    : 上游 BFS 最大深度（預設 3）。

        Returns
        -------
        {
          "query_concepts": [{"id", "name", "type", "score"}],
          "nodes":  [...],   # 完整，含 paper meta
          "edges":  [...],
          "paths":  [...],   # 每條 path 含 direction="upstream"|"downstream"
        }
        """
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        all_paths: List[dict]      = []

        self._upsert_concept_node(
            nodes_map, start_eid, rc_name, "Resulting_Concept", matched=True
        )

        with self.driver.session() as session:
            kind = session.execute_read(self._detect_concept_kind, start_eid)

            if kind == "resulting_concept":
                effective_eid = start_eid
            elif kind == "sub_concept":
                parent_rc = session.execute_read(
                    self._find_parent_resulting_concept, start_eid
                )
                if parent_rc:
                    self._upsert_concept_node(
                        nodes_map, parent_rc.element_id,
                        parent_rc.name, parent_rc.type
                    )
                    effective_eid = parent_rc.element_id
                else:
                    effective_eid = start_eid
            else:
                effective_eid = None

            # ── 下游 BFS ──────────────────────────────────────
            if effective_eid:
                raw_down = session.execute_read(
                    self._walk_downstream, effective_eid, max_depth_down
                )
            else:
                raw_down = session.execute_read(
                    self._walk_downstream_from_other, start_eid, max_depth_down
                )

            for raw in raw_down:
                self._integrate_path(raw, nodes_map, edges_map, all_paths,
                                     direction="downstream")

            # ── 上游 BFS ──────────────────────────────────────
            raw_up = session.execute_read(
                self._walk_upstream, start_eid, max_depth_up
            )
            for raw in raw_up:
                self._integrate_path(raw, nodes_map, edges_map, all_paths,
                                     direction="upstream")

        # 去重（相同 node_ids 序列視為同一條路徑）
        seen_sigs: set = set()
        unique_paths: List[dict] = []
        for p in all_paths:
            sig = (p["direction"], tuple(p["node_ids"]))
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)

        # 補充 analysis_types 到 concept 節點（供 path scoring 使用）
        self._enrich_nodes_with_analysis_types(nodes_map)

        return {
            "query_concepts": [{
                "id":    start_eid,
                "name":  rc_name,
                "type":  "Resulting_Concept",
                "score": 1.0,
            }],
            "nodes": list(nodes_map.values()),
            "edges": list(edges_map.values()),
            "paths": unique_paths,
        }

    def _enrich_nodes_with_analysis_types(
        self, nodes_map: Dict[str, dict]
    ) -> None:
        """從 _embedding_index 補充 analysis_types 到 concept 節點。"""
        eid_to_types: Dict[str, List[str]] = {
            v["element_id"]: v.get("analysis_types", [])
            for v in self._embedding_index
        }
        for node in nodes_map.values():
            if node.get("node_type") == "concept":
                node.setdefault(
                    "analysis_types",
                    eid_to_types.get(node["id"], []),
                )

    # ----------------------------------------------------------
    # Public: project_concepts_only()
    # ----------------------------------------------------------

    @staticmethod
    def project_concepts_only(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """將完整 graph_data 投影為只含 concept 節點與 concept-to-concept 邊。"""
        nodes_map  = {n["id"]: n for n in graph_data.get("nodes", [])}
        concept_ids = {
            nid for nid, n in nodes_map.items()
            if n.get("node_type") == "concept"
        }

        proj_nodes = [
            n for n in graph_data.get("nodes", [])
            if n.get("node_type") == "concept"
        ]

        # 投影邊
        proj_edges: Dict[str, dict] = {}

        edges_by_source: Dict[str, List[dict]] = {}
        for e in graph_data.get("edges", []):
            edges_by_source.setdefault(e["source"], []).append(e)

        for e in graph_data.get("edges", []):
            src, tgt = e["source"], e["target"]

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
                                "paper_title":     nodes_map.get(paper_eid, {}).get("label", ""),
                                "via_paper_title": nodes_map.get(paper_eid, {}).get("label"),
                            }

        # 投影 paths
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
                eid = f"{concept_node_ids[i]}__{concept_node_ids[i+1]}"
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

        return {
            "query_concepts":   graph_data.get("query_concepts", []),
            "nodes":            proj_nodes,
            "edges":            list(proj_edges.values()),
            "paths":            proj_paths,
            "selected_path_id": graph_data.get("selected_path_id"),
        }

    # ----------------------------------------------------------
    # Graph assembly helpers
    # ----------------------------------------------------------

    @staticmethod
    def _upsert_concept_node(
        nodes_map: Dict[str, dict],
        eid: str, name: str, ctype: Optional[str],
        matched: bool = False,
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

    @staticmethod
    def _upsert_paper_node(
        nodes_map: Dict[str, dict],
        eid: str, title: str,
        premises: Optional[str]    = None,
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

    @staticmethod
    def _upsert_edge(
        edges_map: Dict[str, dict],
        source: str, target: str,
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
        self,
        raw: dict,
        nodes_map: Dict[str, dict],
        edges_map: Dict[str, dict],
        all_paths: List[dict],
        direction: str = "downstream",
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
            self._upsert_concept_node(
                nodes_map,
                first_c["element_id"], first_c["name"], first_c.get("type"))
            node_ids.append(first_c["element_id"])

            for step in steps:
                paper = step.get("via_paper")
                to_c  = step["to_concept"]

                if paper:
                    self._upsert_paper_node(
                        nodes_map,
                        paper["element_id"], paper["title"],
                        paper.get("conceptual_premises"),
                        paper.get("contextual_model_constraints"),
                    )
                    self._upsert_concept_node(
                        nodes_map,
                        to_c["element_id"], to_c["name"], to_c.get("type"))

                    e1 = self._upsert_edge(
                        edges_map,
                        step["from_concept"]["element_id"],
                        paper["element_id"],
                        paper["title"],
                    )
                    e2 = self._upsert_edge(
                        edges_map,
                        paper["element_id"],
                        to_c["element_id"],
                        paper["title"],
                    )
                    node_ids += [paper["element_id"], to_c["element_id"]]
                    edge_ids += [e1, e2]
                else:
                    self._upsert_concept_node(
                        nodes_map,
                        to_c["element_id"], to_c["name"], to_c.get("type"))
                    e1 = self._upsert_edge(
                        edges_map,
                        step["from_concept"]["element_id"],
                        to_c["element_id"],
                        "",
                    )
                    node_ids.append(to_c["element_id"])
                    edge_ids.append(e1)
        else:
            self._upsert_concept_node(
                nodes_map,
                final_c["element_id"], final_c["name"], final_c.get("type"))
            node_ids.append(final_c["element_id"])

        if final_p:
            last_paper_eid = None
            if steps:
                last_via = steps[-1].get("via_paper")
                if last_via:
                    last_paper_eid = last_via["element_id"]

            if final_p["element_id"] != last_paper_eid:
                self._upsert_paper_node(
                    nodes_map,
                    final_p["element_id"], final_p["title"],
                    final_p.get("conceptual_premises"),
                    final_p.get("contextual_model_constraints"),
                )
                efp = self._upsert_edge(
                    edges_map,
                    final_c["element_id"],
                    final_p["element_id"],
                    final_p["title"],
                )
                node_ids.append(final_p["element_id"])
                edge_ids.append(efp)

        all_paths.append({
            "path_id":       len(all_paths),
            "direction":     direction,
            "node_ids":      node_ids,
            "edge_ids":      edge_ids,
            "final_concept": final_c["element_id"],
            "final_label":   final_c.get("name", ""),
            "intent_score":  0.0,   # 由 entry.py _score_paths_by_intent 填入
        })

    # ----------------------------------------------------------
    # Concept kind detection
    # ----------------------------------------------------------

    @staticmethod
    def _detect_concept_kind(tx, eid: str) -> ConceptKind:
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
            """, eid=eid,
        ).single()
        return rec["kind"] if rec else "other"

    # ----------------------------------------------------------
    # BFS: _walk_downstream
    # ----------------------------------------------------------

    @staticmethod
    def _walk_downstream(tx, start_eid: str,
                         max_depth: int = 8) -> List[Dict[str, Any]]:
        rec0 = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
                                AND c.type = 'Resulting_Concept'
            RETURN c, elementId(c) AS eid
            """, eid=start_eid,
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
        queue = deque([(start_concept, [], frozenset([start_eid]))])
        completed: List[Dict[str, Any]] = []

        while queue:
            cur_c, steps, visited = queue.popleft()
            cur_eid = cur_c["element_id"]
            depth   = len(steps)

            if depth >= max_depth:
                fp = _fallback_paper(tx, cur_eid)
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": fp})
                continue

            a_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                MATCH (c)-[:REASONING]->(next:Concept)
                WHERE elementId(next) <> $eid
                RETURN next, elementId(next) AS next_eid
                """, eid=cur_eid,
            ).data()

            b_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                                    AND c.type = 'Resulting_Concept'
                MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|REASONING]->(p:Paper)
                MATCH (p)-[:PRODUCES]->(next:Concept {type:'Resulting_Concept'})
                WHERE elementId(next) <> $eid
                RETURN p, next,
                       elementId(p)    AS p_eid,
                       elementId(next) AS next_eid
                """, eid=cur_eid,
            ).data()

            new_a = [b for b in a_branches if b["next_eid"] not in visited]
            new_b = [b for b in b_branches if b["next_eid"] not in visited]

            if not new_a and not new_b:
                fp = _fallback_paper(tx, cur_eid)
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": fp})
                continue

            for branch in new_a:
                next_c   = _c(branch["next_eid"], branch["next"])
                new_step = {"from_concept": cur_c, "via_paper": None, "to_concept": next_c}
                queue.append((next_c, steps + [new_step], visited | {branch["next_eid"]}))

            for branch in new_b:
                paper    = _p(branch["p_eid"], branch["p"])
                next_c   = _c(branch["next_eid"], branch["next"])
                new_step = {"from_concept": cur_c, "via_paper": paper, "to_concept": next_c}
                queue.append((next_c, steps + [new_step], visited | {branch["next_eid"]}))

        return completed

    # ----------------------------------------------------------
    # BFS: _walk_upstream
    # ----------------------------------------------------------

    @staticmethod
    def _walk_upstream(tx, start_eid: str,
                       max_depth: int = 3) -> List[Dict[str, Any]]:
        rec0 = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            RETURN c, elementId(c) AS eid
            """, eid=start_eid,
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
        queue = deque([(start_concept, [], frozenset([start_eid]))])
        completed: List[Dict[str, Any]] = []

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
                MATCH (prev:Concept)-[:REASONING]->(c)
                WHERE elementId(prev) <> $eid
                RETURN prev, elementId(prev) AS prev_eid
                """, eid=cur_eid,
            ).data()

            b_branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                MATCH (p:Paper)-[:PRODUCES]->(c)
                MATCH (prev:Concept)-[:INSPIRES|INDIRECT_INSPIRES|REASONING]->(p)
                WHERE elementId(prev) <> $eid
                RETURN prev, p,
                       elementId(prev) AS prev_eid,
                       elementId(p)    AS p_eid
                """, eid=cur_eid,
            ).data()

            new_a = [b for b in a_branches if b["prev_eid"] not in visited]
            new_b = [b for b in b_branches if b["prev_eid"] not in visited]

            if not new_a and not new_b:
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": None})
                continue

            for branch in new_a:
                prev_c   = _c(branch["prev_eid"], branch["prev"])
                new_step = {"from_concept": prev_c, "via_paper": None, "to_concept": cur_c}
                queue.append((prev_c, [new_step] + steps, visited | {branch["prev_eid"]}))

            for branch in new_b:
                prev_c   = _c(branch["prev_eid"], branch["prev"])
                paper    = _p(branch["p_eid"], branch["p"])
                new_step = {"from_concept": prev_c, "via_paper": paper, "to_concept": cur_c}
                queue.append((prev_c, [new_step] + steps, visited | {branch["prev_eid"]}))

        return [r for r in completed if r["steps"]]

    # ----------------------------------------------------------
    # BFS: _walk_downstream_from_other
    # ----------------------------------------------------------

    @staticmethod
    def _walk_downstream_from_other(tx, eid: str,
                                    max_depth: int = 8) -> List[Dict[str, Any]]:
        rows = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|APPLY_TO|REASONING]->(p:Paper)
            MATCH (p)-[:PRODUCES]->(rc:Concept {type:'Resulting_Concept'})
            RETURN DISTINCT elementId(rc) AS rc_eid
            """, eid=eid,
        ).data()
        all_paths: List[Dict[str, Any]] = []
        for row in rows:
            sub = Neo4jReasoningRetriever._walk_downstream(
                tx, row["rc_eid"], max_depth)
            all_paths.extend(sub)
        return all_paths

    # ----------------------------------------------------------
    # Helper: find parent RC for sub_concept
    # ----------------------------------------------------------

    @staticmethod
    def _find_parent_resulting_concept(tx, eid: str) -> Optional[ConceptNode]:
        rec = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (rc:Concept {type:'Resulting_Concept'})-[:CONTAINS]->(c)
            RETURN rc, elementId(rc) AS rc_eid LIMIT 1
            """, eid=eid,
        ).single()
        if not rec:
            return None
        rc = rec["rc"]
        return ConceptNode(element_id=rec["rc_eid"], name=rc.get("name"),
                           type=rc.get("type"))


# ----------------------------------------------------------------
# Module-level helper
# ----------------------------------------------------------------

def _fallback_paper(tx, eid: str) -> Optional[dict]:
    rec = tx.run(
        """
        MATCH (p:Paper)-[:PRODUCES]->(rc:Concept) WHERE elementId(rc) = $eid
        RETURN p, elementId(p) AS p_eid LIMIT 1
        """, eid=eid,
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
