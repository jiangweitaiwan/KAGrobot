# -*- coding: utf-8 -*-
"""Neo4jReasoningRetriever

Public API
----------
retrieve_candidates(query, top_k)
    SBERT 語意搜尋，回傳 top-k matched concept 摘要。
    僅在 socratic 階段使用；不走 Neo4j chain。

retrieve_graph(start_eid, rc_name, max_depth)
    以已知 element_id 直接展開 BFS，回傳 {nodes, edges, paths}。
    不呼叫 SBERT，由 resolved_node 使用。

retrieve_concept_detail(rc_name)
    查詢指定 concept 的 description 以及所屬 paper 的
    conceptual_premises / contextual_model_constraints。
    供 SocratesChatBot concept_tool 使用。
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
                RETURN elementId(c)           AS eid,
                       c.name                 AS name,
                       c.type                 AS type,
                       c.description          AS description,
                       c.merged_names         AS merged_names,
                       c.merged_descriptions  AS merged_descriptions
                """
            ).data()

        variants: List[Dict[str, Any]] = []
        for row in rows:
            eid       = row["eid"]
            node_name = row["name"]
            node_type = row["type"]
            m_names   = row["merged_names"]
            m_descs   = row["merged_descriptions"]
            if m_names and len(m_names) > 0:
                for i, mn in enumerate(m_names):
                    md = m_descs[i] if (m_descs and i < len(m_descs)) else ""
                    variants.append({
                        "element_id": eid, "node_name": node_name,
                        "node_type":  node_type, "variant_name": mn,
                        "variant_description": md or "",
                    })
            else:
                variants.append({
                    "element_id": eid, "node_name": node_name,
                    "node_type":  node_type, "variant_name": node_name or "",
                    "variant_description": row["description"] or "",
                })

        BATCH = 64
        texts  = [f"{v['variant_name']} {v['variant_description']}"
                  for v in variants]
        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), BATCH):
            all_embs.append(self.encoder.encode(texts[i:i + BATCH]))

        if all_embs:
            emb_matrix = np.vstack(all_embs)
            for idx, v in enumerate(variants):
                v["embedding"] = emb_matrix[idx]

        self._embedding_index   = variants
        self._embeddings_loaded = True
        print(f"Loaded {len(variants)} concept variants from {len(rows)} nodes.")

    # ----------------------------------------------------------
    # Entity matching（僅供 retrieve_candidates 使用）
    # ----------------------------------------------------------

    def _match_entities(self, query: str,
                        top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._embeddings_loaded:
            self._load_concept_embeddings()
        query_emb = self.encoder.encode(query)[0]
        scored: List[Dict[str, Any]] = []
        for v in self._embedding_index:
            if "embedding" not in v:
                continue
            score = float(np.dot(query_emb, v["embedding"]))
            scored.append({
                "element_id": v["element_id"],
                "node_name":  v["node_name"],
                "node_type":  v["node_type"],
                "score":      score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)

        # 每個 element_id 只保留最高分的 variant
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
        """語意搜尋，回傳 top-k concept 摘要。

        Returns
        -------
        [
          {
            "index":   int,   # 1-based
            "eid":     str,   # Neo4j elementId，供 retrieve_graph 直接使用
            "rc_name": str,
            "type":    str | None,
            "score":   float,
          },
          ...
        ]
        """
        matches = self._match_entities(query, top_k=top_k)
        return [
            {
                "index":   i + 1,
                "eid":     m["element_id"],
                "rc_name": m["node_name"],
                "type":    m["node_type"],
                "score":   round(m["score"], 4),
            }
            for i, m in enumerate(matches)
        ]

    # ----------------------------------------------------------
    # Public: retrieve_concept_detail()
    # 供 SocratesChatBot concept_tool 使用。
    # ----------------------------------------------------------

    def retrieve_concept_detail(
        self,
        rc_name: str,
    ) -> Dict[str, Any]:
        """查詢指定 concept 的描述與所屬論文的理論背景。

        以 concept name 模糊匹配（CONTAINS，不區分大小寫），
        回傳第一筆匹配結果。

        Parameters
        ----------
        rc_name : Resulting Concept 名稱（允許部分匹配）。

        Returns
        -------
        {
          "found":        bool,
          "rc_name":      str,
          "description":  str | None,
          "papers": [
            {
              "title":       str,
              "premises":    str | None,
              "constraints": str | None,
            },
            ...
          ],
          "message": str   # found=False 時的說明
        }
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

        # 過濾掉 title 為 None 的空 paper 項目（OPTIONAL MATCH 無結果時產生）
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
            "found":       True,
            "rc_name":     row["name"],
            "description": row["description"],
            "papers":      papers,
        }

    # ----------------------------------------------------------
    # Public: retrieve_graph()
    # ----------------------------------------------------------

    def retrieve_graph(
        self,
        start_eid: str,
        rc_name:   str = "",
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """以 element_id 直接展開圖，回傳 D3 渲染用結構。

        Parameters
        ----------
        start_eid : Neo4j elementId，由 retrieve_candidates 取得。
        rc_name   : 僅作 label 用，不影響搜尋。
        max_depth : BFS 最大深度。

        Returns
        -------
        {
          "query_concepts": [{"id", "name", "type", "score"}],
          "nodes":  [...],
          "edges":  [...],
          "paths":  [...],
        }
        """
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        all_paths: List[dict]      = []

        self._upsert_concept_node(
            nodes_map, start_eid, rc_name, "Resulting_Concept", matched=True
        )

        with self.driver.session() as session:
            kind = session.execute_read(
                self._detect_concept_kind, start_eid
            )

            if kind == "resulting_concept":
                raw_paths = session.execute_read(
                    self._walk_all_paths, start_eid, max_depth
                )

            elif kind == "sub_concept":
                parent_rc = session.execute_read(
                    self._find_parent_resulting_concept, start_eid
                )
                raw_paths = []
                if parent_rc:
                    self._upsert_concept_node(
                        nodes_map, parent_rc.element_id,
                        parent_rc.name, parent_rc.type
                    )
                    raw_paths = session.execute_read(
                        self._walk_all_paths, parent_rc.element_id, max_depth
                    )

            else:  # "other"
                raw_paths = session.execute_read(
                    self._walk_all_paths_from_other, start_eid, max_depth
                )

            for raw in raw_paths:
                self._integrate_path(raw, nodes_map, edges_map, all_paths)

        seen_sigs: set = set()
        unique_paths: List[dict] = []
        for p in all_paths:
            sig = tuple(p["node_ids"])
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)

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
                "id":           eid,
                "label":        name or eid,
                "node_type":    "concept",
                "concept_type": ctype,
                "matched":      matched,
                "meta":         None,
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
                paper = step["via_paper"]
                to_c  = step["to_concept"]

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
                final_c["element_id"], final_c["name"], final_c.get("type"))
            node_ids.append(final_c["element_id"])

        if final_p:
            last_paper_eid = (steps[-1]["via_paper"]["element_id"]
                              if steps else None)
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
            "node_ids":      node_ids,
            "edge_ids":      edge_ids,
            "final_concept": final_c["element_id"],
            "final_label":   final_c.get("name", ""),
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
    # BFS: _walk_all_paths
    # ----------------------------------------------------------

    @staticmethod
    def _walk_all_paths(tx, start_eid: str,
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

            branches = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                                    AND c.type = 'Resulting_Concept'
                MATCH (c)-[:INSPIRES]->(p:Paper)
                MATCH (p)-[:PRODUCES]->(next:Concept {type:'Resulting_Concept'})
                WHERE elementId(next) <> $eid
                RETURN p, next,
                       elementId(p)    AS p_eid,
                       elementId(next) AS next_eid
                """, eid=cur_eid,
            ).data()

            new_branches = [b for b in branches if b["next_eid"] not in visited]

            if not new_branches:
                fp = _fallback_paper(tx, cur_eid)
                completed.append(
                    {"steps": steps, "final_concept": cur_c, "final_paper": fp})
                continue

            for branch in new_branches:
                paper    = _p(branch["p_eid"], branch["p"])
                next_c   = _c(branch["next_eid"], branch["next"])
                new_step = {
                    "from_concept": cur_c,
                    "via_paper":    paper,
                    "to_concept":   next_c,
                }
                queue.append((
                    next_c,
                    steps + [new_step],
                    visited | {branch["next_eid"]},
                ))

        return completed

    # ----------------------------------------------------------
    # BFS: _walk_all_paths_from_other
    # ----------------------------------------------------------

    @staticmethod
    def _walk_all_paths_from_other(tx, eid: str,
                                   max_depth: int = 8) -> List[Dict[str, Any]]:
        rows = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|APPLY_TO]->(p:Paper)
            MATCH (p)-[:PRODUCES]->(rc:Concept {type:'Resulting_Concept'})
            RETURN DISTINCT elementId(rc) AS rc_eid
            """, eid=eid,
        ).data()
        all_paths: List[Dict[str, Any]] = []
        for row in rows:
            sub = Neo4jReasoningRetriever._walk_all_paths(
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
    """Fetch the Paper that PRODUCES the given concept (leaf fallback)."""
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
