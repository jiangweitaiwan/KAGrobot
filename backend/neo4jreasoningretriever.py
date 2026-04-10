# -*- coding: utf-8 -*-
"""Neo4jReasoningRetriever — with full-path enumeration for visualisation.

Changes vs original:
  • _walk_all_paths()          — BFS that expands every branch instead of
                                 greedily picking the first one.
  • _walk_all_paths_from_other() — same idea for non-RC start nodes.
  • retrieve_graph()           — new public method returning
                                 {nodes, edges, paths} for D3 front-end.
  • retrieve()                 — original signature/behaviour unchanged.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from transformers import AutoTokenizer
import onnxruntime as ort
from neo4j import GraphDatabase


# ============================================================
# SBERT ONNX Encoder  (unchanged)
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
            ids_list.append(ids); mask_list.append(mask)
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
# Data Classes  (unchanged)
# ============================================================

ConceptKind = Literal["resulting_concept", "sub_concept", "other"]


@dataclass
class ConceptNode:
    element_id: str
    name: str
    type: Optional[str]
    matched_name: Optional[str]        = None
    matched_description: Optional[str] = None


@dataclass
class PaperNode:
    element_id: str
    title: str
    conceptual_premises: Optional[str]
    contextual_model_constraints: Optional[str]


@dataclass
class ReasoningStep:
    from_concept: ConceptNode
    via_paper: PaperNode
    to_concept: ConceptNode


@dataclass
class ReasoningChain:
    start_concept: ConceptNode
    steps: List[ReasoningStep]
    final_resulting_concept: ConceptNode
    final_paper: PaperNode


# ============================================================
# Neo4jReasoningRetriever
# ============================================================

class Neo4jReasoningRetriever:

    def __init__(self, uri: str, password: str,
                 model_path: str, tokenizer_dir: str):
        self.driver   = GraphDatabase.driver(uri, auth=("neo4j", password))
        self.encoder  = SbertOnnxEncoder(model_path, tokenizer_dir)
        self._embedding_index: List[Dict[str, Any]] = []
        self._embeddings_loaded = False

    def close(self):
        self.driver.close()

    # ----------------------------------------------------------
    # Embedding loading  (unchanged)
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
    # Entity matching  (unchanged)
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
                "element_id":          v["element_id"],
                "node_name":           v["node_name"],
                "node_type":           v["node_type"],
                "matched_name":        v["variant_name"],
                "matched_description": v["variant_description"],
                "score":               score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ----------------------------------------------------------
    # Public: retrieve()  — original behaviour, UNCHANGED
    # ----------------------------------------------------------

    def retrieve(self, query: str,
                 top_k_entities: int = 3) -> List[ReasoningChain]:
        matches = self._match_entities(query, top_k=top_k_entities)
        if not matches:
            return []

        seen: Dict[str, Dict[str, Any]] = {}
        for m in matches:
            eid = m["element_id"]
            if eid not in seen or m["score"] > seen[eid]["score"]:
                seen[eid] = m

        all_chains: List[ReasoningChain] = []
        with self.driver.session() as session:
            for m in seen.values():
                start = ConceptNode(
                    element_id=m["element_id"],  name=m["node_name"],
                    type=m["node_type"],          matched_name=m["matched_name"],
                    matched_description=m["matched_description"],
                )
                kind = session.execute_read(
                    self._detect_concept_kind, start.element_id)

                if kind == "resulting_concept":
                    ci = session.execute_read(
                        self._walk_resulting_chain, start.element_id)
                    if ci["final_concept"] and ci["final_paper"]:
                        all_chains.append(ReasoningChain(
                            start_concept=start, steps=ci["steps"],
                            final_resulting_concept=ci["final_concept"],
                            final_paper=ci["final_paper"],
                        ))
                elif kind == "sub_concept":
                    parent_rc = session.execute_read(
                        self._find_parent_resulting_concept, start.element_id)
                    if not parent_rc:
                        continue
                    ci = session.execute_read(
                        self._walk_resulting_chain, parent_rc.element_id)
                    if ci["final_concept"] and ci["final_paper"]:
                        all_chains.append(ReasoningChain(
                            start_concept=start, steps=ci["steps"],
                            final_resulting_concept=ci["final_concept"],
                            final_paper=ci["final_paper"],
                        ))
                else:
                    chain_list = session.execute_read(
                        self._walk_from_other_concept, start.element_id)
                    for ci in chain_list:
                        if ci["final_concept"] and ci["final_paper"]:
                            all_chains.append(ReasoningChain(
                                start_concept=start, steps=ci["steps"],
                                final_resulting_concept=ci["final_concept"],
                                final_paper=ci["final_paper"],
                            ))
        return all_chains

    # ----------------------------------------------------------
    # Public: retrieve_graph()  — NEW, for D3 visualisation
    # ----------------------------------------------------------

    def retrieve_graph(
        self,
        query:          str,
        top_k_entities: int = 3,
        max_depth:      int = 8,
    ) -> Dict[str, Any]:
        """Return a rendering-ready graph payload for the D3 force front-end.

        Schema
        ------
        {
          "query_concepts": [{"id", "name", "type", "score"}, ...],
          "nodes": [
            {
              "id":           str,   # Neo4j elementId
              "label":        str,
              "node_type":    "concept" | "paper",
              "concept_type": str | null,   # e.g. "Resulting_Concept"
              "matched":      bool,         # true = query hit this node
              "meta":         {...} | null  # paper-only extra fields
            }, ...
          ],
          "edges": [
            {"id", "source", "target", "paper_title"}, ...
          ],
          "paths": [
            {
              "path_id":       int,
              "node_ids":      [str, ...],
              "edge_ids":      [str, ...],
              "final_concept": str,   # concept node id
              "final_label":   str
            }, ...
          ]
        }
        """
        matches = self._match_entities(query, top_k=top_k_entities)
        if not matches:
            return {"query_concepts": [], "nodes": [], "edges": [], "paths": []}

        seen_match: Dict[str, Dict[str, Any]] = {}
        for m in matches:
            eid = m["element_id"]
            if eid not in seen_match or m["score"] > seen_match[eid]["score"]:
                seen_match[eid] = m

        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        all_paths: List[dict]      = []
        query_concepts: List[dict] = []

        with self.driver.session() as session:
            for m in seen_match.values():
                start_eid  = m["element_id"]
                start_name = m["node_name"]
                query_concepts.append({
                    "id":    start_eid,
                    "name":  start_name,
                    "type":  m["node_type"],
                    "score": round(m["score"], 4),
                })
                self._upsert_concept_node(
                    nodes_map, start_eid, start_name,
                    m["node_type"], matched=True)

                kind = session.execute_read(
                    self._detect_concept_kind, start_eid)

                if kind == "resulting_concept":
                    raw_paths = session.execute_read(
                        self._walk_all_paths, start_eid, max_depth)

                elif kind == "sub_concept":
                    parent_rc = session.execute_read(
                        self._find_parent_resulting_concept, start_eid)
                    raw_paths = []
                    if parent_rc:
                        self._upsert_concept_node(
                            nodes_map, parent_rc.element_id,
                            parent_rc.name, parent_rc.type)
                        raw_paths = session.execute_read(
                            self._walk_all_paths, parent_rc.element_id,
                            max_depth)

                else:  # "other"
                    raw_paths = session.execute_read(
                        self._walk_all_paths_from_other,
                        start_eid, max_depth)

                for raw in raw_paths:
                    self._integrate_path(raw, nodes_map, edges_map, all_paths)

        # deduplicate paths by node-id sequence signature
        seen_sigs: set = set()
        unique_paths: List[dict] = []
        for p in all_paths:
            sig = tuple(p["node_ids"])
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)

        return {
            "query_concepts": query_concepts,
            "nodes":          list(nodes_map.values()),
            "edges":          list(edges_map.values()),
            "paths":          unique_paths,
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
        """Merge one raw path dict into the shared nodes/edges maps and
        record a paths entry for highlight tracking."""
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

        # attach final paper node if not already the last step's paper
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
    # Concept kind detection  (unchanged)
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
    # _walk_resulting_chain  — original, kept for retrieve()
    # ----------------------------------------------------------

    @staticmethod
    def _walk_resulting_chain(tx, eid: str) -> Dict[str, Any]:
        rec0 = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
                                AND c.type = 'Resulting_Concept'
            RETURN c, elementId(c) AS eid
            """, eid=eid,
        ).single()
        if not rec0:
            return {"steps": [], "final_concept": None, "final_paper": None}

        c0  = rec0["c"]
        cur = ConceptNode(element_id=rec0["eid"], name=c0.get("name"),
                          type=c0.get("type"))
        steps: List[ReasoningStep] = []
        last_paper: Optional[PaperNode] = None
        visited = {cur.element_id}

        while True:
            recs = tx.run(
                """
                MATCH (c:Concept) WHERE elementId(c) = $eid
                                    AND c.type = 'Resulting_Concept'
                MATCH (c)-[:INSPIRES]->(p:Paper)
                MATCH (p)-[:PRODUCES]->(next:Concept {type:'Resulting_Concept'})
                WHERE elementId(next) <> $eid
                RETURN p, next, elementId(p) AS p_eid, elementId(next) AS next_eid
                """, eid=cur.element_id,
            ).data()
            found = next((r for r in recs if r["next_eid"] not in visited), None)
            if not found:
                break
            visited.add(found["next_eid"])
            p   = found["p"]; nxt = found["next"]
            paper = PaperNode(
                element_id=found["p_eid"], title=p.get("title"),
                conceptual_premises=p.get("conceptual_premises"),
                contextual_model_constraints=p.get("contextual_model_constraints"),
            )
            last_paper = paper
            nxt_node   = ConceptNode(element_id=found["next_eid"],
                                     name=nxt.get("name"), type=nxt.get("type"))
            steps.append(ReasoningStep(from_concept=cur, via_paper=paper,
                                       to_concept=nxt_node))
            cur = nxt_node

        if last_paper is None:
            rec2 = tx.run(
                """
                MATCH (p:Paper)-[:PRODUCES]->(rc:Concept)
                WHERE elementId(rc) = $eid
                RETURN p, elementId(p) AS p_eid LIMIT 1
                """, eid=cur.element_id,
            ).single()
            if rec2:
                p = rec2["p"]
                last_paper = PaperNode(
                    element_id=rec2["p_eid"], title=p.get("title"),
                    conceptual_premises=p.get("conceptual_premises"),
                    contextual_model_constraints=p.get("contextual_model_constraints"),
                )
        return {"steps": steps, "final_concept": cur, "final_paper": last_paper}

    # ----------------------------------------------------------
    # _walk_all_paths  — NEW BFS expanding every branch
    # ----------------------------------------------------------

    @staticmethod
    def _walk_all_paths(tx, start_eid: str,
                        max_depth: int = 8) -> List[Dict[str, Any]]:
        """BFS from a Resulting_Concept, following EVERY INSPIRES→PRODUCES
        branch.  Returns a list of complete root-to-leaf path dicts, each
        matching the schema expected by _integrate_path().

        Cycle prevention: each BFS state carries its own `visited` frozenset
        so different branches do not interfere with each other.
        """
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

        # queue item: (current_concept_dict, steps_list, visited_frozenset)
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
                # leaf node
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
    # _walk_all_paths_from_other  — NEW, for non-RC start nodes
    # ----------------------------------------------------------

    @staticmethod
    def _walk_all_paths_from_other(tx, eid: str,
                                   max_depth: int = 8) -> List[Dict[str, Any]]:
        """Find all RC nodes reachable from a non-RC concept via
        INSPIRES/INDIRECT_INSPIRES/APPLY_TO, then BFS each."""
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
    # Unchanged helpers
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

    @staticmethod
    def _walk_from_other_concept(tx, eid: str) -> List[Dict[str, Any]]:
        rows = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|APPLY_TO]->(p:Paper)
            MATCH (p)-[:PRODUCES]->(rc:Concept {type:'Resulting_Concept'})
            RETURN DISTINCT elementId(rc) AS rc_eid
            """, eid=eid,
        ).data()
        chains: List[Dict[str, Any]] = []
        for row in rows:
            sub = Neo4jReasoningRetriever._walk_resulting_chain(
                tx, row["rc_eid"])
            chains.append(sub)
        return chains


# ----------------------------------------------------------------
# Module-level helper used inside static methods
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
