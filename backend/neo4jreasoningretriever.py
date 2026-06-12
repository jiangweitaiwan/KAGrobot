# -*- coding: utf-8 -*-
"""Neo4jReasoningRetriever

Public API
----------
retrieve_candidates(query, top_k)
    SBERT 語意搜尋，回傳 top-k matched concept 摘要。
    僅在 socratic 階段使用；不走 Neo4j chain。
    使用 name + function embedding 做匹配。

retrieve_graph(start_eid, rc_name, max_depth)
    以已知 element_id 展開下游 BFS，回傳 {nodes, edges, paths}。
    支援：
      A 類圖：(s)-[:REASONING]->(t)  直接 concept→concept
      B 類圖：s -[:INSPIRES|INDIRECT_INSPIRES|REASONING]-> Paper -[:PRODUCES]-> t
    上游資訊改由 retrieve_foundations 負責。

retrieve_foundations(start_eid)
    從已知 element_id 走上游 BFS（max_depth=3），回傳：
      - final_foundations : 每條路徑最末端 leaf（給 KnowledgeGraph diamond 節點）
      - foundation_tree   : 所有上游中間節點+邊（給 GlobalBrain 樹狀線段）

retrieve_concept_detail(rc_name)
    查詢指定 concept 的 description 以及所屬 paper 的
    conceptual_premises / contextual_model_constraints。
    供 SocratesChatBot concept_tool 使用，直接查 Neo4j。

project_concepts_only(graph_data)
    靜態方法。將完整 graph_data 投影為只含 concept 節點與
    concept-to-concept 邊的精簡結構，供前端與 graph_context 使用。
    同時保留 foundations 與 foundation_tree 欄位。
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import os
import tempfile
from google.cloud import storage as gcs_storage
from transformers import AutoTokenizer
import onnxruntime as ort
from neo4j import GraphDatabase
import pickle
from pathlib import Path
from sklearn.decomposition import PCA


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

        retrieval_texts = [
            f"{v['variant_name']} {v['variant_function']}".strip()
            for v in variants
        ]
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
    # Entity matching
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
        """查詢指定 concept 的描述與所屬論文的理論背景。直接查 Neo4j。"""
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
    # Public: retrieve_foundations()  ← UPDATED
    # ----------------------------------------------------------

    def retrieve_foundations(
        self,
        start_eid: str,
        max_depth: int = 10,
        max_branches: int = 4,
    ) -> Dict[str, Any]:
        """走上游 BFS（max_depth=3），回傳兩份資料。
        
        以 _select_diverse_deep_paths() 選出兼顧深度與多樣性的
        top max_branches 條路徑，再從中組裝結果。

        - final_foundations : 每條路徑最末端 leaf（給 KnowledgeGraph diamond 節點）
        - foundation_tree   : 所有上游中間節點 + 邊（給 GlobalBrain 樹狀線段）

        Returns
        -------
        {
            "final_foundations": [{"element_id": str, "label": str}, ...],
            "foundation_tree": {
                "nodes": [{"element_id": str, "label": str}, ...],
                "edges": [{"source": str, "target": str}, ...]
            }
        }
        """
        with self.driver.session() as session:
            raw_paths = session.execute_read(
                self._walk_upstream, start_eid, max_depth
            )

       # ── 選出兼顧深度與多樣性的路徑 ──────────────────────
        selected_paths = _select_diverse_deep_paths(raw_paths, max_branches)

        seen_leaf_eids: set = set()
        final_foundations: List[Dict[str, Any]] = []
        tree_nodes: Dict[str, Dict[str, Any]] = {}
        tree_edges: Dict[str, Dict[str, Any]] = {}

        for path in selected_paths:
            steps = path.get("steps") or []
            if not steps:
                continue

            # leaf = steps[0]["from_concept"]
            # (_walk_upstream 以 [new_step] + steps 的方式 prepend，
            #  所以 steps[0]["from_concept"] 是最末端上游節點)
            leaf = steps[0]["from_concept"]
            leaf_eid = leaf["element_id"]
            if leaf_eid not in seen_leaf_eids:
                seen_leaf_eids.add(leaf_eid)
                final_foundations.append({
                    "element_id": leaf_eid,
                    "label": leaf.get("name") or leaf_eid,
                })

            # 所有途經節點（每個 step 的 from + to）
            for step in steps:
                fc = step["from_concept"]
                fc_eid = fc["element_id"]
                if fc_eid not in tree_nodes:
                    tree_nodes[fc_eid] = {
                        "element_id": fc_eid,
                        "label": fc.get("name") or fc_eid,
                    }

                tc = step["to_concept"]
                tc_eid = tc["element_id"]
                if tc_eid not in tree_nodes:
                    tree_nodes[tc_eid] = {
                        "element_id": tc_eid,
                        "label": tc.get("name") or tc_eid,
                    }

                # 邊方向：from → to（上游流向 candidate，最終匯聚到 start_eid）
                edge_key = f"{fc_eid}__{tc_eid}"
                if edge_key not in tree_edges:
                    tree_edges[edge_key] = {
                        "source": fc_eid,
                        "target": tc_eid,
                    }

        return {
            "final_foundations": final_foundations,
            "foundation_tree": {
                "nodes": list(tree_nodes.values()),
                "edges": list(tree_edges.values()),
            },
        }

    # ----------------------------------------------------------
    # Public: get_global_layout()
    # ----------------------------------------------------------

    def get_global_layout(
            self,
            pkl_path: str,
            *,
            chapter_pkl_path: Optional[str] = None,
            gcs_bucket_name: Optional[str] = None,
            gcs_object_name: str = "results_dict.pkl",
            chapter_gcs_object_name: str = "chapter_results_dict.pkl",
            force_recompute: bool = False,
        ) -> dict:
            """
            讀取 results pkl + chapter pkl，做 PCA 降到 3D，與 Neo4j element_id 對齊。
            每個 node 帶有 source 欄位：'results' 或 'chapter'。
            同一 EID 同時出現在兩個來源時，優先歸為 'results'。

            Returns
            -------
            {
            "nodes": [{"id", "name", "type", "source", "x", "y", "z"}, ...],
            "edges": [{"source", "target"}, ...]
            }
            """
            if not force_recompute and hasattr(self, "_global_layout_cache"):
                return self._global_layout_cache

            bucket = (gcs_bucket_name or os.environ.get("GCS_BUCKETFORRESULT", "")).strip()

            # ── GCS download helper ────────────────────────────────
            def _download_pkl(gcs_obj: str, fallback_local: str) -> str:
                if not bucket:
                    return fallback_local
                try:
                    client = gcs_storage.Client()
                    b = client.bucket(bucket)
                    blob = b.blob(gcs_obj)
                    if not blob.exists(client):
                        raise FileNotFoundError(
                            f"GCS object not found: gs://{bucket}/{gcs_obj}"
                        )
                    with tempfile.NamedTemporaryFile(
                        prefix=f"pkl_{gcs_obj.replace('/', '_')}_",
                        suffix=".pkl",
                        delete=False,
                    ) as tmp:
                        tmp_path = tmp.name
                    blob.download_to_filename(tmp_path)
                    return tmp_path
                except Exception as exc:
                    print(
                        f"[GlobalLayout] GCS load failed for {gcs_obj}, "
                        f"fallback to '{fallback_local}': {exc}"
                    )
                    return fallback_local

            local_results_path = _download_pkl(gcs_object_name, pkl_path)
            local_chapter_path = _download_pkl(
                chapter_gcs_object_name,
                chapter_pkl_path or "",
            )

            # ── Step 1a：解析 results pkl ──────────────────────────
            import pickle
            from collections import defaultdict

            with open(local_results_path, "rb") as f:
                results_raw = pickle.load(f)

            results_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
            for _paper, concepts in results_raw.items():
                for c in concepts:
                    emb_dict = c["alignment"]["embeddings"]
                    all_embs: list[np.ndarray] = []
                    for emb_list in emb_dict.values():
                        for emb in emb_list:
                            arr = np.array(emb)
                            if arr.shape == (384,):
                                all_embs.append(arr)
                    if all_embs:
                        results_by_name[c["cluster_name"]].append(
                            np.mean(all_embs, axis=0)
                        )

            results_concepts: list[dict] = []
            for name, embs in results_by_name.items():
                results_concepts.append({
                    "name":      name,
                    "source":    "results",
                    "embedding": np.mean(embs, axis=0),
                })

            # ── Step 1b：解析 chapter pkl ──────────────────────────
            chapter_concepts: list[dict] = []
            if local_chapter_path and os.path.exists(local_chapter_path):
                try:
                    with open(local_chapter_path, "rb") as f:
                        chapter_raw = pickle.load(f)
                    chapter_by_name: dict[str, np.ndarray] = {}
                    for _chap, items in chapter_raw.items():
                        for item in items:
                            cname = item["concept"]
                            emb   = np.array(item["embeddings"])
                            if emb.shape == (384,):
                                if cname not in chapter_by_name:
                                    chapter_by_name[cname] = emb
                                else:
                                    chapter_by_name[cname] = (
                                        chapter_by_name[cname] + emb
                                    ) / 2
                    for name, emb in chapter_by_name.items():
                        chapter_concepts.append({
                            "name":      name,
                            "source":    "chapter",
                            "embedding": emb,
                        })
                except Exception as exc:
                    print(f"[GlobalLayout] failed to load chapter pkl: {exc}")
            else:
                print("[GlobalLayout] chapter pkl not found, skipping.")

            # ── Step 1c：合併，results 優先 ────────────────────────
            results_names = {c["name"] for c in results_concepts}
            all_concepts  = results_concepts + [
                c for c in chapter_concepts if c["name"] not in results_names
            ]

            print(
                f"[GlobalLayout] concepts: {len(results_concepts)} results, "
                f"{len(chapter_concepts)} chapter raw, "
                f"{len(all_concepts)} merged (unique)"
            )

            # ── Step 2：PCA 384 → 3 ───────────────────────────────
            embs_matrix = np.array([c["embedding"] for c in all_concepts])
            try:
                from sklearn.decomposition import PCA as _PCA
                pca       = _PCA(n_components=3, random_state=42)
                coords_3d = pca.fit_transform(embs_matrix)
            except Exception:
                centered  = embs_matrix - embs_matrix.mean(axis=0)
                _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
                coords_3d = centered @ Vt[:3].T

            for i in range(3):
                max_abs = np.abs(coords_3d[:, i]).max()
                if max_abs > 1e-9:
                    coords_3d[:, i] /= max_abs

            name_to_coord: dict[str, tuple] = {
                c["name"]: (
                    float(coords_3d[idx, 0]),
                    float(coords_3d[idx, 1]),
                    float(coords_3d[idx, 2]),
                )
                for idx, c in enumerate(all_concepts)
            }
            name_to_source: dict[str, str] = {
                c["name"]: c["source"] for c in all_concepts
            }

            # ── Step 3：查 Neo4j ───────────────────────────────────
            with self.driver.session() as session:
                concept_rows = session.run(
                    """
                    MATCH (c:Concept)
                    RETURN elementId(c) AS eid,
                        c.name       AS name,
                        c.type       AS type
                    """
                ).data()

                direct_edges = session.run(
                    """
                    MATCH (a:Concept)-[:REASONING]->(b:Concept)
                    RETURN elementId(a) AS src, elementId(b) AS tgt
                    """
                ).data()

                paper_edges = session.run(
                    """
                    MATCH (a:Concept)-[:INSPIRES|INDIRECT_INSPIRES|REASONING]->
                        (:Paper)-[:PRODUCES]->(b:Concept)
                    RETURN elementId(a) AS src, elementId(b) AS tgt
                    """
                ).data()

            # ── Step 4：對齊 name → element_id ───────────────────
            nodes_out: list[dict] = []
            eid_set:   set[str]   = set()
            matched_eids: set[str] = set()

            for row in concept_rows:
                eid   = row["eid"]
                name  = row["name"] or ""
                ctype = row["type"] or "other"
                eid_set.add(eid)

                coord  = name_to_coord.get(name)
                source = name_to_source.get(name, "results")

                if coord:
                    matched_eids.add(eid)
                    nodes_out.append({
                        "id":     eid,
                        "name":   name,
                        "type":   ctype,
                        "source": source,
                        "x":      coord[0],
                        "y":      coord[1],
                        "z":      coord[2],
                    })
                else:
                    rng = np.random.default_rng(abs(hash(name)) % (2**32))
                    nodes_out.append({
                        "id":     eid,
                        "name":   name,
                        "type":   ctype,
                        "source": "results",
                        "x":      float(rng.uniform(-0.15, 0.15)),
                        "y":      float(rng.uniform(-0.15, 0.15)),
                        "z":      float(rng.uniform(-0.15, 0.15)),
                    })

            print(
                f"[GlobalLayout] Neo4j concepts: {len(concept_rows)}, "
                f"pkl-matched: {len(matched_eids)}, "
                f"unmatched: {len(concept_rows) - len(matched_eids)}"
            )

            # ── Step 5：整理邊（去重）──────────────────────────────
            seen_edges: set[tuple] = set()
            edges_out:  list[dict] = []

            for e in direct_edges + paper_edges:
                src, tgt = e["src"], e["tgt"]
                if src not in eid_set or tgt not in eid_set:
                    continue
                key = (min(src, tgt), max(src, tgt))
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges_out.append({"source": src, "target": tgt})

            print(
                f"[GlobalLayout] edges: {len(edges_out)} "
                f"(direct={len(direct_edges)}, via-paper={len(paper_edges)})"
            )

            result = {"nodes": nodes_out, "edges": edges_out}
            self._global_layout_cache = result
            return result

    # ----------------------------------------------------------
    # Public: retrieve_graph()  — 下游 BFS only
    # ----------------------------------------------------------

    def retrieve_graph(
        self,
        start_eid:      str,
        rc_name:        str = "",
        max_depth_down: int = 8,
    ) -> Dict[str, Any]:
        """以 element_id 展開下游 BFS，回傳 D3 渲染用結構。

        上游資訊改由 retrieve_foundations() 負責，此方法不再走上游。
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

        seen_sigs: set = set()
        unique_paths: List[dict] = []
        for p in all_paths:
            sig = tuple(p["node_ids"])
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)

        self._enrich_nodes_with_analysis_types(nodes_map)

        return {
            "query_concepts": [{
                "id":    start_eid,
                "name":  rc_name,
                "type":  "Resulting_Concept",
                "score": 1.0,
            }],
            "nodes":       list(nodes_map.values()),
            "edges":       list(edges_map.values()),
            "paths":       unique_paths,
            "foundations": [],       # filled by _expand_all_graphs → retrieve_foundations
            "foundation_tree": {"nodes": [], "edges": []},  # idem
        }

    def _enrich_nodes_with_analysis_types(
        self, nodes_map: Dict[str, dict]
    ) -> None:
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
        """將完整 graph_data 投影為只含 concept 節點與 concept-to-concept 邊。
        foundations 與 foundation_tree 欄位直接透傳。
        """
        nodes_map  = {n["id"]: n for n in graph_data.get("nodes", [])}
        concept_ids = {
            nid for nid, n in nodes_map.items()
            if n.get("node_type") == "concept"
        }

        proj_nodes = [
            n for n in graph_data.get("nodes", [])
            if n.get("node_type") == "concept"
        ]

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
            "foundations":      graph_data.get("foundations", []),
            "foundation_tree":  graph_data.get("foundation_tree", {"nodes": [], "edges": []}),
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
            "intent_score":  0.0,
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
    # BFS: _walk_upstream（供 retrieve_foundations 使用）
    # ----------------------------------------------------------

    @staticmethod
    def _walk_upstream(tx, start_eid: str,
                       max_depth: int = 3) -> List[Dict[str, Any]]:
        """走上游 BFS，回傳所有完整路徑（含 steps）。
        retrieve_foundations 從中提取各路徑的 leaf concept 及所有中間節點。
        """
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


# ----------------------------------------------------------------
# Foundation path selection — 可獨立替換的選路模組
# ----------------------------------------------------------------

def _path_node_set(path: Dict[str, Any]) -> set:
    """提取一條 foundation path 中所有 concept element_id 的集合。"""
    nodes: set = set()
    for step in path.get("steps", []):
        fc = step.get("from_concept")
        tc = step.get("to_concept")
        if fc:
            nodes.add(fc["element_id"])
        if tc:
            nodes.add(tc["element_id"])
    return nodes


def _jaccard_distance(set_a: set, set_b: set) -> float:
    """兩個集合的 Jaccard distance（越大代表越不重疊）。"""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - len(set_a & set_b) / union


def _select_diverse_deep_paths(
    paths: List[Dict[str, Any]],
    max_branches: int = 3,
) -> List[Dict[str, Any]]:
    """從 foundation raw_paths 中選出兼顧深度與多樣性的 top-N 條路徑。

    演算法：
      1. 以路徑步數的中位數為深度門檻，過濾太短的路徑。
      2. 貪婪選取：第一條取最深的；後續每輪選與已選路徑
         node set 的最小 Jaccard distance 最大的那條
         （即「與現有選擇最不重疊」）。

    若過濾後候選為空，fallback 到全部路徑再跑一次。
    若 paths 為空，回傳空列表。

    替換此函數即可切換選路策略，不需動 retrieve_foundations()。
    """
    if not paths:
        return []

    if max_branches <= 0:
        return []

    # ── 1. 裝飾：計算每條路徑的深度與 node set ──────────────
    decorated = [
        {
            "path":     p,
            "depth":    len(p.get("steps", [])),
            "node_set": _path_node_set(p),
        }
        for p in paths
    ]

    # ── 2. 深度門檻（中位數）────────────────────────────────
    depths   = sorted(d["depth"] for d in decorated)
    median   = depths[len(depths) // 2]
    candidates = [d for d in decorated if d["depth"] >= median]

    # fallback：若全被過濾（理論上不會，但防呆）
    if not candidates:
        candidates = list(decorated)

    # ── 3. 貪婪選取 ──────────────────────────────────────────
    candidates.sort(key=lambda d: d["depth"], reverse=True)

    selected:            List[Dict[str, Any]] = [candidates[0]]
    selected_node_sets:  List[set]            = [candidates[0]["node_set"]]
    remaining = candidates[1:]

    while len(selected) < max_branches and remaining:
        # 對每條候選算它與所有已選路徑的最小 Jaccard distance
        def _min_dist(d: Dict[str, Any]) -> float:
            return min(
                _jaccard_distance(d["node_set"], s)
                for s in selected_node_sets
            )

        remaining.sort(key=_min_dist, reverse=True)
        best = remaining.pop(0)
        selected.append(best)
        selected_node_sets.append(best["node_set"])

    return [d["path"] for d in selected]


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
