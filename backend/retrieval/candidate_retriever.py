# -*- coding: utf-8 -*-
"""retrieval/candidate_retriever.py — 候選 Concept 語意搜尋。

使用 SBERT embedding 對 Neo4j 中的 Resulting_Concept 做向量搜尋，
回傳與使用者 query 最相關的 top-k 候選清單。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from retrieval.encoder import SbertOnnxEncoder


_BATCH_SIZE = 64


class CandidateRetriever:
    """語意搜尋候選 Concept。

    Parameters
    ----------
    driver  : Neo4j driver
    encoder : SbertOnnxEncoder 實例
    """

    def __init__(self, driver, encoder: SbertOnnxEncoder) -> None:
        self._driver  = driver
        self._encoder = encoder
        self._index:            List[Dict[str, Any]] = []
        self._embeddings_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_embeddings(self) -> None:
        """從 Neo4j 載入所有 Resulting_Concept 的 embedding index。"""
        self._load_concept_embeddings()

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """語意搜尋，回傳 top-k concept 摘要。"""
        if not self._embeddings_loaded:
            self._load_concept_embeddings()

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_concept_embeddings(self) -> None:
        with self._driver.session() as session:
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

        variants: List[Dict[str, Any]] = self._build_variants(rows)
        self._encode_variants(variants)

        self._index            = variants
        self._embeddings_loaded = True
        print(f"[CandidateRetriever] loaded {len(variants)} variants "
              f"from {len(rows)} nodes.")

    @staticmethod
    def _build_variants(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        for row in rows:
            eid         = row["eid"]
            node_name   = row["name"]
            node_type   = row["type"]
            node_func   = row["function"] or ""
            node_atypes = row["analysis_types"] or []
            m_names     = row["merged_names"]
            m_descs     = row["merged_descriptions"]
            m_funcs     = row["merged_functions"]

            if m_names:
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
        return variants

    def _encode_variants(self, variants: List[Dict[str, Any]]) -> None:
        retrieval_texts  = [
            f"{v['variant_name']} {v['variant_function']}".strip()
            for v in variants
        ]
        similarity_texts = [
            f"{v['variant_name']} {v['variant_description']}".strip()
            for v in variants
        ]

        ret_embs: List[np.ndarray] = []
        sim_embs: List[np.ndarray] = []
        for i in range(0, len(variants), _BATCH_SIZE):
            ret_embs.append(
                self._encoder.encode(retrieval_texts[i:i + _BATCH_SIZE]))
            sim_embs.append(
                self._encoder.encode(similarity_texts[i:i + _BATCH_SIZE]))

        if ret_embs:
            ret_matrix = np.vstack(ret_embs)
            sim_matrix = np.vstack(sim_embs)
            for idx, v in enumerate(variants):
                v["retrieval_embedding"]  = ret_matrix[idx]
                v["similarity_embedding"] = sim_matrix[idx]

    def _match_entities(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_emb = self._encoder.encode(query)[0]

        scored: List[Dict[str, Any]] = []
        for v in self._index:
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

        # 同一 element_id 只保留最高分
        seen: Dict[str, Dict[str, Any]] = {}
        for s in scored:
            eid = s["element_id"]
            if eid not in seen:
                seen[eid] = s

        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return deduped[:top_k]

    # ------------------------------------------------------------------
    # 提供給 DownstreamBFS 的 analysis_types 豐富化（保留舊功能）
    # ------------------------------------------------------------------

    def get_eid_to_analysis_types(self) -> Dict[str, List[str]]:
        """回傳 {element_id: [analysis_type, ...]} 供 BFS 節點豐富化使用。"""
        return {
            v["element_id"]: v.get("analysis_types", [])
            for v in self._index
        }
