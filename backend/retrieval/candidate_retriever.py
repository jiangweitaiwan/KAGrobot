# -*- coding: utf-8 -*-
"""retrieval/candidate_retriever.py — 候選 Concept 語意搜尋。

使用 SBERT embedding 對 Neo4j 中的 Resulting_Concept 做向量搜尋，
回傳與使用者 query 最相關的 top-k 候選清單。

Neo4j Concept 欄位：
  definition / key_assumptions / mechanism / name：一般 concept 與 RC 皆有。
  function：僅 Resulting Concept 才有值（一般 concept 為 null）。
            mechanism 與 function 是並存欄位，不是替代關係。
  舊版 description / 舊版 function（取代前） / analysis_types 欄位已移除。

Embedding 文字組裝：
  retrieval_embedding  : name + function     （RC 專屬的精簡功能描述，檢索用）
  similarity_embedding : name + mechanism    （概念機制描述，相似度比對用）

此檔案只對 type='Resulting_Concept' 的節點建立 embedding index，
故查到的 function 理論上恆有值；若遇到 RC 節點 function 為 null 的
邊界情況，仍以空字串安全退化，不影響檢索。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

        query_emb = self.embed(query)
        matches = self._score_index(query_emb, top_k=top_k)
        return [
            {
                "index":   i + 1,
                "eid":     m["element_id"],
                "rc_name": m["node_name"],
                "type":    m["node_type"],
                "analysis_types": m.get("analysis_types", []),
                "score":   round(m["score"], 4),
            }
            for i, m in enumerate(matches)
        ]

    def embed(self, text: str) -> np.ndarray:
        """對外部文字（例如 SemanticContext.embedding_text）編碼。"""
        return self._encoder.encode(text)[0]

    def score_all(self, query_emb: np.ndarray) -> List[Dict[str, Any]]:
        """對索引內所有 RC 計算 cosine，不做 top_k 截斷，回傳去重後全部結果。

        供 Semantic 機制（RC 全域排名 / Foundation 推薦的 RC pool）使用，
        與 retrieve_candidates 共用同一份索引與語料（name+function），
        避免與舊版 rc-semantic-scores 各自維護一套 embedding 造成語料不一致。
        """
        if not self._embeddings_loaded:
            self._load_concept_embeddings()
        return self._score_index(query_emb, top_k=None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_concept_embeddings(self) -> None:
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE c.type = 'Resulting_Concept'
                RETURN elementId(c)             AS eid,
                       c.name                   AS name,
                       c.type                   AS type,
                       c.definition             AS definition,
                       c.mechanism              AS mechanism,
                       c.key_assumptions        AS key_assumptions,
                       c.function               AS function,
                       c.analysis_types       AS analysis_types,
                       c.merged_names           AS merged_names

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
            eid           = row["eid"]
            node_name     = row["name"]
            node_type     = row["type"]
            node_func     = row["function"] or ""
            node_atypes = row["analysis_types"] or []
            node_mech     = row["mechanism"] or ""
            m_names       = row["merged_names"]


            if m_names:
                for i, mn in enumerate(m_names):
                    variants.append({
                    "element_id":        eid,
                    "node_name":         node_name,
                    "node_type":         node_type,
                    "analysis_types":      node_atypes,
                    "variant_name":      node_name or "",
                    "variant_function":  node_func,
                    "variant_mechanism": node_mech,
                })
            else:
                variants.append({
                    "element_id":        eid,
                    "node_name":         node_name,
                    "node_type":         node_type,
                    "analysis_types":      node_atypes,
                    "variant_name":      node_name or "",
                    "variant_function":  node_func,
                    "variant_mechanism": node_mech,
                })
        return variants

    def _encode_variants(self, variants: List[Dict[str, Any]]) -> None:
        retrieval_texts  = [
            f"{v['variant_name']} {v['variant_function']}".strip()
            for v in variants
        ]
        similarity_texts = [
            f"{v['variant_name']} {v['variant_mechanism']}".strip()
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




    def _score_index(
        self,
        query_emb: np.ndarray,
        top_k: Optional[int],
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for v in self._index:
            if "retrieval_embedding" not in v:
                continue
            score = float(np.dot(query_emb, v["retrieval_embedding"]))
            scored.append({
                "element_id": v["element_id"],
                "node_name":  v["node_name"],
                "node_type":  v["node_type"],
                "analysis_types": v.get("analysis_types", []),
                "score":      score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # 同一 element_id 只保留最高分
        seen: Dict[str, Dict[str, Any]] = {}
        for s in scored:
            eid = s["element_id"]
            if eid not in seen:
                seen[eid] = s

        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return deduped[:top_k] if top_k is not None else deduped        