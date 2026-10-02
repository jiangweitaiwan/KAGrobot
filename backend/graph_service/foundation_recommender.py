# -*- coding: utf-8 -*-
"""graph_service/foundation_recommender.py — 語意匹配 Foundation 推薦（方案 B）。

範圍限定在「目前 ranking pool 內候選 RC 已展開（或 lazy 展開後）的
foundation 集合」，不建全域 Concept embedding index：

  1. 從 CandidateRetriever.score_all() 拿到全域 RC 排名，取前 pool_size 個
  2. 透過 BFSCoordinator.get_expand_cached_async()（既有 lazy + cache 機制，
     命中不重打 Neo4j）取得這些 RC 的 foundation 清單，去重
  3. 對 foundation label 現場批次 embedding（結果快取在本物件生命週期內，
     圖譜結構不變即永久有效，不需要每次 request 重算）
  4. 與 query embedding 算 cosine，回傳 top-N

與 CandidateRetriever 的全域 RC index 不同，這裡刻意不在啟動時
預先建索引，因為 foundation 的候選集合本身就是動態、有界的
（受 ranking pool 大小限制），現場算完全負擔得起，也符合本專案
一貫的 lazy + cache 設計哲學（見 AdjacencyCache / BFSCoordinator）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import numpy as np

from graph_service.bfs_coordinator import BFSCoordinator
from graph_service.foundation_tree_labels import ExpandLabelEnricher
from retrieval.encoder import SbertOnnxEncoder

_TOP_N_FOUNDATIONS = 8


class FoundationRecommender:
    def __init__(
        self,
        coordinator: BFSCoordinator,
        encoder: SbertOnnxEncoder,
        *,
        label_enricher: ExpandLabelEnricher | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._encoder     = encoder
        self._label_enricher = label_enricher
        # element_id → embedding（圖譜結構不變即永久有效，process 生命週期）
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def recommend(
        self,
        query_emb:         np.ndarray,
        candidate_rc_eids: List[str],
    ) -> List[Dict[str, Any]]:
        """回傳按 cosine 相似度排序的 foundation 推薦清單（去重，取 top-N）。"""
        foundation_labels = await self._collect_foundation_labels(candidate_rc_eids)
        if not foundation_labels:
            return []

        await self._ensure_embeddings(foundation_labels)

        scored = [
            {
                "element_id": eid,
                "label":      label,
                "score":      float(np.dot(query_emb, self._embedding_cache[eid])),
            }
            for eid, label in foundation_labels.items()
            if eid in self._embedding_cache
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:_TOP_N_FOUNDATIONS]

    # ------------------------------------------------------------------

    async def _fetch_expand(self, eid: str) -> Dict[str, Any]:
        enricher = self._label_enricher
        start_name = enricher.anchor_label(eid) if enricher else ""
        result = await self._coordinator.get_expand_cached_async(eid, start_name=start_name)
        if enricher:
            enricher.enrich_result(eid, result)
        return result

    async def _collect_foundation_labels(self, rc_eids: List[str]) -> Dict[str, str]:
        results = await asyncio.gather(*[self._fetch_expand(eid) for eid in rc_eids])
        labels: Dict[str, str] = {}
        for r in results:
            for f in r.get("final_foundations", []):
                labels.setdefault(f["element_id"], f["label"])
        return labels

    async def _ensure_embeddings(self, labels: Dict[str, str]) -> None:
        missing = {eid: lb for eid, lb in labels.items() if eid not in self._embedding_cache}
        if not missing:
            return
        eids  = list(missing.keys())
        texts = [missing[e] for e in eids]
        embs  = await asyncio.to_thread(self._encoder.encode, texts)
        for eid, emb in zip(eids, embs):
            self._embedding_cache[eid] = emb