# -*- coding: utf-8 -*-
"""graph_service/walk_start_pool.py — Random Walk Mode 的起點檢索池。

對應設計討論定案：起點候選只掃描「曾經是某個 RC 的 upstream foundation」
的 concept，不是全圖所有節點——語意上更貼近「已知線索」的直覺，範圍也
天然有界。

「曾是某 RC 的 foundation」在圖上等價於：這個 concept 沒有任何上游鄰居
（既沒有其他 Concept 透過 COMPOSES 指向它，也沒有其他 Concept 經由 Paper
INSPIRES/INDIRECT_INSPIRES/COMPOSES 橋接到它）——也就是 leaf 節點。這與
graph_service/bfs_cache.py 的 AdjacencyCache._query_upstream 判斷「已無
上游鄰居即終止並記為 foundation」的邏輯是同一件事，這裡直接用同樣的
「無上游」條件在 Neo4j 端一次篩出全部候選，而不是逐一對每個 RC 跑一次
upstream BFS 再收斂（那樣對全圖來說是 O(RC 數) 次 BFS，這裡是一次
Cypher）。

Embedding 語料：一般 concept 目前沒有 function 欄位（見使用者 note），
一律用 name + (mechanism or definition)，與 walk_frontier.py 的候選描述
語料保持一致，避免同一個節點在「frontier 呈現」與「起點檢索」用兩套
不同語料、排序邏輯彼此不一致。

與 CandidateRetriever 的關係：職責不同、故意不合併——CandidateRetriever
索引的是「RC 專屬 name+function」語料，只服務「使用者描述需求 → 直接找
RC」這條路徑（既有 dialogue mode）；這裡索引的是「leaf foundation
concept」，服務的是 random walk mode 的起點選取，兩者的候選集合、語料
組成都不同，合併只會讓兩條路徑互相牽制。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from graph_service.walk_state import WalkNodeRef

if TYPE_CHECKING:
    from retrieval.encoder import SbertOnnxEncoder

_BATCH_SIZE = 64


class StartConceptPool:
    """Leaf foundation concept 的語意搜尋索引，供 random walk 起點選取使用。

    Parameters
    ----------
    driver  : Neo4j driver
    encoder : SbertOnnxEncoder 實例
    """

    def __init__(self, driver: Any, encoder: "SbertOnnxEncoder") -> None:
        self._driver  = driver
        self._encoder = encoder
        self._index:  List[Dict[str, Any]] = []
        self._loaded  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """從 Neo4j 載入所有 leaf foundation concept 並建立 embedding index。"""
        rows = self._fetch_leaf_concepts()
        self._index = self._encode_rows(rows)
        self._loaded = True
        print(f"[StartConceptPool] loaded {len(self._index)} leaf foundation concepts.")

    def retrieve(self, query_text: str, k: int = 3) -> List[WalkNodeRef]:
        """語意搜尋起點候選，回傳 top-k（依 cosine 相似度排序）。

        query_text 通常是使用者第一輪輸入的原始描述（或 socratic 精煉過的
        query），與既有 dialogue mode 的 CandidateRetriever.retrieve_candidates
        用法一致，只是候選集合換成 leaf foundation concept。
        """
        if not self._loaded:
            self.load()

        if not self._index:
            return []

        query_emb = self._encoder.encode(query_text)[0]
        scored = [
            (float(np.dot(query_emb, item["embedding"])), item)
            for item in self._index
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            WalkNodeRef(eid=item["eid"], label=item["name"])
            for _score, item in scored[:k]
        ]

    @property
    def size(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_leaf_concepts(self) -> List[Dict[str, Any]]:
        """一次 Cypher 查詢，篩出所有「無上游鄰居」的 Concept 節點。

        條件與 AdjacencyCache._query_upstream 的兩條反向邊完全對應：
          1. 沒有其他 Concept 透過 COMPOSES 指向它
          2. 沒有其他 Concept 經由 Paper（INSPIRES/INDIRECT_INSPIRES/
             COMPOSES）橋接到它
        兩者皆無，代表它是 upstream BFS 走到底的 leaf——也就是某個（或
        某些）RC 的 foundation。
        """
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE NOT (:Concept)-[:COMPOSES]->(c)
                  AND NOT (:Concept)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->
                          (:Paper)-[:PRODUCES]->(c)
                RETURN elementId(c) AS eid,
                       c.name       AS name,
                       c.type       AS type,
                       c.definition AS definition,
                       c.mechanism  AS mechanism
                """
            ).data()
        return rows

    def _encode_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [
            f"{r['name']} {r.get('mechanism') or r.get('definition') or ''}".strip()
            for r in rows
        ]

        embs: List[np.ndarray] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            embs.append(self._encoder.encode(texts[i:i + _BATCH_SIZE]))
        emb_matrix = np.vstack(embs) if embs else np.zeros((0, 0))

        return [
            {
                "eid":        r["eid"],
                "name":       r["name"] or r["eid"],
                "type":       r.get("type"),
                "embedding":  emb_matrix[idx],
            }
            for idx, r in enumerate(rows)
        ]
