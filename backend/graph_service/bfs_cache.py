# -*- coding: utf-8 -*-
"""graph_service/bfs_cache.py — 鄰接關係快取層（AdjacencyCache）

職責：
  - 以 (node_id, direction) 為 key，快取 Neo4j 鄰接查詢結果
  - LRU maxsize=500，TTL=7天（batch 更新後重啟即自動清除）
  - 提供 downstream / upstream 兩個方向的鄰接查詢
  - cache miss 時才打 Neo4j，單次查詢單一節點的所有鄰居

設計原則：
  - 本層只負責「節點鄰接關係」，不負責完整路徑組裝
  - 所有 BFS 過程的 Neo4j query 都必須經過此層
  - 重啟即清空（不需要 invalidate endpoint）

【重構】LRU + TTL 快取的實作（原本私有的 _LRUCache）已抽成共用模組
core/lru_cache.py::LRUCache，讓 graph_service/bfs_coordinator.py 的
結果層 cache（rcpath_cache / expand_cache）也能共用同一套機制與統一
TTL（DEFAULT_TTL_S，7 天），不再各自手刻一套快取邏輯。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.lru_cache import DEFAULT_TTL_S, LRUCache


# ══════════════════════════════════════════════════════════════
# Data classes — 鄰接關係的標準資料結構
# ══════════════════════════════════════════════════════════════

@dataclass
class ConceptInfo:
    """Concept 節點的基本資訊。"""
    element_id: str
    name:       str
    node_type:  Optional[str]


@dataclass
class PaperInfo:
    """Paper 節點的基本資訊。"""
    element_id:                    str
    title:                         str
    conceptual_premises:           Optional[str] = None
    contextual_model_constraints:  Optional[str] = None


@dataclass
class DownstreamNeighbor:
    """Downstream BFS 的一條分支結果。

    分兩種：
      - 直接邊：Concept -[COMPOSES]-> Concept（paper=None）
      - 跨 Paper：Concept -[INSPIRES/...]-> Paper -[PRODUCES]-> Concept
    """
    to_concept: ConceptInfo
    via_paper:  Optional[PaperInfo] = None


@dataclass
class UpstreamNeighbor:
    """Upstream BFS 的一條分支結果。

    分四種（見 CachedUpstreamBFS 的 sign propagation 設計）：
      - 直接邊：Concept <-[COMPOSES]- Concept（edge_type="support"，paper=None）
      - 跨 Paper：Concept <-[PRODUCES]- Paper <-[INSPIRES/INDIRECT_INSPIRES/COMPOSES]- Concept
        （edge_type="support"）
      - 直接反駁：Concept <-[CONTRADICTS]- Concept（edge_type="critique"，paper=None）
      - 跨 Paper 反駁：Concept <-[PRODUCES]- Paper <-[CONTRADICTS]- Concept
        （edge_type="critique"，對應 importer.py 的 critiqued 輸入概念以
        CONTRADICTS 取代 INSPIRES 連到 Paper 的設計）
    """
    from_concept: ConceptInfo
    via_paper:    Optional[PaperInfo] = None
    edge_type:    str = "support"   # "support" | "critique"


# ══════════════════════════════════════════════════════════════
# AdjacencyCache — 核心快取層
# ══════════════════════════════════════════════════════════════

class AdjacencyCache:
    """Concept 節點鄰接關係的快取層。

    所有 BFS 的 Neo4j 鄰接查詢都必須經過此層。
    cache miss 才打 Neo4j；hit 直接回傳。

    Parameters
    ----------
    driver  : Neo4j driver
    maxsize : LRU 最大條目數（預設 500）
    ttl_s   : TTL 秒數（預設 DEFAULT_TTL_S，7 天，與結果層 cache 統一）
    """

    def __init__(
        self,
        driver,
        maxsize: int   = 500,
        ttl_s:   float = DEFAULT_TTL_S,
    ) -> None:
        self._driver = driver
        self._cache  = LRUCache(maxsize=maxsize, ttl_s=ttl_s)

    # ── Public API ────────────────────────────────────────────

    def get_downstream_neighbors(
        self,
        eid: str,
    ) -> List[DownstreamNeighbor]:
        """取得 eid 節點的所有下游鄰居。

        包含：
          1. 直接 COMPOSES 邊到其他 Concept
          2. 經由 Paper（INSPIRES/INDIRECT_INSPIRES/REASONING）到 RC
        """
        cache_key = f"down:{eid}"
        hit, value = self._cache.get(cache_key)
        if hit:
            return value

        neighbors = self._fetch_downstream(eid)
        self._cache.set(cache_key, neighbors)
        return neighbors

    def get_upstream_neighbors(
        self,
        eid: str,
    ) -> List[UpstreamNeighbor]:
        """取得 eid 節點的所有上游鄰居。

        包含：
          1. 反向 COMPOSES 邊來自其他 Concept
          2. 經由 Paper 反向追溯的前驅 Concept
        """
        cache_key = f"up:{eid}"
        hit, value = self._cache.get(cache_key)
        if hit:
            return value

        neighbors = self._fetch_upstream(eid)
        self._cache.set(cache_key, neighbors)
        return neighbors

    def get_concept_kind(self, eid: str) -> str:
        """判斷節點類型：resulting_concept / sub_concept / other。"""
        cache_key = f"kind:{eid}"
        hit, value = self._cache.get(cache_key)
        if hit:
            return value

        with self._driver.session() as session:
            kind = session.execute_read(self._query_concept_kind, eid)
        self._cache.set(cache_key, kind)
        return kind

    def get_parent_rc(self, eid: str) -> Optional[ConceptInfo]:
        """取得 sub_concept 的父 RC（若無則回傳 None）。"""
        cache_key = f"parent_rc:{eid}"
        hit, value = self._cache.get(cache_key)
        if hit:
            return value

        with self._driver.session() as session:
            parent = session.execute_read(self._query_parent_rc, eid)
        self._cache.set(cache_key, parent)
        return parent

    def get_fallback_paper(self, eid: str) -> Optional[PaperInfo]:
        """取得 RC 的 fallback paper（BFS 達到最大深度時使用）。"""
        cache_key = f"fallback_paper:{eid}"
        hit, value = self._cache.get(cache_key)
        if hit:
            return value

        with self._driver.session() as session:
            paper = session.execute_read(self._query_fallback_paper, eid)
        self._cache.set(cache_key, paper)
        return paper

    def clear(self) -> None:
        """清空全部快取（重啟時自動清空，通常不需手動呼叫）。"""
        self._cache.clear()
        print("[AdjacencyCache] cache cleared.")

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats

    # ── Neo4j 查詢（只在 cache miss 時呼叫）──────────────────

    def _fetch_downstream(self, eid: str) -> List[DownstreamNeighbor]:
        with self._driver.session() as session:
            return session.execute_read(self._query_downstream, eid)

    def _fetch_upstream(self, eid: str) -> List[UpstreamNeighbor]:
        with self._driver.session() as session:
            return session.execute_read(self._query_upstream, eid)

    # ── Neo4j transaction methods ─────────────────────────────

    @staticmethod
    def _query_downstream(tx, eid: str) -> List[DownstreamNeighbor]:
        """下游鄰接查詢：合併兩條 Cypher 為一次 session。"""
        neighbors: List[DownstreamNeighbor] = []

        # 1. 直接 COMPOSES 邊
        rows_a = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (c)-[:COMPOSES]->(next:Concept)
            WHERE elementId(next) <> $eid
            RETURN elementId(next) AS next_eid,
                   next.name       AS next_name,
                   next.type       AS next_type
            """,
            eid=eid,
        ).data()

        for row in rows_a:
            neighbors.append(DownstreamNeighbor(
                to_concept=ConceptInfo(
                    element_id=row["next_eid"],
                    name=row["next_name"] or "",
                    node_type=row["next_type"],
                ),
                via_paper=None,
            ))

        # 2. 經由 Paper 到 RC
        rows_b = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
                                AND c.type = 'Resulting_Concept'
            MATCH (c)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->(p:Paper)
            MATCH (p)-[:PRODUCES]->(next:Concept {type:'Resulting_Concept'})
            WHERE elementId(next) <> $eid
            RETURN elementId(p)    AS p_eid,
                   p.title         AS p_title,
                   p.conceptual_premises          AS p_premises,
                   p.contextual_model_constraints AS p_constraints,
                   elementId(next) AS next_eid,
                   next.name       AS next_name,
                   next.type       AS next_type
            """,
            eid=eid,
        ).data()

        for row in rows_b:
            neighbors.append(DownstreamNeighbor(
                to_concept=ConceptInfo(
                    element_id=row["next_eid"],
                    name=row["next_name"] or "",
                    node_type=row["next_type"],
                ),
                via_paper=PaperInfo(
                    element_id=row["p_eid"],
                    title=row["p_title"] or "",
                    conceptual_premises=row["p_premises"],
                    contextual_model_constraints=row["p_constraints"],
                ),
            ))

        return neighbors

    @staticmethod
    def _query_upstream(tx, eid: str) -> List[UpstreamNeighbor]:
        """上游鄰接查詢。

        support 兩段（COMPOSES 直接邊、Paper 中介）維持原樣；
        新增 critique 兩段，對應方向與 support 完全對稱，只是關係
        換成 CONTRADICTS——「source -[:CONTRADICTS]-> target」代表
        source 不認同 target，故 target 的 upstream 多一個「批評來源」。
        """
        neighbors: List[UpstreamNeighbor] = []

        # 1. 反向 COMPOSES 邊（support）
        rows_a = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (prev:Concept)-[:COMPOSES]->(c)
            WHERE elementId(prev) <> $eid
            RETURN elementId(prev) AS prev_eid,
                   prev.name       AS prev_name,
                   prev.type       AS prev_type
            """,
            eid=eid,
        ).data()

        for row in rows_a:
            neighbors.append(UpstreamNeighbor(
                from_concept=ConceptInfo(
                    element_id=row["prev_eid"], name=row["prev_name"] or "",
                    node_type=row["prev_type"],
                ),
                via_paper=None,
                edge_type="support",
            ))

        # 2. 反向 Paper 追溯（support）
        rows_b = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (p:Paper)-[:PRODUCES]->(c)
            MATCH (prev:Concept)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->(p)
            WHERE elementId(prev) <> $eid
            RETURN elementId(prev) AS prev_eid,
                   prev.name       AS prev_name,
                   prev.type       AS prev_type,
                   elementId(p)    AS p_eid,
                   p.title         AS p_title,
                   p.conceptual_premises          AS p_premises,
                   p.contextual_model_constraints AS p_constraints
            """,
            eid=eid,
        ).data()

        for row in rows_b:
            neighbors.append(UpstreamNeighbor(
                from_concept=ConceptInfo(
                    element_id=row["prev_eid"], name=row["prev_name"] or "",
                    node_type=row["prev_type"],
                ),
                via_paper=PaperInfo(
                    element_id=row["p_eid"], title=row["p_title"] or "",
                    conceptual_premises=row["p_premises"],
                    contextual_model_constraints=row["p_constraints"],
                ),
                edge_type="support",
            ))

        # 3. 反向 CONTRADICTS 邊（critique，直接）—— 新增
        #    critic -[:CONTRADICTS]-> cur：critic 不認同 cur，
        #    是 cur 的批評來源，故列入 cur 的 upstream。
        rows_c = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (critic:Concept)-[:CONTRADICTS]->(c)
            WHERE elementId(critic) <> $eid
            RETURN elementId(critic) AS prev_eid,
                   critic.name       AS prev_name,
                   critic.type       AS prev_type
            """,
            eid=eid,
        ).data()

        for row in rows_c:
            neighbors.append(UpstreamNeighbor(
                from_concept=ConceptInfo(
                    element_id=row["prev_eid"], name=row["prev_name"] or "",
                    node_type=row["prev_type"],
                ),
                via_paper=None,
                edge_type="critique",
            ))

        # 4. 反向 Paper 追溯（critique）—— 新增
        #    對應 importer.py::_create_input_concepts 對 critiqued 輸入概念
        #    改用 CONTRADICTS（而非 INSPIRES）連到 Paper：
        #        prev -[:CONTRADICTS]-> p -[:PRODUCES]-> cur
        #    代表 cur 是這篇論文「反駁 prev」而產出的結論。
        rows_d = tx.run(
            """
            MATCH (c:Concept) WHERE elementId(c) = $eid
            MATCH (p:Paper)-[:PRODUCES]->(c)
            MATCH (prev:Concept)-[:CONTRADICTS]->(p)
            WHERE elementId(prev) <> $eid
            RETURN elementId(prev) AS prev_eid,
                   prev.name       AS prev_name,
                   prev.type       AS prev_type,
                   elementId(p)    AS p_eid,
                   p.title         AS p_title,
                   p.conceptual_premises          AS p_premises,
                   p.contextual_model_constraints AS p_constraints
            """,
            eid=eid,
        ).data()

        for row in rows_d:
            neighbors.append(UpstreamNeighbor(
                from_concept=ConceptInfo(
                    element_id=row["prev_eid"], name=row["prev_name"] or "",
                    node_type=row["prev_type"],
                ),
                via_paper=PaperInfo(
                    element_id=row["p_eid"], title=row["p_title"] or "",
                    conceptual_premises=row["p_premises"],
                    contextual_model_constraints=row["p_constraints"],
                ),
                edge_type="critique",
            ))

        return neighbors

    @staticmethod
    def _query_concept_kind(tx, eid: str) -> str:
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
    def _query_parent_rc(tx, eid: str) -> Optional[ConceptInfo]:
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
        return ConceptInfo(
            element_id=rec["rc_eid"],
            name=rec["rc"].get("name") or "",
            node_type=rec["rc"].get("type"),
        )

    @staticmethod
    def _query_fallback_paper(tx, eid: str) -> Optional[PaperInfo]:
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
        return PaperInfo(
            element_id=rec["p_eid"],
            title=p.get("title") or "",
            conceptual_premises=p.get("conceptual_premises"),
            contextual_model_constraints=p.get("contextual_model_constraints"),
        )
