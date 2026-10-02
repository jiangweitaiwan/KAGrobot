# -*- coding: utf-8 -*-
"""graph_service/bfs_coordinator.py — BFS 統一排程器

職責：
  - 對外唯一介面，取代原本直接使用 DownstreamBFS / UpstreamBFS 的呼叫
  - 管理兩層 cache：
      1. 鄰接層（AdjacencyCache）：node_id → 鄰接關係，跨 request 共享
      2. 結果層（rcpath_cache / expand_cache）：eid(+參數) → 完整展開結果，
         避免重複 BFS
  - 結果層統一在此層管理（取代 app.state.expand_cache / rcpath_cache）
  - 提供 cache stats endpoint 用的統計資訊

呼叫端替換方式：
  原本：
    downstream_bfs.run(eid, rc_name)
    upstream_bfs.run(eid)
  改為：
    coordinator.downstream(eid, rc_name)
    coordinator.upstream(eid)

【重構 — cache 機制統一】
  結果層原本是各自手刻的「裸 dict + threading.Lock」，沒有 maxsize、
  沒有 TTL，長時間運行會無界成長。現在改用與 AdjacencyCache 相同的
  core/lru_cache.LRUCache（LRU + TTL + 統一格式的 stats），TTL 統一走
  DEFAULT_TTL_S（7 天），避免「鄰接層已因 TTL 過期重抓最新資料，
  但結果層聚合快取還停在舊資料」這種兩層不同步的情況。maxsize 依
  聚合結果體積另外調小（單一 entry 比 AdjacencyCache 的鄰接清單重
  很多），預設各 200。

  同時，cache key 明確把會影響結果的參數（max_depth / max_branches）
  編進去（例如 f"{eid}:{max_depth}"），不再依賴「所有呼叫端剛好都用
  同一組預設深度參數」這個沒有型別保證的隱性假設——之後若有新呼叫點
  傳入不同深度，也不會誤命中錯誤深度的快取結果。

  原本的 _result_cache / expand_one_cached / expand_one_async 已移除：
  確認沒有任何路由呼叫（純死碼），而且它內部直接呼叫
  self._down_bfs.run() / self._up_bfs.run()，完全繞過
  get_rcpath_cached / get_expand_cached，等於跟其餘兩層結果快取各自
  為政、無法互相命中，本身就是需要一併清掉的重工來源。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

from core.infra.lru_cache            import DEFAULT_TTL_S, LRUCache
from graph_service.bfs_cache   import AdjacencyCache
from graph_service.bfs_engine  import CachedDownstreamBFS, CachedUpstreamBFS
from retrieval.path_selector   import PathSelector


class BFSCoordinator:
    """BFS 統一排程器，管理鄰接層與結果層兩層 cache。

    Parameters
    ----------
    adjacency_cache : AdjacencyCache  鄰接關係 cache（注入）
    path_selector   : PathSelector    upstream path 選取策略（注入）
    rcpath_cache_maxsize : rcpath_cache 的 LRU 上限（預設 200）
    expand_cache_maxsize : expand_cache 的 LRU 上限（預設 200）
    """

    def __init__(
        self,
        adjacency_cache: AdjacencyCache,
        path_selector:   PathSelector,
        *,
        rcpath_cache_maxsize: int = 200,
        expand_cache_maxsize: int = 200,
    ) -> None:
        self._adj_cache = adjacency_cache
        self._down_bfs  = CachedDownstreamBFS(adjacency_cache)
        self._up_bfs    = CachedUpstreamBFS(adjacency_cache, path_selector)

        # 結果層 cache（LRU + TTL，統一 DEFAULT_TTL_S=7天，見檔頭說明）。
        # key 明確編入會影響結果的參數，見 get_rcpath_cached / get_expand_cached。
        self._rcpath_cache: LRUCache = LRUCache(
            maxsize=rcpath_cache_maxsize, ttl_s=DEFAULT_TTL_S,
        )
        self._expand_cache: LRUCache = LRUCache(
            maxsize=expand_cache_maxsize, ttl_s=DEFAULT_TTL_S,
        )

    # ──────────────────────────────────────────────────────────
    # Public API — 基本 BFS
    # ──────────────────────────────────────────────────────────

    def downstream(
        self,
        eid:       str,
        rc_name:   str = "",
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """Downstream BFS，結果不快取（呼叫端自行決定是否快取）。"""
        return self._down_bfs.run(eid, rc_name, max_depth)

    def upstream(
        self,
        eid:          str,
        max_depth:    int = 10,
        max_branches: int = 4,
        start_name:   str = "",
    ) -> Dict[str, Any]:
        """Upstream BFS，結果不快取（呼叫端自行決定是否快取）。"""
        return self._up_bfs.run(eid, max_depth, max_branches, start_name=start_name)

    # ──────────────────────────────────────────────────────────
    # Public API — 結果層 cache（取代 app.state 的各種 cache）
    # ──────────────────────────────────────────────────────────

    def get_rcpath_cached(
        self,
        eid:       str,
        rc_name:   str = "",
        max_depth: int = 6,
    ) -> Dict[str, Any]:
        """Downstream BFS 結果快取，對應 /graph/rc-paths/{eid}。

        取代 app.state.rcpath_cache。key 含 max_depth，避免不同深度的
        呼叫誤命中彼此的快取結果。
        """
        key = f"{eid}:{max_depth}"
        hit, value = self._rcpath_cache.get(key)
        if hit:
            return value

        result = self._down_bfs.run(eid, rc_name, max_depth)
        self._rcpath_cache.set(key, result)
        return result

    def get_expand_cached(
        self,
        eid:          str,
        max_depth:    int = 6,
        max_branches: int = 4,
        start_name:   str = "",
    ) -> Dict[str, Any]:
        """Upstream BFS 結果快取，對應 /graph/expand/{eid}。

        取代 app.state.expand_cache。key 含 max_depth + max_branches，
        理由同 get_rcpath_cached。
        """
        key = f"{eid}:{max_depth}:{max_branches}"
        hit, value = self._expand_cache.get(key)
        if hit:
            return value

        result = self._up_bfs.run(eid, max_depth, max_branches, start_name=start_name)
        self._expand_cache.set(key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # Public API — 非同步包裝（供 GraphExpander 使用）
    # ──────────────────────────────────────────────────────────

    async def downstream_async(
        self,
        eid:       str,
        rc_name:   str = "",
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.downstream, eid, rc_name, max_depth)

    async def upstream_async(
        self,
        eid:          str,
        max_depth:    int = 10,
        max_branches: int = 4,
        start_name:   str = "",
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.upstream, eid, max_depth, max_branches, start_name,
        )

    async def get_rcpath_cached_async(
        self,
        eid:       str,
        rc_name:   str = "",
        max_depth: int = 6,
    ) -> Dict[str, Any]:
        """get_rcpath_cached 的非同步版本。

        main_agent 與 /graph/rc-paths/{eid} 共用此入口與同一個
        _rcpath_cache，確保兩邊算出的 path_id 定址空間完全一致。
        """
        return await asyncio.to_thread(self.get_rcpath_cached, eid, rc_name, max_depth)

    async def get_expand_cached_async(
        self,
        eid:          str,
        max_depth:    int = 6,
        max_branches: int = 4,
        start_name:   str = "",
    ) -> Dict[str, Any]:
        """get_expand_cached 的非同步版本，供 /graph/expand/batch 使用。"""
        return await asyncio.to_thread(
            self.get_expand_cached, eid, max_depth, max_branches, start_name,
        )

    # ──────────────────────────────────────────────────────────
    # Stats & debug
    # ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """回傳各層 cache 的統計資訊，供 /admin/cache/stats 使用。"""
        return {
            "adjacency_cache": self._adj_cache.stats,
            "rcpath_cache":    self._rcpath_cache.stats,
            "expand_cache":    self._expand_cache.stats,
        }
