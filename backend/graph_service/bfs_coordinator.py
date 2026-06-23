# -*- coding: utf-8 -*-
"""graph_service/bfs_coordinator.py — BFS 統一排程器

職責：
  - 對外唯一介面，取代原本直接使用 DownstreamBFS / UpstreamBFS 的呼叫
  - 管理兩層 cache：
      1. 鄰接層（AdjacencyCache）：node_id → 鄰接關係，跨 request 共享
      2. 結果層（expand_cache）  ：eid → 完整展開結果，避免重複 BFS
  - expand_cache 統一在此層管理（取代 app.state.expand_cache / rcpath_cache）
  - 提供 cache stats endpoint 用的統計資訊

呼叫端替換方式：
  原本：
    downstream_bfs.run(eid, rc_name)
    upstream_bfs.run(eid)
  改為：
    coordinator.downstream(eid, rc_name)
    coordinator.upstream(eid)
    coordinator.expand_one(eid, rc_name)   ← downstream + upstream 合併
"""

from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from graph_service.bfs_cache  import AdjacencyCache
from graph_service.bfs_engine import CachedDownstreamBFS, CachedUpstreamBFS
from retrieval.path_selector  import PathSelector


class BFSCoordinator:
    """BFS 統一排程器，管理鄰接層與結果層兩層 cache。

    Parameters
    ----------
    adjacency_cache : AdjacencyCache  鄰接關係 cache（注入）
    path_selector   : PathSelector    upstream path 選取策略（注入）
    """

    def __init__(
        self,
        adjacency_cache: AdjacencyCache,
        path_selector:   PathSelector,
    ) -> None:
        self._adj_cache = adjacency_cache
        self._down_bfs  = CachedDownstreamBFS(adjacency_cache)
        self._up_bfs    = CachedUpstreamBFS(adjacency_cache, path_selector)

        # 結果層 cache（process 生命週期，重啟即清空）
        # key: eid → (downstream_result, upstream_result)
        self._result_cache: Dict[str, Tuple[Dict, Dict]] = {}
        self._result_lock   = Lock()

        # 獨立的 rcpath cache（只有 downstream，供 routes_brain 使用）
        # 保持與原 app.state.rcpath_cache 相同語意
        self._rcpath_cache: Dict[str, Dict[str, Any]] = {}
        self._rcpath_lock   = Lock()

        # expand cache（downstream + upstream 合併結果，供 routes_brain /graph/expand 使用）
        self._expand_cache: Dict[str, Dict[str, Any]] = {}
        self._expand_lock   = Lock()

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
    ) -> Dict[str, Any]:
        """Upstream BFS，結果不快取（呼叫端自行決定是否快取）。"""
        return self._up_bfs.run(eid, max_depth, max_branches)

    # ──────────────────────────────────────────────────────────
    # Public API — 結果層 cache（取代 app.state 的各種 cache）
    # ──────────────────────────────────────────────────────────

    def expand_one_cached(
        self,
        eid:     str,
        rc_name: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Downstream + Upstream 合併展開，結果快取。

        對應原 GraphExpander._expand_one 的單一 eid 展開邏輯。
        第一次呼叫打 Neo4j（透過鄰接層 cache），後續直接回傳結果。

        Returns
        -------
        (downstream_result, upstream_result)
        """
        with self._result_lock:
            if eid in self._result_cache:
                return self._result_cache[eid]

        # cache miss：同步執行（已在 asyncio.to_thread 中被呼叫）
        down_result = self._down_bfs.run(eid, rc_name)
        up_result   = self._up_bfs.run(eid)

        with self._result_lock:
            self._result_cache[eid] = (down_result, up_result)

        return down_result, up_result

    def get_rcpath_cached(
        self,
        eid:     str,
        rc_name: str = "",
        max_depth: int = 6,
    ) -> Dict[str, Any]:
        """Downstream BFS 結果快取，對應 /graph/rc-paths/{eid}。

        取代 app.state.rcpath_cache。
        """
        with self._rcpath_lock:
            if eid in self._rcpath_cache:
                return self._rcpath_cache[eid]

        result = self._down_bfs.run(eid, rc_name, max_depth)

        with self._rcpath_lock:
            self._rcpath_cache[eid] = result

        return result

    def get_expand_cached(
        self,
        eid:          str,
        max_depth:    int = 6,
        max_branches: int = 4,
    ) -> Dict[str, Any]:
        """Upstream BFS 結果快取，對應 /graph/expand/{eid}。

        取代 app.state.expand_cache。
        """
        with self._expand_lock:
            if eid in self._expand_cache:
                return self._expand_cache[eid]

        result = self._up_bfs.run(eid, max_depth, max_branches)

        with self._expand_lock:
            self._expand_cache[eid] = result

        return result

    # ──────────────────────────────────────────────────────────
    # Public API — 非同步包裝（供 GraphExpander 使用）
    # ──────────────────────────────────────────────────────────

    async def expand_one_async(
        self,
        eid:     str,
        rc_name: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """expand_one_cached 的非同步版本，避免阻塞 event loop。"""
        return await asyncio.to_thread(
            self.expand_one_cached, eid, rc_name
        )

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
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.upstream, eid, max_depth, max_branches)

    # ──────────────────────────────────────────────────────────
    # Stats & debug
    # ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """回傳各層 cache 的統計資訊，供 /admin/cache/stats 使用。"""
        with self._result_lock:
            result_size = len(self._result_cache)
        with self._rcpath_lock:
            rcpath_size = len(self._rcpath_cache)
        with self._expand_lock:
            expand_size = len(self._expand_cache)

        return {
            "adjacency_cache": self._adj_cache.stats,
            "result_cache":    {"size": result_size},
            "rcpath_cache":    {"size": rcpath_size},
            "expand_cache":    {"size": expand_size},
        }
