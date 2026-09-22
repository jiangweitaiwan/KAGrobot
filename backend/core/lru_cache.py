# -*- coding: utf-8 -*-
"""core/lru_cache.py — 共用 Thread-safe LRU Cache with TTL。

原本定義在 graph_service/bfs_cache.py 內、只供 AdjacencyCache 私用的
_LRUCache，抽成這裡的公開類別 LRUCache，讓 graph_service/
bfs_coordinator.py 的結果層 cache（rcpath_cache / expand_cache）
也能共用同一套快取機制（maxsize 各自依用途設定，TTL 統一走
DEFAULT_TTL_S，見兩邊呼叫端）。

行為與原本完全相同：
  - get() 回傳 (hit, value)，過期 entry 視同 miss 並刪除
  - set() 寫入時，超過 maxsize 淘汰最久未使用（LRU）的 entry
  - stats 提供 size / maxsize / hits / misses / hit_rate，
    供 /admin/cache/stats 統一格式使用
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Tuple

# 統一 TTL：7 天。所有使用 LRUCache 的地方（AdjacencyCache 的鄰接層、
# BFSCoordinator 的結果層）都應該用同一個值，避免「鄰接關係已經因為
# TTL 過期重新從 Neo4j 抓到最新資料，但聚合結果 cache 還在用舊資料
# 算出的答案」這種兩層資料不同步的情況。
DEFAULT_TTL_S = 7 * 24 * 3600


@dataclass
class _CacheEntry:
    """LRU cache 的內部 entry，含 TTL 與 LRU 排序用的 access_time。"""
    value:       Any
    created_at:  float
    access_time: float


class LRUCache:
    """Thread-safe LRU cache with TTL.

    Parameters
    ----------
    maxsize : int   最多保留幾個 entry（LRU 淘汰）
    ttl_s   : float TTL 秒數，預設 DEFAULT_TTL_S（7 天）
    """

    def __init__(self, maxsize: int = 500, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._maxsize = maxsize
        self._ttl     = ttl_s
        self._store:  Dict[str, _CacheEntry] = {}
        self._lock    = Lock()

        # 統計
        self._hits   = 0
        self._misses = 0

    # ── Public API ────────────────────────────────────────────

    def get(self, key: str) -> Tuple[bool, Any]:
        """回傳 (hit, value)。過期 entry 視同 miss 並刪除。"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return False, None

            now = time.monotonic()
            if now - entry.created_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return False, None

            entry.access_time = now
            self._hits += 1
            return True, entry.value

    def set(self, key: str, value: Any) -> None:
        """寫入 cache，必要時淘汰最舊的 entry。"""
        with self._lock:
            now = time.monotonic()
            if key in self._store:
                self._store[key].value       = value
                self._store[key].access_time = now
                return

            if len(self._store) >= self._maxsize:
                # 淘汰 access_time 最小（最久未使用）的 entry
                lru_key = min(self._store, key=lambda k: self._store[k].access_time)
                del self._store[lru_key]

            self._store[key] = _CacheEntry(
                value=value,
                created_at=now,
                access_time=now,
            )

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":     len(self._store),
                "maxsize":  self._maxsize,
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }
