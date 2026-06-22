# -*- coding: utf-8 -*-
"""retrieval/predecessor_graph.py — 全局前驱图缓存（Lazy Loading）。

特点：
  1. Lazy Loading：只在访问某个节点时才加载其前驱/后继
  2. 线程安全：使用 RLock 保护并发访问
  3. TTL 支持：缓存可设置过期时间，自动刷新
  4. 版本管理：检测 Neo4j 数据变化，必要时全量刷新
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Set, Tuple
from threading import RLock
from collections import defaultdict

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GlobalPredecessorGraph:
    """
    全局前驱图缓存 - Lazy Loading 版本
    
    数据结构：
    {
        'node_eid': {
            'info': {
                'eid': str,
                'name': str,
                'type': str,
            },
            'predecessors': [
                {
                    'eid': str,
                    'name': str,
                    'type': str,
                    'via_paper': {...} or None,
                    'edge_type': 'reasoning' | 'paper',
                },
                ...
            ],
            'successors': [...],  # 对称结构
            'loaded': bool,  # 是否已加载该节点的邻接关系
        },
        ...
    }
    """
    
    def __init__(
        self,
        driver,
        ttl_hours: int = 24,
        max_lazy_batch: int = 100,
    ):
        """
        Parameters
        ----------
        driver : Neo4j driver
        ttl_hours : 缓存过期时间（小时），0 表示不过期
        max_lazy_batch : 单次 lazy 加载的最大节点数
        """
        self._driver = driver
        self._ttl_seconds = ttl_hours * 3600
        self._max_lazy_batch = max_lazy_batch
        
        # 缓存
        self._graph: Dict[str, dict] = {}
        self._loaded_eids: Set[str] = set()  # 已加载前驱/后继的节点
        
        # 元数据
        self._initialized = False
        self._built_at: Optional[float] = None
        self._version_in_cache: Optional[int] = None
        
        # 线程安全
        self._lock = RLock()
    
    # ── Public API ────────────────────────────────────────────
    
    def initialize(self) -> None:
        """
        应用启动时调用一次，加载所有节点的基本信息（不加载邻接关系）。
        这样可以快速初始化，同时支持后续 lazy loading。
        """
        with self._lock:
            if self._initialized:
                logger.warning("[PredecessorGraph] Already initialized, skipping")
                return
            
            logger.info("[PredecessorGraph] Initializing (loading node info)...")
            start = time.time()
            
            try:
                with self._driver.session() as session:
                    # 1. 检查版本
                    self._check_and_update_version(session)
                    
                    # 2. 加载所有节点的基本信息
                    nodes = session.run("""
                        MATCH (c:Concept)
                        RETURN elementId(c) AS eid, 
                               c.name AS name,
                               c.type AS type
                    """).data()
                    
                    for node in nodes:
                        eid = node['eid']
                        self._graph[eid] = {
                            'info': {
                                'eid': eid,
                                'name': node['name'],
                                'type': node['type'],
                            },
                            'predecessors': [],
                            'successors': [],
                            'loaded': False,  # ✨ 标记为未加载邻接关系
                        }
                
                self._initialized = True
                self._built_at = time.time()
                elapsed = time.time() - start
                logger.info(
                    f"[PredecessorGraph] Initialized in {elapsed:.2f}s, "
                    f"{len(self._graph)} nodes"
                )
            
            except Exception as e:
                logger.error(f"[PredecessorGraph] Initialization failed: {e}")
                raise
    
    def get_predecessors(self, eid: str, lazy_load: bool = True) -> List[dict]:
        """
        获取某节点的直接前驱。
        
        Parameters
        ----------
        eid : 节点 elementId
        lazy_load : 如果节点未加载，是否自动加载（True）或返回空（False）
        
        Returns
        -------
        前驱列表，若不存在返回空列表
        """
        with self._lock:
            # 检查缓存是否过期
            self._check_ttl()
            
            # 确保节点存在且已初始化
            if eid not in self._graph:
                if lazy_load:
                    self._lazy_load_node(eid)
                else:
                    return []
            
            # 如果该节点的邻接关系未加载，进行 lazy loading
            if not self._graph[eid]['loaded']:
                if lazy_load:
                    self._lazy_load_adjacent(eid)
                else:
                    return []
            
            return self._graph[eid]['predecessors'].copy()
    
    def get_successors(self, eid: str, lazy_load: bool = True) -> List[dict]:
        """获取某节点的直接后继。"""
        with self._lock:
            self._check_ttl()
            
            if eid not in self._graph:
                if lazy_load:
                    self._lazy_load_node(eid)
                else:
                    return []
            
            if not self._graph[eid]['loaded']:
                if lazy_load:
                    self._lazy_load_adjacent(eid)
                else:
                    return []
            
            return self._graph[eid]['successors'].copy()
    
    def get_node_info(self, eid: str, lazy_load: bool = True) -> dict:
        """获取节点基本信息。"""
        with self._lock:
            self._check_ttl()
            
            if eid not in self._graph:
                if lazy_load:
                    self._lazy_load_node(eid)
                else:
                    return {}
            
            return self._graph[eid]['info'].copy()
    
    def preload_nodes(self, eids: List[str]) -> None:
        """
        预加载一批节点的邻接关系。
        用于性能优化：在 BFS 前预加载可能用到的节点。
        """
        with self._lock:
            self._check_ttl()
            
            # 过滤出未加载的节点
            to_load = [eid for eid in eids if eid in self._graph and not self._graph[eid]['loaded']]
            
            if not to_load:
                return
            
            logger.debug(f"[PredecessorGraph] Preloading {len(to_load)} nodes")
            
            for batch_eids in self._batch(to_load, self._max_lazy_batch):
                self._load_adjacent_batch(batch_eids)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息。"""
        with self._lock:
            total_nodes = len(self._graph)
            loaded_nodes = len(self._loaded_eids)
            total_edges = sum(
                len(n['predecessors']) for n in self._graph.values()
            )
            
            return {
                'total_nodes': total_nodes,
                'loaded_nodes': loaded_nodes,
                'loading_rate': (loaded_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                'total_edges': total_edges,
                'initialized': self._initialized,
                'is_stale': self._is_stale(),
                'built_at': self._built_at,
                'ttl_hours': self._ttl_seconds / 3600,
            }
    
    # ── Private helpers ────────────────────────────────────────
    
    def _check_ttl(self) -> None:
        """检查缓存是否过期。"""
        if self._ttl_seconds == 0:
            return  # TTL 为 0 表示永不过期
        
        if self._built_at is None:
            return
        
        if time.time() - self._built_at > self._ttl_seconds:
            logger.warning("[PredecessorGraph] Cache expired, clearing...")
            self.clear()
    
    def _is_stale(self) -> bool:
        """检查缓存是否过期。"""
        if self._ttl_seconds == 0:
            return False
        if self._built_at is None:
            return True
        return time.time() - self._built_at > self._ttl_seconds
    
    def _check_and_update_version(self, session) -> None:
        """检查 Neo4j 中的版本号，如果不同则标记需要刷新。"""
        try:
            # 这里假设有一个 GraphMetadata 节点存储版本
            # 如果没有，跳过版本检查
            result = session.run("""
                MATCH (m:GraphMetadata)
                RETURN m.version AS version
            """).single()
            
            if result:
                version = result['version']
                if (self._version_in_cache is not None and 
                    version != self._version_in_cache):
                    logger.warning(
                        f"[PredecessorGraph] Version mismatch: "
                        f"cache={self._version_in_cache}, db={version}"
                    )
                    # 触发完全刷新
                    self.clear()
                else:
                    self._version_in_cache = version
        except Exception as e:
            logger.debug(f"[PredecessorGraph] Version check failed (expected): {e}")
    
    def _lazy_load_node(self, eid: str) -> None:
        """
        Lazy load 单个节点的基本信息。
        如果节点不存在，从 Neo4j 加载。
        """
        if eid in self._graph:
            return
        
        logger.debug(f"[PredecessorGraph] Lazy loading node info for {eid}")
        
        try:
            with self._driver.session() as session:
                node = session.run("""
                    MATCH (c:Concept)
                    WHERE elementId(c) = $eid
                    RETURN elementId(c) AS eid,
                           c.name AS name,
                           c.type AS type
                """, eid=eid).single()
                
                if node:
                    self._graph[eid] = {
                        'info': {
                            'eid': eid,
                            'name': node['name'],
                            'type': node['type'],
                        },
                        'predecessors': [],
                        'successors': [],
                        'loaded': False,
                    }
                else:
                    # 节点不存在，创建空占位符
                    self._graph[eid] = {
                        'info': {
                            'eid': eid,
                            'name': f'(Unknown {eid[:8]})',
                            'type': 'unknown',
                        },
                        'predecessors': [],
                        'successors': [],
                        'loaded': True,  # 标记为已处理，避免重复查询
                    }
        except Exception as e:
            logger.error(f"[PredecessorGraph] Failed to lazy load node {eid}: {e}")
    
    def _lazy_load_adjacent(self, eid: str) -> None:
        """
        Lazy load 单个节点的前驱/后继关系。
        """
        if eid not in self._graph or self._graph[eid]['loaded']:
            return
        
        logger.debug(f"[PredecessorGraph] Lazy loading adjacent for {eid}")
        self._load_adjacent_batch([eid])
    
    def _load_adjacent_batch(self, eids: List[str]) -> None:
        """
        批量加载多个节点的前驱/后继关系。
        """
        if not eids:
            return
        
        try:
            with self._driver.session() as session:
                # ✨ 核心查询：一次性获取所有前驱关系
                edges = session.run("""
                    UNWIND $eids AS eid
                    MATCH (curr:Concept) WHERE elementId(curr) = eid
                    
                    // 模式 A：直接 REASONING
                    OPTIONAL MATCH (prev_a:Concept)-[:REASONING]->(curr)
                    WHERE elementId(prev_a) <> eid
                    
                    // 模式 B：通过 Paper
                    OPTIONAL MATCH (prev_b:Concept)-[:INSPIRES|INDIRECT_INSPIRES|REASONING]->
                                   (p:Paper)-[:PRODUCES]->(curr)
                    WHERE elementId(prev_b) <> eid
                    
                    WITH eid, curr, prev_a, prev_b, p
                    
                    RETURN eid,
                           CASE 
                               WHEN prev_a IS NOT NULL 
                               THEN elementId(prev_a)
                               WHEN prev_b IS NOT NULL
                               THEN elementId(prev_b)
                               ELSE null
                           END AS prev_eid,
                           CASE 
                               WHEN prev_a IS NOT NULL THEN prev_a
                               WHEN prev_b IS NOT NULL THEN prev_b
                               ELSE null
                           END AS prev_node,
                           p AS paper,
                           CASE 
                               WHEN prev_a IS NOT NULL THEN 'reasoning'
                               WHEN prev_b IS NOT NULL THEN 'paper'
                               ELSE null
                           END AS edge_type
                """, eids=eids).data()
                
                # 处理结果
                edges_by_curr = defaultdict(list)
                for edge in edges:
                    if edge['prev_eid'] is not None:
                        edges_by_curr[edge['eid']].append({
                            'prev_eid': edge['prev_eid'],
                            'prev_node': edge['prev_node'],
                            'paper': edge['paper'],
                            'edge_type': edge['edge_type'],
                        })
                
                # 更新缓存
                for eid in eids:
                    if eid not in self._graph:
                        continue
                    
                    predecessors = []
                    successors = []
                    
                    for edge_info in edges_by_curr.get(eid, []):
                        prev_eid = edge_info['prev_eid']
                        prev_node = edge_info['prev_node']
                        paper = edge_info['paper']
                        edge_type = edge_info['edge_type']
                        
                        # 确保前驱节点在缓存中
                        if prev_eid not in self._graph:
                            self._graph[prev_eid] = {
                                'info': {
                                    'eid': prev_eid,
                                    'name': prev_node.get('name') if prev_node else f'(Unknown {prev_eid[:8]})',
                                    'type': prev_node.get('type') if prev_node else 'unknown',
                                },
                                'predecessors': [],
                                'successors': [],
                                'loaded': False,
                            }
                        
                        # 构建前驱记录
                        pred_record = {
                            'eid': prev_eid,
                            'name': prev_node.get('name') if prev_node else self._graph[prev_eid]['info']['name'],
                            'type': prev_node.get('type') if prev_node else self._graph[prev_eid]['info']['type'],
                            'via_paper': None,
                            'edge_type': edge_type,
                        }
                        
                        if paper:
                            pred_record['via_paper'] = {
                                'eid': paper.element_id,
                                'title': paper.get('title'),
                                'premises': paper.get('conceptual_premises'),
                                'constraints': paper.get('contextual_model_constraints'),
                            }
                        
                        predecessors.append(pred_record)
                        
                        # 构建后继记录（对称）
                        succ_record = {
                            'eid': eid,
                            'name': self._graph[eid]['info']['name'],
                            'type': self._graph[eid]['info']['type'],
                            'via_paper': pred_record['via_paper'],
                            'edge_type': edge_type,
                        }
                        
                        # 更新前驱节点的后继
                        successors_list = self._graph[prev_eid]['successors']
                        if not any(s['eid'] == eid for s in successors_list):
                            successors_list.append(succ_record)
                    
                    # 更新当前节点的前驱
                    self._graph[eid]['predecessors'] = predecessors
                    self._graph[eid]['loaded'] = True
                    self._loaded_eids.add(eid)
        
        except Exception as e:
            logger.error(f"[PredecessorGraph] Failed to load adjacent batch: {e}")
    
    def clear(self) -> None:
        """清空缓存并重置状态。"""
        with self._lock:
            self._graph.clear()
            self._loaded_eids.clear()
            self._initialized = False
            self._built_at = None
            logger.info("[PredecessorGraph] Cache cleared")
    
    @staticmethod
    def _batch(items: List, batch_size: int):
        """将列表分批。"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]