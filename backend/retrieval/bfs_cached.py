# -*- coding: utf-8 -*-
"""retrieval/bfs_cached.py — 基于前驱图缓存的 BFS 实现。

从 GlobalPredecessorGraph 缓存中直接读取前驱/后继关系，
无需再查询 Neo4j，性能显著提升。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections import deque

from retrieval.path_selector import PathSelector
from retrieval.predecessor_graph import GlobalPredecessorGraph

logger = logging.getLogger(__name__)


def _upsert_concept_node(
    nodes_map: Dict[str, dict],
    eid: str,
    name: str,
    ctype: Optional[str],
    matched: bool = False,
) -> None:
    """向节点字典中插入或更新概念节点。"""
    if eid not in nodes_map:
        nodes_map[eid] = {
            "id": eid,
            "label": name or eid,
            "node_type": "concept",
            "concept_type": ctype,
            "matched": matched,
            "analysis_types": [],
            "meta": None,
        }
    elif matched:
        nodes_map[eid]["matched"] = True


def _upsert_paper_node(
    nodes_map: Dict[str, dict],
    eid: str,
    title: str,
    premises: Optional[str] = None,
    constraints: Optional[str] = None,
) -> None:
    """向节点字典中插入或更新论文节点。"""
    if eid not in nodes_map:
        nodes_map[eid] = {
            "id": eid,
            "label": title or eid,
            "node_type": "paper",
            "concept_type": None,
            "matched": False,
            "meta": {
                "conceptual_premises": premises,
                "contextual_model_constraints": constraints,
            },
        }


def _upsert_edge(
    edges_map: Dict[str, dict],
    source: str,
    target: str,
    paper_title: str = "",
) -> str:
    """向边字典中插入或更新边。"""
    eid = f"{source}__{target}"
    if eid not in edges_map:
        edges_map[eid] = {
            "id": eid,
            "source": source,
            "target": target,
            "paper_title": paper_title,
        }
    return eid


class UpstreamBFSFromCache:
    """
    从前驱图缓存中追溯上游（基础概念）。
    
    与原始 UpstreamBFS 不同，此版本不查询 Neo4j，
    而是从 GlobalPredecessorGraph 缓存中读取数据。
    """
    
    def __init__(
        self,
        pred_graph: GlobalPredecessorGraph,
        path_selector: PathSelector,
    ):
        self._pred_graph = pred_graph
        self._selector = path_selector
    
    def run(
        self,
        start_eid: str,
        max_depth: int = 10,
        max_branches: int = 4,
    ) -> Dict[str, Any]:
        """
        从前驱图缓存中展开上游路径。
        
        Parameters
        ----------
        start_eid : 起始节点 elementId
        max_depth : 最大深度
        max_branches : 最大分支数
        
        Returns
        -------
        Dict with keys: final_foundations, foundation_tree, 等
        """
        logger.debug(f"[UpstreamBFSFromCache] Starting from {start_eid}")
        
        queue = deque([(start_eid, [])])
        visited = {start_eid}
        all_paths: List[Dict[str, Any]] = []
        
        # BFS 遍历
        while queue:
            curr_eid, steps = queue.popleft()
            curr_info = self._pred_graph.get_node_info(curr_eid)
            
            # 达到深度限制
            if len(steps) >= max_depth:
                all_paths.append({
                    "steps": steps,
                    "final_concept": {
                        "element_id": curr_eid,
                        "name": curr_info.get('name', curr_eid),
                        "type": curr_info.get('type'),
                    },
                    "final_paper": None,
                })
                continue
            
            # ✨ 从缓存获取前驱（无 Neo4j 查询！）
            predecessors = self._pred_graph.get_predecessors(curr_eid, lazy_load=True)
            
            if not predecessors:
                # 叶子节点
                all_paths.append({
                    "steps": steps,
                    "final_concept": {
                        "element_id": curr_eid,
                        "name": curr_info.get('name', curr_eid),
                        "type": curr_info.get('type'),
                    },
                    "final_paper": None,
                })
                continue
            
            # BFS 继续
            for pred in predecessors:
                pred_eid = pred['eid']
                if pred_eid not in visited:
                    visited.add(pred_eid)
                    
                    new_step = {
                        "from_concept": {
                            "element_id": curr_eid,
                            "name": curr_info.get('name', curr_eid),
                            "type": curr_info.get('type'),
                        },
                        "to_concept": {
                            "element_id": pred_eid,
                            "name": pred['name'],
                            "type": pred['type'],
                        },
                        "via_paper": pred['via_paper'],
                    }
                    
                    queue.append((pred_eid, steps + [new_step]))
        
        # 去重
        seen: set = set()
        unique_paths: List[Dict[str, Any]] = []
        for p in all_paths:
            sig = tuple(p["steps"])
            if sig not in seen:
                seen.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)
        
        # 筛选路径
        selected_paths = self._selector.select(unique_paths, max_branches)
        
        # 构建节点和边
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        
        for path in selected_paths:
            if path.get("steps"):
                # 起始节点
                first_c = path["steps"][0]["from_concept"]
                _upsert_concept_node(
                    nodes_map,
                    first_c["element_id"],
                    first_c["name"],
                    first_c.get("type"),
                )
                
                # 逐步处理
                for step in path["steps"]:
                    to_c = step["to_concept"]
                    from_c = step["from_concept"]
                    paper = step.get("via_paper")
                    
                    if paper:
                        _upsert_paper_node(
                            nodes_map,
                            paper["eid"],
                            paper["title"],
                            paper.get("premises"),
                            paper.get("constraints"),
                        )
                        _upsert_concept_node(
                            nodes_map,
                            to_c["element_id"],
                            to_c["name"],
                            to_c.get("type"),
                        )
                        e1 = _upsert_edge(
                            edges_map,
                            from_c["element_id"],
                            paper["eid"],
                            paper["title"],
                        )
                        e2 = _upsert_edge(
                            edges_map,
                            paper["eid"],
                            to_c["element_id"],
                            paper["title"],
                        )
                    else:
                        _upsert_concept_node(
                            nodes_map,
                            to_c["element_id"],
                            to_c["name"],
                            to_c.get("type"),
                        )
                        e1 = _upsert_edge(
                            edges_map,
                            from_c["element_id"],
                            to_c["element_id"],
                            "",
                        )
        
        logger.debug(
            f"[UpstreamBFSFromCache] Result: {len(unique_paths)} paths, "
            f"{len(nodes_map)} nodes, {len(edges_map)} edges"
        )
        
        return {
            "final_foundations": self._extract_foundations(selected_paths),
            "foundation_tree": self._build_foundation_tree(selected_paths),
        }
    
    @staticmethod
    def _extract_foundations(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从路径中提取叶子节点（基础概念）。"""
        seen_leaf_eids: set = set()
        final_foundations: List[Dict[str, Any]] = []
        
        for path in paths:
            if not path.get("steps"):
                # 单个节点
                concept = path.get("final_concept", {})
                eid = concept.get("element_id")
                if eid and eid not in seen_leaf_eids:
                    seen_leaf_eids.add(eid)
                    final_foundations.append({
                        "element_id": eid,
                        "label": concept.get("name", eid),
                    })
            else:
                # 路径的最后一步的前驱
                first_step = path["steps"][0]
                concept = first_step["to_concept"]
                eid = concept["element_id"]
                if eid not in seen_leaf_eids:
                    seen_leaf_eids.add(eid)
                    final_foundations.append({
                        "element_id": eid,
                        "label": concept.get("name", eid),
                    })
        
        return final_foundations
    
    @staticmethod
    def _build_foundation_tree(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建基础概念树。"""
        tree_nodes: Dict[str, dict] = {}
        tree_edges: Dict[str, dict] = {}
        
        for path in paths:
            steps = path.get("steps", [])
            if not steps:
                continue
            
            for step in steps:
                for role in ("from_concept", "to_concept"):
                    c = step[role]
                    eid = c["element_id"]
                    if eid not in tree_nodes:
                        tree_nodes[eid] = {
                            "element_id": eid,
                            "label": c.get("name", eid),
                        }
                
                fc = step["from_concept"]
                tc = step["to_concept"]
                edge_key = f"{fc['element_id']}__{tc['element_id']}"
                if edge_key not in tree_edges:
                    tree_edges[edge_key] = {
                        "source": fc["element_id"],
                        "target": tc["element_id"],
                    }
        
        return {
            "nodes": list(tree_nodes.values()),
            "edges": list(tree_edges.values()),
        }


class DownstreamBFSFromCache:
    """
    从前驱图缓存中展开下游（结果概念）。
    """
    
    def __init__(self, pred_graph: GlobalPredecessorGraph):
        self._pred_graph = pred_graph
    
    def run(
        self,
        start_eid: str,
        rc_name: str = "",
        max_depth_down: int = 8,
    ) -> Dict[str, Any]:
        """
        从前驱图缓存中展开下游路径。
        
        Parameters
        ----------
        start_eid : 起始 RC 的 elementId
        rc_name : RC 名称（用于日志）
        max_depth_down : 最大深度
        
        Returns
        -------
        Dict with keys: nodes, edges, paths, 等
        """
        logger.debug(f"[DownstreamBFSFromCache] Starting downstream from {rc_name}")
        
        queue = deque([(start_eid, [])])
        visited = {start_eid}
        all_paths: List[Dict[str, Any]] = []
        
        # BFS 遍历
        while queue:
            curr_eid, steps = queue.popleft()
            curr_info = self._pred_graph.get_node_info(curr_eid)
            
            # 达到深度限制
            if len(steps) >= max_depth_down:
                all_paths.append({
                    "steps": steps,
                    "final_concept": {
                        "element_id": curr_eid,
                        "name": curr_info.get('name', curr_eid),
                        "type": curr_info.get('type'),
                    },
                    "final_paper": None,
                })
                continue
            
            # ✨ 从缓存获取后继
            successors = self._pred_graph.get_successors(curr_eid, lazy_load=True)
            
            if not successors:
                # 终点节点
                all_paths.append({
                    "steps": steps,
                    "final_concept": {
                        "element_id": curr_eid,
                        "name": curr_info.get('name', curr_eid),
                        "type": curr_info.get('type'),
                    },
                    "final_paper": None,
                })
                continue
            
            # BFS 继续
            for succ in successors:
                succ_eid = succ['eid']
                if succ_eid not in visited:
                    visited.add(succ_eid)
                    
                    new_step = {
                        "from_concept": {
                            "element_id": curr_eid,
                            "name": curr_info.get('name', curr_eid),
                            "type": curr_info.get('type'),
                        },
                        "to_concept": {
                            "element_id": succ_eid,
                            "name": succ['name'],
                            "type": succ['type'],
                        },
                        "via_paper": succ['via_paper'],
                    }
                    
                    queue.append((succ_eid, steps + [new_step]))
        
        # 去重并重新编号
        seen: set = set()
        unique_paths: List[Dict[str, Any]] = []
        for p in all_paths:
            sig = tuple(p["steps"])
            if sig not in seen:
                seen.add(sig)
                p["path_id"] = len(unique_paths)
                unique_paths.append(p)
        
        # 构建节点和边
        nodes_map: Dict[str, dict] = {}
        edges_map: Dict[str, dict] = {}
        
        _upsert_concept_node(
            nodes_map,
            start_eid,
            rc_name,
            "Resulting_Concept",
            matched=True,
        )
        
        for path in unique_paths[:3]:  # 只取 top-3 路径用于构建节点
            if path.get("steps"):
                for step in path["steps"]:
                    to_c = step["to_concept"]
                    from_c = step["from_concept"]
                    paper = step.get("via_paper")
                    
                    if paper:
                        _upsert_paper_node(
                            nodes_map,
                            paper["eid"],
                            paper["title"],
                            paper.get("premises"),
                            paper.get("constraints"),
                        )
                        _upsert_concept_node(
                            nodes_map,
                            to_c["element_id"],
                            to_c["name"],
                            to_c.get("type"),
                        )
                        e1 = _upsert_edge(
                            edges_map,
                            from_c["element_id"],
                            paper["eid"],
                            paper["title"],
                        )
                        e2 = _upsert_edge(
                            edges_map,
                            paper["eid"],
                            to_c["element_id"],
                            paper["title"],
                        )
                    else:
                        _upsert_concept_node(
                            nodes_map,
                            to_c["element_id"],
                            to_c["name"],
                            to_c.get("type"),
                        )
                        e1 = _upsert_edge(
                            edges_map,
                            from_c["element_id"],
                            to_c["element_id"],
                            "",
                        )
        
        logger.debug(
            f"[DownstreamBFSFromCache] Result: {len(unique_paths)} paths, "
            f"{len(nodes_map)} nodes"
        )
        
        return {
            "query_concepts": [{
                "id": start_eid,
                "name": rc_name,
                "type": "Resulting_Concept",
                "score": 1.0,
            }],
            "nodes": list(nodes_map.values()),
            "edges": list(edges_map.values()),
            "paths": unique_paths,
            "selected_path_id": 0 if unique_paths else None,
            "foundations": [],
            "foundation_tree": {"nodes": [], "edges": []},
        }