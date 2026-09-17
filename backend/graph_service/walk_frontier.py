# -*- coding: utf-8 -*-
"""graph_service/walk_frontier.py — Random Walk Mode 的 frontier_fetcher 真實實作。

把 core/walk_bot.py 需要的 frontier_fetcher 介面接上真實資料源：

  1. AdjacencyCache.get_downstream_neighbors()：取得一跳下游鄰居。
     刻意不用 BFSCoordinator.downstream()（那是多跳展開到 RC 的完整
     DownstreamBFS，回傳的是完整 path 集合，不是「下一步有哪些選項」）。
     Random walk 每一步只需要一跳，AdjacencyCache 本身就有 LRU+TTL 快取，
     且與 dialogue mode 共用同一份快取實例（見 app.py 的 adjacency_cache
     單例）——同一個節點不論在哪個 mode 被展開過都能互相命中，不重複打
     Neo4j。

  2. 一次批次 Cypher 查詢，補齊候選節點的 definition/mechanism。
     AdjacencyCache 回傳的 ConceptInfo 只有 element_id/name/node_type
     （見 bfs_cache.py），不含 definition/mechanism——它的職責是「鄰接
     關係」，不是「節點詳情」。這裡刻意不擴充 AdjacencyCache 的回傳格式
     去塞這兩個欄位，避免鄰接層背負跟它職責無關的欄位，改在這一層另外
     用 UNWIND 批次查一次，維持兩層各自的職責邊界（同樣的分層原則見
     bfs_cache.py 檔頭說明）。

  3. sbert prefilter：仿照 api/routes/concept.py 的 _sbert_prefilter，
     依「目前對話累積的語意脈絡」排序候選，只取 top_n 塞進 LLM prompt。

介面設計（呼叫端見 core/walk_bot.py）：
    frontier_fetcher(current_eid: str, context_query: str) -> List[FrontierCandidate]

  刻意把 context_query 一併傳入，而不是只傳 current_eid——「下游有哪些
  節點」跟「LLM 該優先看哪些候選」是兩件事，若不依語意脈絡排序，
  frontier 選項多的時候，真正相關的候選可能被截在 top_n 之外，永遠不會
  被 LLM 看到。

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from graph_service.walk_state import FrontierCandidate

if TYPE_CHECKING:
    from graph_service.bfs_cache import AdjacencyCache
    from retrieval.encoder import SbertOnnxEncoder

_DEFAULT_TOP_N = 6


# ══════════════════════════════════════════════════════════════
# Public factory
# ══════════════════════════════════════════════════════════════

def make_frontier_fetcher(
    adjacency_cache: "AdjacencyCache",
    driver:          Any,
    encoder:         "SbertOnnxEncoder",
    top_n:           int = _DEFAULT_TOP_N,
) -> Callable[[str, str], List[FrontierCandidate]]:
    """組出可直接餵給 WalkTurnRunner 的 frontier_fetcher closure。

    Parameters
    ----------
    adjacency_cache : 既有 app.state 的 AdjacencyCache 單例（見 app.py）。
    driver          : Neo4j driver，僅用於批次補齊 definition/mechanism。
    encoder         : SbertOnnxEncoder 單例。
    top_n           : sbert prefilter 後保留的候選數量上限。
    """

    def frontier_fetcher(current_eid: str, context_query: str) -> List[FrontierCandidate]:
        neighbor_infos = _unique_downstream_concepts(adjacency_cache, current_eid)
        if not neighbor_infos:
            return []

        details = _fetch_concept_details(driver, [n["eid"] for n in neighbor_infos])

        candidates = [
            FrontierCandidate(
                eid=n["eid"],
                label=n["name"],
                definition=details.get(n["eid"], {}).get("definition"),
                function=details.get(n["eid"], {}).get("function"),
                node_type=n["node_type"],
            )
            for n in neighbor_infos
        ]

        return _prefilter_by_semantic(candidates, context_query, encoder, top_n)

    return frontier_fetcher


# ══════════════════════════════════════════════════════════════
# Step 1：取得一跳下游鄰居，依 element_id 去重
# ══════════════════════════════════════════════════════════════

def _unique_downstream_concepts(
    adjacency_cache: "AdjacencyCache",
    current_eid:     str,
) -> List[Dict[str, Any]]:
    """AdjacencyCache.get_downstream_neighbors() 回傳的清單可能因為
    「同一個下游概念經由不同 paper 都能到達」而重複，這裡依 element_id
    去重，只保留第一次出現的——via_paper 差異在 frontier 這層不重要，
    那是 downstream BFS 組完整 path 時才需要的資訊。
    """
    neighbors = adjacency_cache.get_downstream_neighbors(current_eid)
    seen: Dict[str, Dict[str, Any]] = {}
    for n in neighbors:
        eid = n.to_concept.element_id
        if eid not in seen:
            seen[eid] = {
                "eid":       eid,
                "name":      n.to_concept.name,
                "node_type": n.to_concept.node_type,
            }
    return list(seen.values())


# ══════════════════════════════════════════════════════════════
# Step 2：批次補齊 definition/mechanism（避免 N+1 查詢）
# ══════════════════════════════════════════════════════════════

def _fetch_concept_details(
    driver: Any,
    eids:   List[str],
) -> Dict[str, Dict[str, Optional[str]]]:
    if not eids:
        return {}

    with driver.session() as session:
        rows = session.run(
            """
            UNWIND $eids AS eid
            MATCH (c:Concept) WHERE elementId(c) = eid
            RETURN elementId(c) AS eid,
                   c.definition AS definition,
                   c.function   AS function
            """,
            eids=eids,
        ).data()

    return {
        row["eid"]: {"definition": row["definition"], "function": row["function"]}
        for row in rows
    }


# ══════════════════════════════════════════════════════════════
# Step 3：sbert prefilter（比照 api/routes/concept.py 的 _sbert_prefilter）
# ══════════════════════════════════════════════════════════════

def _prefilter_by_semantic(
    candidates:    List[FrontierCandidate],
    context_query: str,
    encoder:       "SbertOnnxEncoder",
    top_n:         int,
) -> List[FrontierCandidate]:
    if len(candidates) <= top_n:
        return candidates

    query_text = (context_query or "").strip()
    if not query_text:
        return candidates[:top_n]

    try:
        query_emb = encoder.encode(query_text)[0]
        texts = [
            f"{c.label} {c.function or c.definition or ''}".strip()
            for c in candidates
        ]
        cand_embs = encoder.encode(texts)
        scores    = cand_embs @ query_emb
        order     = np.argsort(scores)[::-1][:top_n]
        return [candidates[i] for i in order]
    except Exception as exc:
        print(f"[walk_frontier] sbert prefilter failed: {exc}, "
              f"falling back to first {top_n}")
        return candidates[:top_n]
