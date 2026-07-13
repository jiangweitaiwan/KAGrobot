# -*- coding: utf-8 -*-
"""graph_service/semantic_scorer.py — SemanticContext 為基礎的 RC 排名。

結合兩個訊號（權重寫死，不做前端可調）：
  - embed_score  : SemanticContext.embedding_text 的 SBERT embedding
                   與 RC retrieval_embedding 的 cosine 相似度
  - intent_score : intent_types 與 RC analysis_types 的重疊比例

這裡獨立一份邏輯，不與 graph_service/scorer.py::PathScorer 共用，
因為 PathScorer 的職責是「已展開候選內部 path 重排」，
本模組是「全域 RC 排名」，兩者輸入輸出範圍不同，硬合併反而增加耦合。
"""

from __future__ import annotations

from typing import List

W_EMBED  = 0.7
W_INTENT = 0.3


def type_overlap(node_analysis_types: List[str], intent_types: List[str]) -> float:
    if not node_analysis_types or not intent_types:
        return 0.0
    matched = len(set(node_analysis_types) & set(intent_types))
    return matched / len(intent_types)


def combine_score(embed_score: float, intent_score: float) -> float:
    return W_EMBED * embed_score + W_INTENT * intent_score