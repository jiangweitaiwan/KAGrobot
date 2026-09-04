# -*- coding: utf-8 -*-
"""graph_service/semantic_scorer.py — SemanticContext 為基礎的 RC 排名。

結合兩個訊號（權重寫死，不做前端可調）：
  - embed_score  : SemanticContext.embedding_text 的 SBERT embedding
                   與 RC retrieval_embedding 的 cosine 相似度
  - intent_score : intent_types 與 RC analysis_types 的重疊比例
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