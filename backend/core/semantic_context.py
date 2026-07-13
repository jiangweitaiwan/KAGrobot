# -*- coding: utf-8 -*-
"""core/semantic_context.py — Semantic 機制的核心資料結構。

SemanticContext 統合每個 session 目前的語意上下文：
  - query          : socratic 提煉出的英文查詢語句（既有）
  - intent_types   : 使用者需求對應的分析意圖類型（既有）
  - summary        : 每輪滾動更新的對話需求摘要（新增）

供以下用途消費：
  - RC 語意排名（graph_service/semantic_scorer.py）
  - Foundation 語意推薦（graph_service/foundation_recommender.py）
  - 前端 Semantic Panel 直接展示
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SemanticContext:
    query:        str       = ""
    intent_types: List[str] = field(default_factory=list)
    summary:      str       = ""

    @property
    def embedding_text(self) -> str:
        """組合 query + summary，供 embedding 相似度計算使用。

        summary 是每輪重寫的對話需求摘要，涵蓋比單一 query 更完整的
        上下文重心；query 仍保留，因為它是 socratic 階段刻意提煉過的
        精確英文查詢語句，兩者疊加比單用其一更穩定。
        """
        parts = [p.strip() for p in (self.query, self.summary) if p and p.strip()]
        return " ".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.embedding_text and not self.intent_types