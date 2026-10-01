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
    query:            str       = ""
    intent_types:     List[str] = field(default_factory=list)
    summary:          str       = ""
    thinking_process: str       = ""   # 新增：walk 完成後累積的探索摘要

    @property
    def embedding_text(self) -> str:
        """組合 query + summary + thinking_process。

        thinking_process 是 random walk（慢想）完成後留下的探索軌跡摘要，
        與 socratic 的 summary（使用者需求本身）語意角色不同：summary
        回答「使用者要什麼」，thinking_process 回答「探索過程中發現了
        什麼、往哪個方向收斂」，兩者疊加比單用其一更貼近真正該推薦的 RC。
        """
        parts = [
            p.strip() for p in (self.query, self.summary, self.thinking_process)
            if p and p.strip()
        ]
        return " ".join(parts)

    @property
    def requirement_text(self) -> str:
        """只組 query + summary：使用者的「需求」本身，不含慢想探索軌跡。

        RC briefing 需要把「使用者要什麼」與「使用者探索過什麼」分開呈現，
        fit_gaps 只能對照需求，不能把探索當成需求，所以不能用 embedding_text。
        """
        parts = [p.strip() for p in (self.query, self.summary) if p and p.strip()]
        return " ".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.embedding_text and not self.intent_types

