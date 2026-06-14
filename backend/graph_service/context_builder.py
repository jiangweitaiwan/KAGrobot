# -*- coding: utf-8 -*-
"""graph_service/context_builder.py — 知識圖譜 context 文字組裝。

將 all_graphs 轉換為 SocratesChatBot system prompt 可嵌入的純文字摘要，
說明各候選的下游終點與基礎概念。
"""

from __future__ import annotations

from typing import Any, Dict, List

from graph_service.projector import GraphProjector


class GraphContextBuilder:
    """將 all_graphs 組裝為繁體中文 graph_context 文字。"""

    def __init__(self) -> None:
        self._projector = GraphProjector()

    def build(self, all_graphs: Dict[int, Any]) -> str:
        """組裝並回傳 graph_context 字串。"""
        lines: List[str] = []

        for idx, g in sorted(all_graphs.items()):
            if not g:
                lines.append(f"【候選 {idx}】（圖譜展開失敗）")
                continue

            concept_name = (
                g.get("query_concepts") or [{}]
            )[0].get("name", f"選項 {idx}")
            lines.append(f"【候選 {idx}】{concept_name}")

            proj              = self._projector.project(g)
            downstream_finals = self._extract_downstream_finals(proj)

            if downstream_finals:
                lines.append(f"  下游終點：{', '.join(downstream_finals)}")
            else:
                lines.append("  下游終點：（無）")

            foundation_labels = [
                f["label"]
                for f in (g.get("foundations") or [])
                if f.get("label")
            ]
            if foundation_labels:
                lines.append(f"  基礎概念：{', '.join(foundation_labels)}")
            else:
                lines.append("  基礎概念：（無）")

            lines.append("")

        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_downstream_finals(proj: Dict[str, Any]) -> List[str]:
        seen:   set        = set()
        result: List[str]  = []
        for path in proj.get("paths", []):
            if path.get("direction", "downstream") != "downstream":
                continue
            label = path.get("final_label", "")
            if label and label not in seen:
                seen.add(label)
                result.append(label)
        return result
