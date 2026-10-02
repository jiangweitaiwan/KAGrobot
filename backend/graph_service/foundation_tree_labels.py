# -*- coding: utf-8 -*-
"""foundation_tree 節點 label 補齊（layout_cache 為統一 fallback）。

Upstream BFS 的 label 來自 Neo4j Concept.name；缺 name 時會退回 eid。
Global layout 與 anchor 查詢則另有 layout_cache / Neo4j 的顯示名稱。
此模組在 expand 結果寫出前，把仍缺 label 的節點對齊 layout。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def anchor_label_from_layout(layout_cache: Optional[Dict[str, Any]], eid: str) -> Optional[str]:
    if not layout_cache:
        return None
    for n in layout_cache.get("nodes", []):
        if n.get("id") == eid:
            name = (n.get("name") or "").strip()
            return name or None
    return None


def anchor_label_from_neo4j(driver: Any, eid: str) -> str:
    with driver.session() as session:
        rec = session.run(
            "MATCH (c:Concept) WHERE elementId(c) = $eid RETURN c.name AS name",
            eid=eid,
        ).single()
    return (rec["name"] if rec and rec["name"] else None) or eid


def resolve_anchor_label(
    eid: str,
    layout_cache: Optional[Dict[str, Any]],
    driver: Any,
) -> str:
    from_layout = anchor_label_from_layout(layout_cache, eid)
    if from_layout:
        return from_layout
    return anchor_label_from_neo4j(driver, eid)


class ExpandLabelEnricher:
    """HTTP / 背景服務共用的 expand label 補齊（layout + anchor Neo4j）。"""

    def __init__(
        self,
        driver: Any,
        layout_cache_getter: Callable[[], Optional[Dict[str, Any]]],
    ) -> None:
        self._driver = driver
        self._get_layout = layout_cache_getter

    def anchor_label(self, eid: str) -> str:
        return resolve_anchor_label(eid, self._get_layout(), self._driver)

    def enrich_result(self, anchor_eid: str, result: Dict[str, Any]) -> None:
        enrich_upstream_expand_result(
            result,
            anchor_eid=anchor_eid,
            anchor_label=self.anchor_label(anchor_eid),
            layout_cache=self._get_layout(),
        )


def build_layout_label_index(layout_cache: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not layout_cache:
        return {}
    out: Dict[str, str] = {}
    for n in layout_cache.get("nodes", []):
        eid = n.get("id")
        name = (n.get("name") or "").strip()
        if eid and name:
            out[eid] = name
    return out


def label_needs_fallback(label: Optional[str], eid: str) -> bool:
    s = (label or "").strip()
    return not s or s == eid


def resolve_foundation_label(
    eid: str,
    label: Optional[str],
    *,
    anchor_eid: str,
    anchor_label: str,
    layout_index: Dict[str, str],
) -> str:
    if eid == anchor_eid:
        al = (anchor_label or "").strip()
        if al and al != eid:
            return al
    cur = (label or "").strip()
    if not label_needs_fallback(cur, eid):
        return cur
    from_layout = layout_index.get(eid, "").strip()
    if from_layout and from_layout != eid:
        return from_layout
    return cur or eid


def enrich_upstream_expand_result(
    result: Dict[str, Any],
    *,
    anchor_eid: str,
    anchor_label: str,
    layout_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """就地更新 final_foundations 與 foundation_tree.nodes 的 label。"""
    layout_index = build_layout_label_index(layout_cache)
    kwargs = dict(anchor_eid=anchor_eid, anchor_label=anchor_label, layout_index=layout_index)

    for f in result.get("final_foundations", []):
        f["label"] = resolve_foundation_label(
            f["element_id"], f.get("label"), **kwargs,
        )

    tree = result.get("foundation_tree")
    if tree:
        for n in tree.get("nodes", []):
            n["label"] = resolve_foundation_label(
                n["element_id"], n.get("label"), **kwargs,
            )
    return result
