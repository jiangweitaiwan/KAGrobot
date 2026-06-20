# -*- coding: utf-8 -*-
"""retrieval/concept_retriever.py — Concept 詳情查詢。

直接查詢 Neo4j，回傳指定 Concept 的定義、機制、關鍵假設、（若為 RC）功能，
以及所屬論文的理論前提與模型約束。
供 SocratesChatBot 的 retrieve_concept_detail tool 使用。

Neo4j Concept 節點欄位：
    definition       : 概念定義（一般 concept / RC 皆有）
    key_assumptions  : 關鍵假設（一般 concept / RC 皆有）
    mechanism        : 作用機制（一般 concept / RC 皆有）
    name             : 概念名稱（一般 concept / RC 皆有）
    function         : 功能說明（僅 Resulting Concept 才有值；
                        一般 concept 此欄位為 null）

注意：mechanism 與 function 是兩個獨立欄位並存，不是替代關係。
若 function 為 null，代表該 concept 是一般 concept（非 RC）。

舊版 analysis_types 欄位已移除，不再查詢。
Paper 節點上的 conceptual_premises / contextual_model_constraints
維持不變，與 concept 欄位並存回傳。
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConceptDetailRetriever:
    """查詢 Concept 詳情的 Neo4j 查詢器。

    Parameters
    ----------
    driver : Neo4j driver
    """

    def __init__(self, driver) -> None:
        self._driver = driver

    def retrieve(self, rc_name: str) -> Dict[str, Any]:
        """查詢指定 concept 的定義、機制、關鍵假設、功能（若為 RC）與所屬論文的理論背景。

        Parameters
        ----------
        rc_name : Concept 名稱（允許部分匹配）

        Returns
        -------
        dict，含 found / rc_name / definition / mechanism /
              key_assumptions / function / papers 欄位。
              function 為 null 時代表該 concept 是一般 concept（非 RC）。
        """
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($name)
                OPTIONAL MATCH (p:Paper)-[:PRODUCES]->(c)
                RETURN
                    c.name             AS name,
                    c.definition       AS definition,
                    c.mechanism        AS mechanism,
                    c.key_assumptions  AS key_assumptions,
                    c.function         AS function,
                    c.analysis_types   AS analysis_types,
                    collect({
                        title:       p.title,
                        premises:    p.conceptual_premises,
                        constraints: p.contextual_model_constraints
                    }) AS papers
                LIMIT 1
                """,
                name=rc_name,
            ).data()

        if not rows:
            return {
                "found":   False,
                "rc_name": rc_name,
                "message": f"找不到與「{rc_name}」相符的 concept。",
                "papers":  [],
            }

        row    = rows[0]
        papers = [
            {
                "title":       p["title"],
                "premises":    p["premises"],
                "constraints": p["constraints"],
            }
            for p in (row["papers"] or [])
            if p.get("title")
        ]

        return {
            "found":           True,
            "rc_name":         row["name"],
            "definition":      row["definition"],
            "mechanism":       row["mechanism"],
            "key_assumptions": row["key_assumptions"],
            "function":        row["function"],
            "analysis_types": row["analysis_types"] or [],
            "papers":          papers,
        }