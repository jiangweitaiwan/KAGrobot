# -*- coding: utf-8 -*-
"""retrieval/concept_retriever.py — Concept 詳情查詢。

直接查詢 Neo4j，回傳指定 Resulting Concept 的描述、
所屬論文的理論前提與模型約束。
供 SocratesChatBot 的 retrieve_concept_detail tool 使用。
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
        """查詢指定 concept 的描述與所屬論文的理論背景。

        Parameters
        ----------
        rc_name : Resulting Concept 名稱（允許部分匹配）

        Returns
        -------
        dict，含 found / rc_name / description / function /
              analysis_types / papers 欄位。
        """
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($name)
                OPTIONAL MATCH (p:Paper)-[:PRODUCES]->(c)
                RETURN
                    c.name             AS name,
                    c.description      AS description,
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
            "found":          True,
            "rc_name":        row["name"],
            "description":    row["description"],
            "function":       row["function"],
            "analysis_types": row["analysis_types"] or [],
            "papers":         papers,
        }
