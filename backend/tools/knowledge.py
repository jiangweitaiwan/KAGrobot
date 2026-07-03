"""
tools/knowledge.py
─────────────────────────────────────────────────────────────
知識圖譜查詢與推理邏輯追蹤工具。

包含：
  - knowledge_qa               — 查詢 COMPOSES chain 知識背景（工廠函數）
  - trace_rc_logic              — 追蹤 Resulting Concept 的推導邏輯鏈（工廠函數）
  - read_skill_reference        — 按需讀取 RC Skill 的 references 文件（工廠函數，新增）

這些工具以「工廠函數」形式提供，由 core/main_agent.py 在每次
build_tools_from_state 時動態建立，因此不透過 _TOOL_IMPLEMENTATIONS
全域字典管理，而是直接在 build_tools_from_state 內呼叫這裡的工廠。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase

from settings import get_settings


# ═══════════════════════════════════════════════════════════════
# knowledge_qa（工廠函數）
# ═══════════════════════════════════════════════════════════════

def make_knowledge_qa_tool(reasoning_chains: List[dict]) -> StructuredTool:
    """建立一個 closure 工具，讓 Agent 可查詢 COMPOSES chain 知識。

    Args:
        reasoning_chains: 來自 SocratesChatBot resolved state 的推理鏈列表。

    Returns:
        LangChain StructuredTool。
    """
    knowledge_text = _format_reasoning_chains(reasoning_chains)

    def knowledge_qa(question: str) -> str:
        """查詢知識圖譜推理鏈中的理論背景與概念關聯。
        當你需要理論依據或概念解釋時使用此工具。"""
        return json.dumps(
            {"question": question, "knowledge_context": knowledge_text},
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        func=knowledge_qa,
        name="knowledge_qa",
        description=(
            "查詢從知識圖譜中檢索到的推理鏈與理論背景。"
            "當需要理論依據、概念定義或論文脈絡時呼叫此工具。"
        ),
    )


def _format_reasoning_chains(chains: List[dict]) -> str:
    """將 COMPOSES chain 列表序列化為易讀文字。"""
    if not chains:
        return "（無可用推理鏈）"

    parts: List[str] = []
    for i, chain in enumerate(chains, 1):
        start = chain.get("start_concept", {}).get("name", "?")
        final = chain.get("final_resulting_concept", {}).get("name", "?")

        step_names = [start]
        for step in chain.get("steps", []):
            to_name = step.get("to_concept", {}).get("name", "?")
            step_names.append(to_name)
            paper = step.get("via_paper")
            if paper and paper.get("title"):
                step_names[-1] += f" (via: {paper['title']})"

        path_str = " → ".join(step_names)

        final_paper = chain.get("final_paper")
        paper_info  = ""
        if final_paper:
            paper_info = f"\n   關聯論文: {final_paper.get('title', 'N/A')}"
            if premises := final_paper.get("conceptual_premises"):
                paper_info += f"\n   前提假設: {premises}"
            if constraints := final_paper.get("contextual_model_constraints"):
                paper_info += f"\n   模型約束: {constraints}"

        parts.append(f"推理鏈 {i}: {path_str}{paper_info}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# trace_rc_logic（工廠函數）
# ═══════════════════════════════════════════════════════════════

def make_trace_rc_logic_tool() -> StructuredTool:
    """建立 trace_rc_logic 工具。

    每次呼叫都從 settings 讀取 Neo4j 連線資訊，不持有連線狀態。

    Returns:
        LangChain StructuredTool。
    """

    def _trace_rc_logic(rc_name: str, mode: str = "logic_step") -> str:
        """根據 Resulting Concept 名稱，回傳從所有 Input Concepts
        經由 LogicUnit 推論到該 RC 的內部邏輯步驟。"""
        mode = (mode or "logic_step").lower().strip()
        if mode not in ("logic_step", "theory"):
            mode = "logic_step"

        s = get_settings()
        uri      = s.neo4j_uri
        password = s.neo4j_password

        if not uri or not password:
            return json.dumps(
                {"ok": False, "error": "缺少 Neo4j 連線設定。"},
                ensure_ascii=False,
            )

        cypher = """
        MATCH (p:Paper)-[:PRODUCES]->(rc:Concept {type:'Resulting_Concept', name:$rc_name})
        WITH p, rc

        MATCH (c:Concept)-[r:INSPIRES]->(p)
        WHERE coalesce(r.type, 'Input') = 'Input'
        WITH p, rc, collect(c) AS all_inputs

        WITH p, rc, all_inputs, all_inputs[0] AS representative_c
        OPTIONAL MATCH path = (representative_c)-[:NEXT_STEP*1..10]->(l:LogicUnit)
        WHERE l.uid STARTS WITH p.title + '_'

        WITH p, rc, all_inputs, path
        WHERE path IS NOT NULL
        ORDER BY length(path) DESC
        LIMIT 1

        RETURN
          p.title AS paper_title,
          rc.name AS rc_name,
          [i IN all_inputs | i.name] AS input_names,
          [n IN nodes(path) WHERE n:LogicUnit |
            {logic_step: n.logic_step, theory: n.theory}
          ] AS logic_units
        """

        driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        try:
            with driver.session() as session:
                result = session.run(cypher, rc_name=rc_name).single()
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        finally:
            driver.close()

        if not result:
            return json.dumps({"ok": False, "error": "No path found."}, ensure_ascii=False)

        paper_title = result.get("paper_title")
        inputs      = result.get("input_names") or []
        logic_units = result.get("logic_units") or []

        raw_steps = [lu.get(mode) for lu in logic_units if lu.get(mode)]
        num_steps = len(raw_steps)

        structured_steps: List[dict] = []
        if num_steps == 1:
            structured_steps.append({"step": "final", "content": raw_steps[0]})
            path_desc = "inputs -> final -> rc"
        elif num_steps >= 2:
            for i in range(num_steps - 1):
                structured_steps.append({"step": f"T{i+1}", "content": raw_steps[i]})
            structured_steps.append({"step": "final", "content": raw_steps[-1]})
            t_labels  = " -> ".join([f"T{i+1}" for i in range(num_steps - 1)])
            path_desc = f"inputs -> {t_labels} -> final -> rc"
        else:
            path_desc = "inputs -> rc"

        return json.dumps({
            "ok":   True,
            "context": {"paper": paper_title, "mode": mode},
            "logic_path": {
                "origin":             inputs,
                "intermediate_steps": structured_steps,
                "destination":        rc_name,
                "path_summary":       path_desc,
            },
        }, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_trace_rc_logic,
        name="trace_rc_logic",
        description=(
            "此工具用於追溯 concept 的邏輯推導鏈條。"
            "回傳結果包含 intermediate_steps，"
            "其中標籤為 T1, T2... 的是中間邏輯單元，"
            "標籤為 final 的是產出 concept 前的最後一個關鍵理論/步驟。"
        ),
    )


# ═══════════════════════════════════════════════════════════════
# read_skill_reference（工廠函數，新增）
# ═══════════════════════════════════════════════════════════════

def make_read_skill_reference_tool(
    skill_dir:     Path,
    allowed_files: List[str],
) -> StructuredTool:
    """建立按需讀取 RC Skill references 文件的 closure 工具。

    採 lazy loading 設計：SKILL.md 的 prompt_context 只包含 references
    的檔名索引，不含全文。當 LLM 判斷需要更詳細的理論依據、公式推導或
    參數說明時，才呼叫此工具讀取指定檔案的完整內容。

    Args:
        skill_dir:     此 skill 在本地（/tmp）解壓後的根目錄。
        allowed_files: 該 skill 底下 references/ 目錄中實際存在的檔名清單，
                        用於防止 LLM 傳入任意路徑（也順帶避免 path traversal）。

    Returns:
        LangChain StructuredTool。
    """
    allowed_set = set(allowed_files)

    def read_skill_reference(file_name: str) -> str:
        """讀取當前 RC Skill 底下 references/ 目錄中指定檔案的完整內容。
        當 SKILL 說明中提及某份參考文件、而你判斷需要更深入的理論依據、
        公式推導或參數校準細節時才呼叫此工具，傳入檔名（不含路徑）。"""
        if file_name not in allowed_set:
            return json.dumps({
                "ok":    False,
                "error": f"'{file_name}' 不在此 Skill 的 references 清單中。",
                "available_files": sorted(allowed_set),
            }, ensure_ascii=False)

        target_path = skill_dir / "references" / file_name

        # 防禦性檢查：確保解析後的路徑仍在 references/ 目錄內
        try:
            target_path = target_path.resolve()
            refs_root   = (skill_dir / "references").resolve()
            if refs_root not in target_path.parents and target_path != refs_root:
                return json.dumps(
                    {"ok": False, "error": "非法路徑。"}, ensure_ascii=False
                )
        except Exception:
            return json.dumps({"ok": False, "error": "路徑解析失敗。"}, ensure_ascii=False)

        if not target_path.exists():
            return json.dumps(
                {"ok": False, "error": f"檔案不存在：{file_name}"}, ensure_ascii=False
            )

        try:
            content = target_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"讀取失敗：{exc}"}, ensure_ascii=False
            )

        return json.dumps({
            "ok":        True,
            "file_name": file_name,
            "content":   content,
        }, ensure_ascii=False)

    return StructuredTool.from_function(
        func=read_skill_reference,
        name="read_skill_reference",
        description=(
            "讀取此 RC Skill 底下 references/ 目錄中的補充文件完整內容。"
            "當 SKILL 說明中提及某份參考文件（例如「詳見 references/xxx.md」），"
            "而你判斷當前需求需要更詳細的理論依據、公式推導或參數說明時，"
            "才呼叫此工具，傳入確切檔名（不含路徑，例如 'equations.md'）。"
            "若當前問題不需要更多細節，則不必主動呼叫。"
        ),
    )