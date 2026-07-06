# -*- coding: utf-8 -*-
"""tools/skill_code.py — Skill 專屬泛用程式碼執行工具（工廠函數）。

取代原本的具名 tool 生成邏輯（identify_shocks / forecast_economy 等）：
不再把 skill 內函式逐一包成獨立 StructuredTool，改成單一泛用工具，
讓 LLM 自己撰寫 Python import 並呼叫 skill 內的函式。

sandbox_state 是 mutable dict {"id": Optional[str]}，由 MainAgent.chat() 建立，
呼叫結束後讀出，藉此把「首次執行時才建立」產生的新 sandbox_id
回傳給 orchestrator 保存到 OrchestratorState.skill_sandbox_id。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from langchain_core.tools import StructuredTool

from core.skill_sandbox import SkillSandboxProvisioner


def make_run_skill_code_tool(
    provisioner:   SkillSandboxProvisioner,
    sandbox_state: Dict[str, Optional[str]],
    skill_dir:     Path,
    remote_root:   str,
) -> StructuredTool:

    def run_skill_code(code: str) -> str:
        """在此 Skill 專屬的沙盒環境中執行 Python 程式碼。
        sandbox 內已放好此 Skill 目錄下的所有檔案，其中的函式可直接 import 使用，
        不需要重新實作邏輯。若需要繪圖，將檔案存到 /home/user/outputs/ 目錄。"""
        if sandbox_state["id"] is None:
            sandbox_state["id"] = provisioner.create(skill_dir, remote_root)

        full_code = f"import sys; sys.path.insert(0, '{remote_root}')\n{code}"
        result    = provisioner.run_code(sandbox_state["id"], full_code)
        return result.to_text()

    return StructuredTool.from_function(
        func=run_skill_code,
        name="run_skill_code",
        description=(
            f"在此 Skill（{skill_dir.name}）專屬的沙盒環境中執行 Python 程式碼。"
            "sandbox 內已包含此 Skill 的所有原始檔案，可直接 import 其中的函式"
            "進行計算、模擬或繪圖，無需重新實作邏輯。"
        ),
    )