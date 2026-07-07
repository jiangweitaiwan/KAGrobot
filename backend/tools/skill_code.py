# -*- coding: utf-8 -*-
"""tools/skill_code.py — Skill 專屬泛用程式碼執行工具（工廠函數）。

取代原本的具名 tool 生成邏輯（identify_shocks / forecast_economy 等）：
不再把 skill 內函式逐一包成獨立 StructuredTool，改成單一泛用工具，
讓 LLM 自己撰寫 Python import 並呼叫 skill 內的函式。

sandbox_state 是 mutable dict {"id": Optional[str]}，由 MainAgent.chat() 建立，
呼叫結束後讀出，藉此把「首次執行時才建立」產生的新 sandbox_id
回傳給 orchestrator 保存到 OrchestratorState.skill_sandbox_id。

sys.path 設定（重要修正）：
  skill_dir 整包上傳到 sandbox 後會保留原始目錄結構，例如
  {remote_root}/scripts/xxx.py、{remote_root}/references/yyy.md。
  多數 skill（含 skill-creator 產出的慣例）在 SKILL.md 範例中會寫
  `from xxx import yyy` 這種 flat import，代表撰寫時假設 scripts/
  本身就在 sys.path 上，而不是 `from scripts.xxx import yyy`。
  因此除了 remote_root 本身，還必須把 SkillLoader 掃描到的
  code_dirs（例如 "scripts"）逐一加入 sys.path，否則會直接得到
  ModuleNotFoundError。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import StructuredTool

from core.skill_sandbox import SkillSandboxProvisioner


def make_run_skill_code_tool(
    provisioner:   SkillSandboxProvisioner,
    sandbox_state: Dict[str, Optional[str]],
    skill_dir:     Path,
    remote_root:   str,
    code_dirs:     Optional[List[str]] = None,
) -> StructuredTool:
    """建立此 Skill 專屬的 run_skill_code 工具。

    Parameters
    ----------
    code_dirs : SkillLoader 掃描到的相對目錄清單（例如 ["scripts"]），
                除了 remote_root 本身之外，會逐一加入 sandbox 內的
                sys.path，讓 SKILL.md 慣用的 flat import 可以正常運作。
                "." 代表根目錄本身，已由 remote_root 涵蓋，這裡會略過。
    """
    code_dirs = code_dirs or []
    # "." 代表 skill 根目錄本身，remote_root 已經涵蓋，不需要重複加入
    extra_path_dirs = [d for d in code_dirs if d and d != "."]
    path_hint = (
        "、".join(f"`{d}/`" for d in extra_path_dirs)
        if extra_path_dirs else "skill 根目錄"
    )

    def run_skill_code(code: str) -> str:
        """在此 Skill 專屬的沙盒環境中執行 Python 程式碼。
        sandbox 內已放好此 Skill 目錄下的所有檔案（含 references/），
        其中的函式庫程式碼目錄已加入 sys.path，可直接 import 使用，
        references 檔案可用相對路徑讀取，例如：
        open('references/equations.md').read()。
        若需要繪圖，將檔案存到 /home/user/outputs/ 目錄。"""
        if sandbox_state["id"] is None:
            try:
                sandbox_state["id"] = provisioner.create(skill_dir, remote_root)
            except Exception as exc:
                return f"[error]\n沙盒環境建立失敗，無法執行程式碼：{exc}"

        path_setup = "".join(
            f"sys.path.insert(0, '{remote_root}/{d}')\n" for d in extra_path_dirs
        )
        full_code = (
            f"import sys, os\n"
            f"sys.path.insert(0, '{remote_root}')\n"
            f"{path_setup}"
            f"os.chdir('{remote_root}')\n"
            f"{code}"
        )

        try:
            result = provisioner.run_code(sandbox_state["id"], full_code)
        except Exception as exc:
            # 連線失敗多半代表 sandbox 已過期，重置狀態讓下次呼叫重建
            sandbox_state["id"] = None
            return f"[error]\n沙盒執行失敗：{exc}"

        return result.to_text()

    return StructuredTool.from_function(
        func=run_skill_code,
        name="run_skill_code",
        description=(
            f"在此 Skill（{skill_dir.name}）專屬的沙盒環境中執行 Python 程式碼。"
            f"sandbox 內已包含此 Skill 的所有原始檔案，其中 {path_hint} 已加入 "
            "Python path，可直接 import 對應模組進行計算、模擬或繪圖，"
            "無需重新實作邏輯，也不需自行處理路徑設定。"
        ),
    )