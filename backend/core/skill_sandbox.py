# -*- coding: utf-8 -*-
"""core/skill_sandbox.py — Skill 專屬 e2b sandbox 生命週期管理。

設計原則：
  - 本類別不持有任何 per-session 狀態（不快取 sandbox_id）。
    sandbox_id 由呼叫端（OrchestratorState.skill_sandbox_id）逐 session 保存，
    確保不同 session／不同 RC 之間不會共用同一個 sandbox 實例。
  - create()：lazy，只在 MainAgent 第一次真的需要執行 skill 程式碼時才呼叫
    （見 tools/skill_code.py 的 make_run_skill_code_tool）。
  - keep_alive()：每次 MainAgent.chat() 呼叫開頭執行一次，延長既有 sandbox 的
    timeout。失敗（多半代表 sandbox 已過期或被回收）回傳 False，
    呼叫端應視同 sandbox 不存在，交由下一次 run_skill_code 呼叫時 lazy 重建。

模板（template）與圖表回傳機制暫緩優化，先以 e2b 預設 template 運作，
後續視實際 skill 依賴需求再建立自訂 template（見討論記錄）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from e2b_code_interpreter import Sandbox

_DEFAULT_TIMEOUT_S = 600  # 10 分鐘


@dataclass
class SandboxRunResult:
    """run_code() 的標準化結果，供 tools/skill_code.py 組裝成給 LLM 看的文字。"""
    stdout:  str
    stderr:  str
    error:   Optional[str]
    results: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts: List[str] = []
        if self.stdout:
            parts.append(f"[stdout]\n{self.stdout}")
        if self.results:
            parts.append("[results]\n" + "\n".join(self.results))
        if self.error:
            parts.append(f"[error]\n{self.error}")
        elif self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n\n".join(parts) if parts else "（無輸出）"


class SkillSandboxProvisioner:
    """管理 skill 專屬 e2b sandbox 的建立、續命與程式碼執行。

    Parameters
    ----------
    api_key   : e2b API key
    template  : e2b sandbox 模板名稱（預設用 e2b 官方基礎模板，
                之後可換成自訂、預裝 numpy/pandas/matplotlib 的模板）
    timeout_s : sandbox 閒置逾時秒數，keep_alive() 每次呼叫都會重設回此值
    """

    def __init__(
        self,
        api_key:   str,
        template:  str = "base",
        timeout_s: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key   = api_key
        self._template  = template
        self._timeout_s = timeout_s

    # ── 建立（lazy，由 run_skill_code 第一次呼叫時觸發）─────────

    def create(self, skill_dir: Path, remote_root: str) -> str:
        """建立新 sandbox，上傳 skill 目錄，回傳 sandbox_id。

        Parameters
        ----------
        skill_dir   : 本地已解壓的 skill 根目錄（RCSkill.skill_dir）
        remote_root : 上傳到 sandbox 內的目標路徑，
                      例如 "/home/user/skills/dsge_model"
        """
        sbx = Sandbox(
            template=self._template,
            api_key=self._api_key,
            timeout=self._timeout_s,
        )
        self._upload_dir(sbx, skill_dir, remote_root)

        requirements = skill_dir / "requirements.txt"
        if requirements.exists():
            result = sbx.commands.run(f"pip install -r {remote_root}/requirements.txt")
            if result.exit_code != 0:
                print(f"[SkillSandboxProvisioner] pip install failed "
                      f"(sandbox={sbx.sandbox_id}): {result.stderr}")

        print(f"[SkillSandboxProvisioner] created sandbox={sbx.sandbox_id} "
              f"root={remote_root}")
        return sbx.sandbox_id

    # ── 續命（每次 MainAgent.chat() 呼叫開頭執行）───────────────

    def keep_alive(self, sandbox_id: str) -> bool:
        """延長既有 sandbox 的 timeout。失敗代表 sandbox 已不可用。"""
        try:
            sbx = Sandbox.connect(sandbox_id, api_key=self._api_key)
            sbx.set_timeout(self._timeout_s)
            return True
        except Exception as exc:
            print(f"[SkillSandboxProvisioner] keep_alive failed "
                  f"sandbox={sandbox_id}: {exc}")
            return False

    # ── 執行 ─────────────────────────────────────────────────

    def run_code(self, sandbox_id: str, code: str) -> SandboxRunResult:
        """在既有 sandbox 內執行程式碼，回傳標準化結果。"""
        sbx       = Sandbox.connect(sandbox_id, api_key=self._api_key)
        execution = sbx.run_code(code)
        return SandboxRunResult(
            stdout  = execution.logs.stdout or "",
            stderr  = execution.logs.stderr or "",
            error   = str(execution.error) if execution.error else None,
            results = [str(r) for r in execution.results],
        )

    # ── Private ──────────────────────────────────────────────

    @staticmethod
    def _upload_dir(sbx: "Sandbox", local_dir: Path, remote_root: str) -> None:
        for p in local_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(local_dir)
                sbx.files.write(f"{remote_root}/{rel}", p.read_bytes())