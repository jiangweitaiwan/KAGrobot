# -*- coding: utf-8 -*-
"""core/skill_sandbox.py — Skill 專屬 e2b sandbox 生命週期管理。

設計原則：
  - SkillSandboxProvisioner 不持有任何 per-session 狀態（不快取 sandbox_id）。
    sandbox_id 由呼叫端（OrchestratorState.skill_sandbox_id）逐 session 保存，
    確保不同 session／不同 RC 之間不會共用同一個 sandbox 實例。
  - create()：lazy，只在 MainAgent 第一次真的需要執行 skill 程式碼時才呼叫
    （見 tools/skill_code.py 的 make_run_skill_code_tool）。
  - 續命併入 connect(timeout=...)：單輪 chat() 內第一次真正操作 sandbox 時
    才連線並延長 timeout，不再在 chat() 開頭額外打一輪 keep_alive。
  - SandboxComputer 在單次 chat() 生命週期內快取已連線的 Sandbox 實例，
    避免同一輪多次工具呼叫反覆 reconnect；僅在 sandbox 明確不可用時才
    清空 id、交由下一次呼叫 lazy 重建。

模板（template）與圖表回傳機制暫緩優化，先以 e2b 預設 template 運作，
後續視實際 skill 依賴需求再建立自訂 template（見討論記錄）。
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from e2b_code_interpreter import Sandbox

_DEFAULT_TIMEOUT_S = 600  # 10 分鐘

_IGNORED_DIR_NAMES = {
    "__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".pytest_cache",
}
_IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}

_T = TypeVar("_T")


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


def is_sandbox_unavailable(exc: BaseException) -> bool:
    """判斷例外是否代表 sandbox 已死／不可再連（應清空 id 並 lazy 重建）。

    一般執行錯誤、檔案不存在、request/execution timeout 不應觸發重建。
    """
    try:
        from e2b.exceptions import SandboxNotFoundException, TimeoutException
    except ImportError:  # pragma: no cover — SDK 極舊或未安裝時的後備
        SandboxNotFoundException = ()  # type: ignore[misc, assignment]
        TimeoutException = ()  # type: ignore[misc, assignment]

    if isinstance(exc, SandboxNotFoundException):
        return True

    if isinstance(exc, TimeoutException):
        t = getattr(exc, "type", None)
        t_key = (
            getattr(t, "name", None)
            or getattr(t, "value", None)
            or (str(t) if t is not None else None)
        )
        if t_key and str(t_key).upper() in {"SANDBOX", "UNAVAILABLE"}:
            return True
        msg = str(exc).lower()
        # 舊版 SDK 無 type 欄位時，僅在訊息明確指向 sandbox idle timeout 才重建
        return "sandbox timeout" in msg or (
            t is None and "unavailable" in msg
        )

    msg = str(exc).lower()
    return (
        "sandbox not found" in msg
        or "no longer running" in msg
        or ("sandbox" in msg and "does not exist" in msg)
        or ("sandbox" in msg and "doesn't exist" in msg)
    )


class SkillSandboxProvisioner:
    """管理 skill 專屬 e2b sandbox 的建立、連線與程式碼執行。

    Parameters
    ----------
    api_key   : e2b API key
    template  : e2b sandbox 模板名稱（預設用 e2b 官方基礎模板，
                之後可換成自訂、預裝 numpy/pandas/matplotlib 的模板）
    timeout_s : sandbox 閒置逾時秒數；connect() 會以此值延長 timeout
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

    # ── 建立（lazy，由 SandboxComputer 第一次操作時觸發）─────────

    def create(self, skill_dir: Path, remote_root: str) -> Sandbox:
        """建立新 sandbox，上傳 skill 目錄，回傳已連線的 Sandbox。"""
        sbx = Sandbox.create(
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
        return sbx

    def connect(self, sandbox_id: str) -> Sandbox:
        """連線既有 sandbox，並以 timeout_s 延長其生命週期。"""
        return Sandbox.connect(
            sandbox_id,
            api_key=self._api_key,
            timeout=self._timeout_s,
        )

    # ── 執行（呼叫端傳入已快取的 Sandbox 實例）────────────────

    def run_code(self, sbx: Sandbox, code: str) -> SandboxRunResult:
        """在既有 sandbox 內執行程式碼，回傳標準化結果。"""
        execution = sbx.run_code(code)
        return SandboxRunResult(
            stdout  = execution.logs.stdout or "",
            stderr  = execution.logs.stderr or "",
            error   = str(execution.error) if execution.error else None,
            results = [str(r) for r in execution.results],
        )

    def run_command(
        self,
        sbx:     Sandbox,
        command: str,
        cwd:     Optional[str] = None,
    ) -> SandboxRunResult:
        """在既有 sandbox 內執行 shell 指令，回傳標準化結果。

        注意：commands.run 與 run_code 是兩條不同機制（前者每次獨立
        process，後者是有狀態 Jupyter kernel），彼此不共享 shell 狀態，
        因此每次呼叫都需要顯式傳入 cwd，不能依賴「上次 cd 過」的假設。
        """
        result = sbx.commands.run(command, cwd=cwd, timeout=0)
        return SandboxRunResult(
            stdout  = result.stdout or "",
            stderr  = result.stderr or "",
            error   = None if result.exit_code == 0 else f"exit_code={result.exit_code}",
            results = [],
        )

    def write_file(self, sbx: Sandbox, remote_path: str, content: str) -> None:
        """在既有 sandbox 內建立/覆寫單一檔案（上層目錄不存在會自動建立）。"""
        sbx.files.write(remote_path, content)

    def read_file(self, sbx: Sandbox, remote_path: str) -> bytes:
        """從既有 sandbox 內讀出檔案原始內容（bytes）。"""
        return bytes(sbx.files.read(remote_path, format="bytes"))

    def kill(self, sandbox_id: str) -> None:
        """Best-effort 終止一個即將被捨棄的 sandbox（例如重新選路時）。

        失敗（sandbox 已過期、已被 e2b 回收等）不拋出例外，
        純粹是盡力清理，避免呼叫端因為這個非關鍵動作而中斷主流程。
        """
        try:
            sbx = self.connect(sandbox_id)
            sbx.kill()
            print(f"[SkillSandboxProvisioner] killed sandbox={sandbox_id}")
        except Exception as exc:
            print(f"[SkillSandboxProvisioner] kill failed "
                  f"sandbox={sandbox_id}: {exc}")

    # ── Private ──────────────────────────────────────────────

    @staticmethod
    def _upload_dir(sbx: Sandbox, local_dir: Path, remote_root: str) -> None:
        """遞迴上傳 skill 目錄，保留相對路徑，並過濾快取/版控等雜訊檔案。

        過濾規則與 skill_loader._scan_code_dirs / _scan_references 一致，
        避免把 __pycache__、.pyc、隱藏檔等上傳進 sandbox
        （既浪費頻寬，也可能造成 import 到過期 bytecode 的問題）。
        """
        for p in local_dir.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(local_dir)

            if any(part in _IGNORED_DIR_NAMES for part in rel.parts):
                continue
            if p.name.startswith(".") or p.suffix in _IGNORED_FILE_SUFFIXES:
                continue

            sbx.files.write(f"{remote_root}/{rel}", p.read_bytes())


# ══════════════════════════════════════════════════════════════
# SandboxComputer — 單一 skill session 的沙盒執行門面
# ══════════════════════════════════════════════════════════════


class SandboxPathError(ValueError):
    """write_file 傳入越界路徑（跳出 remote_root）時拋出。"""


class SandboxComputer:
    """單一 skill session 的沙盒執行門面。

    由 MainAgent._build_tools() 每次 chat() 呼叫建立一個實例，包住
    sandbox_state（mutable dict，跨工具共享、chat() 結束後讀出寫回
    OrchestratorState.skill_sandbox_id）。run_skill_code / run_bash /
    write_file / get_sandbox_file 四個工具的工廠函數皆透過此物件操作，
    確保「lazy create、連線快取、僅在 sandbox 不可用時重置」這三件事
    只有一份實作，不會在四個工具的 closure 裡各自複製一次而逐漸走鐘。

    Parameters
    ----------
    provisioner   : SkillSandboxProvisioner（app 級單例，e2b 綁定層）
    sandbox_state : {"id": Optional[str]}
    skill_dir     : 本地已解壓的 skill 根目錄（RCSkill.skill_dir）
    remote_root   : sandbox 內此 skill 的根目錄
    code_dirs     : SkillLoader 掃描到的相對目錄清單（例如 ["scripts"]），
                    run_python 時逐一加入 sys.path；"." 已由 remote_root
                    本身涵蓋，會被過濾掉
    """

    def __init__(
        self,
        provisioner:   SkillSandboxProvisioner,
        sandbox_state: dict,
        skill_dir:     Path,
        remote_root:   str,
        code_dirs:     Optional[List[str]] = None,
    ) -> None:
        self._provisioner   = provisioner
        self._sandbox_state = sandbox_state
        self._skill_dir     = skill_dir
        self._remote_root   = remote_root.rstrip("/")
        self._extra_paths   = [d for d in (code_dirs or []) if d and d != "."]
        # 單輪 chat() 內快取；跨 chat 不保留（Computer 本身隨 chat 銷毀）
        self._sbx: Optional[Sandbox] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_python(self, code: str) -> SandboxRunResult:
        """以 Python 執行程式碼（有狀態，跨呼叫保留變數）。

        自動處理 sys.path（remote_root 本身 + code_dirs）與 cwd，
        呼叫端只需要提供業務邏輯本身的程式碼。
        """
        path_setup = "".join(
            f"sys.path.insert(0, '{self._remote_root}/{d}')\n"
            for d in self._extra_paths
        )
        full_code = (
            f"import sys, os\n"
            f"sys.path.insert(0, '{self._remote_root}')\n"
            f"{path_setup}"
            f"os.chdir('{self._remote_root}')\n"
            f"{code}"
        )
        return self._run(lambda sbx: self._provisioner.run_code(sbx, full_code))

    def run_bash(self, command: str) -> SandboxRunResult:
        """執行 shell 指令（無狀態，每次獨立 process，不與 run_python 共享狀態）。

        用於系統操作、pip install、檔案瀏覽等；數值運算仍應走 run_python。
        """
        return self._run(
            lambda sbx: self._provisioner.run_command(
                sbx, command, cwd=self._remote_root
            )
        )

    def write_file(self, rel_path: str, content: str) -> None:
        target = self._resolve_path(rel_path)
        self._run(lambda sbx: self._provisioner.write_file(sbx, target, content))

    def get_file(self, rel_path: str) -> bytes:
        """讀出 sandbox 內檔案原始內容，rel_path 相對於 remote_root。"""
        target = self._resolve_path(rel_path)
        return self._run(lambda sbx: self._provisioner.read_file(sbx, target))

    @property
    def sandbox_id(self) -> Optional[str]:
        """目前綁定的 sandbox_id（尚未建立時為 None）。"""
        return self._sandbox_state["id"]

    # ------------------------------------------------------------------
    # Private — lazy create / 連線快取 / 條件式重置
    # ------------------------------------------------------------------

    def _ensure(self) -> Sandbox:
        if self._sbx is not None:
            return self._sbx

        sandbox_id = self._sandbox_state["id"]
        if sandbox_id is None:
            self._sbx = self._provisioner.create(self._skill_dir, self._remote_root)
            self._sandbox_state["id"] = self._sbx.sandbox_id
            return self._sbx

        try:
            self._sbx = self._provisioner.connect(sandbox_id)
            return self._sbx
        except Exception as exc:
            if not is_sandbox_unavailable(exc):
                raise
            # 既有 id 已失效：清空後當場重建，讓本輪工具呼叫仍可成功
            print(f"[SandboxComputer] sandbox '{sandbox_id}' unavailable "
                  f"({exc}); recreating.")
            self._invalidate()
            self._sbx = self._provisioner.create(self._skill_dir, self._remote_root)
            self._sandbox_state["id"] = self._sbx.sandbox_id
            return self._sbx

    def _invalidate(self) -> None:
        self._sbx = None
        self._sandbox_state["id"] = None

    def _run(self, fn: Callable[[Sandbox], _T]) -> _T:
        """統一包裝：確保連線 → 執行 → 僅在 sandbox 不可用時重置狀態。

        重置後不在此處重試；由呼叫端（工具層／LLM）決定是否再呼叫一次，
        避免隱性重試掩蓋真正的錯誤原因。_ensure() 內對「連線時發現已死」
        會當場重建一次，那是探活路徑，與此處操作期失敗分開處理。
        """
        sbx = self._ensure()
        try:
            return fn(sbx)
        except Exception as exc:
            if is_sandbox_unavailable(exc):
                print(f"[SandboxComputer] sandbox became unavailable "
                      f"during operation ({exc}); clearing id.")
                self._invalidate()
            raise

    def _resolve_path(self, rel_path: str) -> str:
        """組出絕對路徑並確認落在 remote_root 內，防止越界讀寫。"""
        candidate = posixpath.normpath(posixpath.join(self._remote_root, rel_path))
        if candidate != self._remote_root and not candidate.startswith(self._remote_root + "/"):
            raise SandboxPathError(
                f"路徑越界：'{rel_path}' 解析後為 '{candidate}'，"
                f"須落在 '{self._remote_root}/' 之內"
            )
        return candidate
