# -*- coding: utf-8 -*-
"""tools/skill_code.py — Skill 專屬泛用執行工具（工廠函數，四件套）。

四個工具（run_skill_code / run_bash / write_file / get_sandbox_file）
共用同一個 SandboxComputer 實例（見 core/skill_sandbox.py），確保
lazy create、連線快取、僅在 sandbox 不可用時重置三件事只有一份實作。

分工原則（寫進各工具 description，供 LLM 判斷該用哪個）：
  - run_skill_code : 數值運算、模擬、繪圖，有狀態（跨呼叫保留變數）
  - run_bash        : 系統操作、pip install、檔案瀏覽，無狀態
  - write_file      : 建立/覆寫 sandbox 內檔案，取代不穩定的 heredoc
  - get_sandbox_file: 把 sandbox 內產出的檔案（例如圖表）取出，
                       上傳至 GCS 並回傳簽名 URL

GCS 路徑分割策略：
  用 sandbox_id（而非 session_id）作為路徑前綴 —— sandbox_id 本身就是
  「此 session + 此 skill」的唯一鍵，不需要額外把 session_id 往下傳，
  也天然避免不同使用者的輸出檔案互相碰撞或覆蓋。
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import List, Optional

from langchain_core.tools import StructuredTool

from core.skill_sandbox import SandboxComputer, SandboxPathError


# ══════════════════════════════════════════════════════════════
# 共用 helper：code_dirs 顯示文字
# ══════════════════════════════════════════════════════════════

def _code_dirs_hint(code_dirs: Optional[List[str]]) -> str:
    dirs = [d for d in (code_dirs or []) if d and d != "."]
    return "、".join(f"`{d}/`" for d in dirs) if dirs else "skill 根目錄"


# ══════════════════════════════════════════════════════════════
# run_skill_code（Python，有狀態）
# ══════════════════════════════════════════════════════════════

def make_run_skill_code_tool(
    computer:   SandboxComputer,
    skill_name: str,
    code_dirs:  Optional[List[str]] = None,
) -> StructuredTool:
    path_hint = _code_dirs_hint(code_dirs)

    def run_skill_code(code: str) -> str:
        """在此 Skill 專屬的沙盒環境中執行 Python 程式碼。
        sandbox 內已放好此 Skill 目錄下的所有檔案（含 references/），
        其中的函式庫程式碼目錄已加入 sys.path，可直接 import 使用，
        references 檔案可用相對路徑讀取，例如：
        open('references/equations.md').read()。
        跨多次呼叫共用同一個執行環境（變數會保留）。
        若需要繪圖，將檔案存到 outputs/ 目錄，之後用 get_sandbox_file 取回。"""
        try:
            result = computer.run_python(code)
        except Exception as exc:
            return f"[error]\n沙盒執行失敗：{exc}"
        return result.to_text()

    return StructuredTool.from_function(
        func=run_skill_code,
        name="run_skill_code",
        description=(
            f"在此 Skill（{skill_name}）專屬的沙盒環境中執行 Python 程式碼。"
            f"sandbox 內已包含此 Skill 的所有原始檔案，其中 {path_hint} 已加入 "
            "Python path，可直接 import 對應模組進行計算、模擬或繪圖。"
            "適合數值運算類任務；系統操作（安裝套件、瀏覽檔案）請改用 run_bash。"
        ),
    )


# ══════════════════════════════════════════════════════════════
# run_bash（Shell，無狀態）
# ══════════════════════════════════════════════════════════════

def make_run_bash_tool(computer: SandboxComputer, skill_name: str) -> StructuredTool:
    def run_bash(command: str) -> str:
        """在此 Skill 專屬的沙盒環境中執行 shell 指令。
        工作目錄已固定為此 Skill 的根目錄。適合系統操作
        （pip install、瀏覽檔案結構、grep/查看檔案內容等）。
        注意：每次呼叫是獨立 process，不與 run_skill_code 共享變數狀態，
        數值運算、模擬、繪圖請改用 run_skill_code。"""
        try:
            result = computer.run_bash(command)
        except Exception as exc:
            return f"[error]\n沙盒指令執行失敗：{exc}"
        return result.to_text()

    return StructuredTool.from_function(
        func=run_bash,
        name="run_bash",
        description=(
            f"在此 Skill（{skill_name}）專屬的沙盒環境中執行 shell 指令，"
            "工作目錄已固定為此 Skill 根目錄。用於系統操作、安裝套件、"
            "瀏覽或搜尋檔案內容。不適合數值運算（那個用 run_skill_code），"
            "且不與 run_skill_code 共享變數狀態。"
        ),
    )


# ══════════════════════════════════════════════════════════════
# write_file（取代不穩定的 heredoc）
# ══════════════════════════════════════════════════════════════

def make_write_file_tool(computer: SandboxComputer, skill_name: str) -> StructuredTool:
    def write_file(rel_path: str, content: str) -> str:
        """在此 Skill 專屬的沙盒環境中建立或覆寫檔案。
        rel_path 為相對於此 Skill 根目錄的路徑（例如 'outputs/notes.md'），
        上層目錄不存在會自動建立。優先使用此工具建立檔案，
        不要在 run_bash 裡用 heredoc（cat > file << EOF），
        內容含特殊字元、引號或大量非 ASCII 文字時 heredoc 容易轉義出錯。"""
        try:
            computer.write_file(rel_path, content)
        except SandboxPathError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"寫入失敗：{exc}"}, ensure_ascii=False
            )
        return json.dumps({"ok": True, "path": rel_path}, ensure_ascii=False)

    return StructuredTool.from_function(
        func=write_file,
        name="write_file",
        description=(
            f"在此 Skill（{skill_name}）專屬的沙盒環境中建立或覆寫檔案。"
            "rel_path 相對於 Skill 根目錄（例如 'outputs/notes.md'）。"
            "建立檔案時優先使用此工具，避免在 run_bash 用 heredoc 寫檔"
            "（特殊字元、引號、非 ASCII 文字容易造成 shell 轉義錯誤）。"
        ),
    )


# ══════════════════════════════════════════════════════════════
# get_sandbox_file（取回產出檔案，上傳 GCS）
# ══════════════════════════════════════════════════════════════

def make_get_sandbox_file_tool(
    computer:    SandboxComputer,
    skill_name:  str,
    bucket_name: str,
) -> StructuredTool:
    """建立 get_sandbox_file 工具。

    bucket_name 由呼叫端傳入（main_agent._build_tools() 從
    settings.skill_outputs_bucket_name 讀取），此處不直接依賴
    settings 模組，維持工具工廠的可測試性（可直接傳假 bucket 名測試）。
    """

    def get_sandbox_file(rel_path: str) -> str:
        """把此 Skill 沙盒內的檔案（例如 run_skill_code 產出的圖表）取出，
        上傳至雲端儲存並回傳可下載的連結，供你在回覆中附給使用者。
        rel_path 為相對於此 Skill 根目錄的路徑，
        例如 run_skill_code 存到 outputs/chart.png 後，這裡傳 'outputs/chart.png'。
        連結有效期 24 小時。"""
        try:
            data = computer.get_file(rel_path)
        except SandboxPathError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"讀取檔案失敗：{exc}"}, ensure_ascii=False
            )

        sandbox_id = computer.sandbox_id or "unknown-sandbox"
        filename   = os.path.basename(rel_path) or "output"
        object_key = f"{sandbox_id}/{filename}"

        try:
            url = _upload_and_sign(bucket_name, object_key, data)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"上傳雲端儲存失敗：{exc}"}, ensure_ascii=False
            )

        return json.dumps({
            "ok":         True,
            "url":        url,
            "expires_in": "24h",
        }, ensure_ascii=False)

    return StructuredTool.from_function(
        func=get_sandbox_file,
        name="get_sandbox_file",
        description=(
            f"把此 Skill（{skill_name}）沙盒內的檔案取出（例如 run_skill_code "
            "產出的圖表），上傳至雲端儲存並回傳可下載連結（有效 24 小時），"
            "供你在回覆中附給使用者。rel_path 相對於此 Skill 根目錄。"
        ),
    )


# ── GCS 上傳 helper（lazy import，避免未使用此工具時仍付初始化成本）──

def _upload_and_sign(bucket_name: str, object_key: str, data: bytes) -> str:
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(object_key)
    blob.upload_from_string(data)

    return _generate_signed_url(blob, expiration=timedelta(hours=24))


def _generate_signed_url(blob, expiration: timedelta) -> str:
    """產生 v4 簽名 URL。

    重要 GCP 細節：Cloud Run 上的 Application Default Credentials
    沒有私鑰，無法在本地直接簽名，必須透過 IAM Credentials API 的
    signBlob 遠端簽名（google-cloud-storage 會自動處理，但需要
    Cloud Run 的服務帳號被授予對「自己」的
    roles/iam.serviceAccountTokenCreator，否則會拋出 403）。
    """
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default()
    if not credentials.valid:
        credentials.refresh(Request())

    return blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )