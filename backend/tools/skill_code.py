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
  物件路徑改為 "{uid}/{sandbox_id}/{filename}"（原本只用 sandbox_id）。
  uid 為已驗證的使用者 id（見 core/main_agent.py 的 chat(..., uid=...)），
  sandbox_id 仍是「此 session + 此 skill」的唯一鍵，兩者疊加後，
  即使未來要依使用者盤點、清理或審計輸出檔案，也有明確前綴可用；
  uid 缺失時（理論上不會發生，因為 /chat 已強制登入）退回 "anon"，
  不讓工具因此直接失敗。

Attachment 結構化回傳（新增）：
  get_sandbox_file 不再要求 LLM 自己把簽名 URL 貼進回覆文字——那樣做
  既不穩定（模型可能忘記貼、可能貼錯格式），前端也很難穩定解析。
  改成：工具把每個成功產出的檔案登記進呼叫端傳入的 attachment_state
  （與 sandbox_state 同樣的「mutable dict closure」pattern），並回傳
  一個短編號 ref，要求 LLM 只需要在回覆文字中插入 [[attachment:{ref}]]
  佔位符來標記「這個檔案跟哪一段敘述有關」，不需要處理實際網址。
  chat() 結束後從 attachment_state 讀出完整清單，交給呼叫端（
  OrchestratorNodes → ChatResponse）組成結構化的 attachments 欄位，
  前端再依 ref 把佔位符換成實際的縮圖/檔案卡片。
"""

from __future__ import annotations

import json
import os
import time
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
# get_sandbox_file（取回產出檔案，上傳 GCS，登記為結構化 attachment）
# ══════════════════════════════════════════════════════════════

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def make_get_sandbox_file_tool(
    computer:         SandboxComputer,
    skill_name:       str,
    bucket_name:      str,
    attachment_state: dict,
    uid:              Optional[str] = None,
) -> StructuredTool:
    """建立 get_sandbox_file 工具。

    Parameters
    ----------
    computer         : SandboxComputer
    skill_name        : 供 description 顯示用
    bucket_name       : 由呼叫端傳入（main_agent._build_tools() 從
                         settings.skill_outputs_bucket_name 讀取），
                         此處不直接依賴 settings 模組，維持工具工廠的
                         可測試性（可直接傳假 bucket 名測試）。
    attachment_state  : {"items": []}，與 sandbox_state 同樣的
                         mutable dict closure pattern。每次成功呼叫
                         會 append 一筆 {ref, kind, url, filename,
                         created_at}，chat() 結束後由 MainAgent 讀出
                         回傳給呼叫端，最終組成 ChatResponse.attachments。
    uid               : 已驗證的使用者 id（見 core/main_agent.py），
                         用於在 GCS 物件路徑加上使用者命名空間
                         "{uid}/{sandbox_id}/{filename}"。為 None 時
                         退回 "anon"，不影響工具正常運作。
    """
    uid_segment = uid or "anon"

    def get_sandbox_file(rel_path: str) -> str:
        """把此 Skill 沙盒內的檔案（例如 run_skill_code 產出的圖表或報告）
        取出，上傳至雲端儲存並登記為一筆 attachment。
        rel_path 為相對於此 Skill 根目錄的路徑，
        例如 run_skill_code 存到 outputs/chart.png 後，這裡傳 'outputs/chart.png'。
        呼叫成功後會拿到一個 ref 編號——請在你的回覆文字中，於你認為適合
        的位置插入 [[attachment:{ref}]] 這個佔位符來標記此檔案，
        **不要**自己把網址貼進回覆內容（前端會依 ref 自動換成縮圖／檔案卡片，
        貼原始網址反而會讓顯示重複或格式跑掉）。連結有效期 24 小時。"""
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
        object_key = f"{uid_segment}/{sandbox_id}/{filename}"

        try:
            url = _upload_and_sign(bucket_name, object_key, data)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"上傳雲端儲存失敗：{exc}"}, ensure_ascii=False
            )

        ref  = str(len(attachment_state["items"]) + 1)
        kind = "image" if os.path.splitext(filename)[1].lower() in _IMAGE_SUFFIXES else "file"
        attachment_state["items"].append({
            "ref":              ref,
            "kind":             kind,
            "url":              url,
            "filename":         filename,
            "created_at":       int(time.time() * 1000),
            "expires_in_hours": 24,
        })

        return json.dumps({
            "ok":          True,
            "ref":         ref,
            "kind":        kind,
            "instruction": (
                f"請在回覆中適合的位置插入 [[attachment:{ref}]] 佔位符標記此檔案，"
                "不要直接貼上網址。"
            ),
        }, ensure_ascii=False)

    return StructuredTool.from_function(
        func=get_sandbox_file,
        name="get_sandbox_file",
        description=(
            f"把此 Skill（{skill_name}）沙盒內的檔案取出（例如 run_skill_code "
            "產出的圖表或報告），上傳至雲端儲存並登記為一筆 attachment（有效 24 小時）。"
            "呼叫成功後會拿到一個 ref 編號，請在回覆文字中用 [[attachment:{ref}]] "
            "佔位符標記此檔案的位置，不要自己貼網址。"
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
