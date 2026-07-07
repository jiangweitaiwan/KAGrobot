# -*- coding: utf-8 -*-


"""skill_loader.py — RC Skill 載入器 (GCS 雲端相容版)

References 採 lazy loading：
  - 優先直接掃描 skill_dir/references/ 目錄下實際存在的檔案，
    列出「檔名清單」放進 RCSkill.reference_paths
  - 不在此處讀取檔案內容，避免每次注入完整 prompt_context 造成 token 浪費
  - 實際內容改由 core/main_agent.py 掛載的 read_skill_reference 工具按需讀取
    （見 tools/knowledge.py 的 make_read_skill_reference_tool）
  - frontmatter 的 `references:` 宣告清單降級為向後相容的 fallback，
    僅在目錄掃描為空時才採用（極端邊界情況，例如 references 檔案不在
    references/ 目錄底下）。

Functions（原 tools）改採目錄掃描為主：
  - 是否需要掛載 sandbox（RCSkill.has_functions）改由掃描 skill_dir 底下
    是否存在含 .py 檔案的一級子目錄（慣例：scripts/）決定，
    不再依賴 SKILL.md frontmatter 的 `tools:` 陣列是否有完整宣告。
    理由：許多 skill 作者只在 SKILL.md body 用自然語言描述檔案結構與
    使用範例（見 skill-creator 產出的慣例），並不會刻意在 frontmatter
    重複宣告一份機器可讀的函式清單；若嚴格要求 frontmatter 完整，
    會導致這類 skill 被誤判為「沒有函式」而喪失 sandbox 執行能力。
  - frontmatter 的 `tools:` 陣列若存在，仍會解析成 SkillFunctionSpec，
    用於在 prompt_context 中提供更精細的函式說明（純粹加分項，
    不影響 has_functions 的判斷，也不影響是否掛載 sandbox）。
  - 實際執行仍由 MainAgent 掛載的泛用 run_skill_code 工具，
    在 e2b sandbox 內直接 import 目錄掃描到的模組執行
    （見 tools/skill_code.py 的 make_run_skill_code_tool）。
  - RCSkill 本身因此不含任何「執行期」狀態，@lru_cache 快取安全，
    不會有不同 session 共用同一份執行狀態的問題。
"""

from __future__ import annotations

import re
import sys
import os
import shutil
from dataclasses import dataclass, field
from threading import Lock   # ← 新增，取代 functools.lru_cache
# 移除: from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import storage  # ── 引入 GCS SDK ──

# YAML / TOML frontmatter 解析（僅依賴標準庫可解析的純文字格式）
# 使用簡易 regex 解析 YAML-like frontmatter，避免額外依賴


# ══════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════

@dataclass
class SkillFunctionSpec:
    """SKILL.md frontmatter 中單一函式的描述（選用，純文件用途，不再動態 import）。

    此函式實際存在於 skill_dir 底下的 script 檔案中，
    由 LLM 透過 run_skill_code 工具在 sandbox 內自行 import 呼叫，
    此處只用來組裝 prompt_context 中的說明文字。
    frontmatter 未宣告時，function_specs 為空列表，
    此時改用目錄掃描結果（RCSkill.code_dirs）產生通用提示，
    不影響是否掛載 sandbox 的判斷。
    """
    name:        str   # 顯示名稱（供 prompt 說明用，預設等同 function）
    script:      str   # 相對於 skill_dir 的路徑
    function:    str   # 函數名稱
    description: str = ""


@dataclass
class ReferenceEntry:
    """單一 reference 的設定（僅檔名/路徑，不含內容）。"""
    rel_path: str    # 相對於 skill_dir 的路徑，例如 "references/equations.md"
    name:     str     # 檔名，例如 "equations.md"（供 read_skill_reference 對照用）


@dataclass
class RCSkill:
    """載入完成的 RC Skill，可直接注入 MainAgent。

    純 metadata 設計：不含任何執行期狀態（無 sandbox_id、無已 import 的函式物件）。
    執行相關的狀態（e2b sandbox_id）由呼叫端逐 session 保存於
    OrchestratorState.skill_sandbox_id，不放進此 dataclass 或 SkillLoader 的快取。
    """
    rc_name:         str
    prompt_context:  str                                 # 注入 system prompt 的動態區塊（不含 references 全文）
    function_specs:  List[SkillFunctionSpec] = field(default_factory=list)  # frontmatter 選用宣告，僅供更精細的 prompt 說明
    reference_paths: List[ReferenceEntry]    = field(default_factory=list)  # references 索引（lazy）
    code_dirs:       List[str]               = field(default_factory=list)  # 新增：含 .py 檔案的一級子目錄清單（相對 skill_dir），
                                                                              # "." 代表根目錄本身，供 sandbox 掛載 sys.path 使用
    skill_dir:       Optional[Path]          = None      # 本地 skill 目錄，供 run_skill_code / read_skill_reference 使用

    @property
    def has_functions(self) -> bool:
        """是否需要掛載 run_skill_code sandbox 工具。

        改依目錄掃描結果（code_dirs）判斷，不再依賴 frontmatter 是否
        完整宣告 function_specs，避免 frontmatter 缺漏時誤判為無函式。
        """
        return len(self.code_dirs) > 0

    @property
    def has_references(self) -> bool:
        return len(self.reference_paths) > 0


# ══════════════════════════════════════════════════════════════
# Frontmatter parser（純文字，不依賴 PyYAML）
# ══════════════════════════════════════════════════════════════

def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """解析 Markdown frontmatter（--- ... ---）。

    Returns
    -------
    (frontmatter_dict, body_text)

    frontmatter_dict 支援：
      - 純字串欄位：  key: value
      - 多行字串：    key: >\n  line1\n  line2
      - tools 陣列：  tools:\n  - name: ...\n    script: ...
    """
    text = text.strip()
    if not text.startswith("---"):
        return {}, text

    end_idx = text.find("---", 3)
    if end_idx == -1:
        return {}, text

    fm_raw = text[3:end_idx].strip()
    body   = text[end_idx + 3:].strip()

    fm = _parse_yaml_lite(fm_raw)
    return fm, body


def _parse_yaml_lite(yaml_str: str) -> Dict[str, Any]:
    """極簡 YAML-like 解析器，支援本專案 frontmatter 用到的格式。"""
    result: Dict[str, Any] = {}
    lines  = yaml_str.splitlines()
    i      = 0

    while i < len(lines):
        line = lines[i]

        # 跳過空行
        if not line.strip():
            i += 1
            continue

        # 頂層 key: value
        top_match = re.match(r'^(\w+):\s*(.*)', line)
        if not top_match:
            i += 1
            continue

        key, val = top_match.group(1), top_match.group(2).strip()

        # 多行字串（key: >）
        if val == ">":
            i += 1
            block_lines: List[str] = []
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                block_lines.append(lines[i].strip())
                i += 1
            result[key] = " ".join(block_lines)
            continue

        # 陣列（key: 後面是空的，下一行開始 -）
        if val == "" and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("- "):
            i += 1
            items: List[Any] = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                # 單行陣列元素 "- value"
                item_line = lines[i].lstrip()[2:].strip()
                item: Dict[str, str] = {}

                # 如果 item_line 本身是 key: value
                kv = re.match(r'^(\w+):\s*(.*)', item_line)
                if kv:
                    item[kv.group(1)] = kv.group(2).strip()
                else:
                    item["value"] = item_line

                # 後續縮排行（屬於同一個 list item）
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    sub = lines[i].strip()
                    sub_kv = re.match(r'^(\w+):\s*(.*)', sub)
                    if sub_kv:
                        item[sub_kv.group(1)] = sub_kv.group(2).strip()
                    i += 1

                items.append(item)
            result[key] = items
            continue

        # 普通字串
        result[key] = val
        i += 1

    return result


# ══════════════════════════════════════════════════════════════
# 目錄掃描 helpers（新增：取代嚴格依賴 frontmatter 的判斷邏輯）
# ══════════════════════════════════════════════════════════════

# 上傳/掃描時一律忽略的目錄與檔案（快取、版控、系統雜訊）
_IGNORED_DIR_NAMES = {
    "__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".pytest_cache",
}
_IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}

# 這些目錄雖然可能含有 .py（例如 examples/ 內的展示腳本），
# 但不視為「正式函式庫程式碼」，不計入 has_functions 判斷，
# 也不掛進 sys.path（避免 import 到範例/測試碼造成混淆）。
_NON_LIBRARY_TOP_DIRS = {"references", "examples", "tests", "test"}


def _is_ignored_path(rel_path: Path) -> bool:
    """判斷相對路徑是否為應忽略的雜訊檔案／目錄底下的檔案。"""
    if any(part in _IGNORED_DIR_NAMES for part in rel_path.parts):
        return True
    if rel_path.name.startswith(".") or rel_path.suffix in _IGNORED_FILE_SUFFIXES:
        return True
    return False


def _scan_code_dirs(skill_dir: Path) -> List[str]:
    """掃描 skill_dir 底下實際存在的「函式庫程式碼」目錄，取代 frontmatter `tools:` 判斷。

    慣例：
      - 若根目錄直接有 .py 檔案（不含 references/examples/tests 底下的），
        回傳 "."，代表根目錄本身要掛進 sys.path。
      - 若有一級子目錄（例如 scripts/）底下（含巢狀）有 .py 檔案，
        且該子目錄名稱不在 _NON_LIBRARY_TOP_DIRS 內，回傳該子目錄名稱。

    回傳的清單只用來決定：
      1. RCSkill.has_functions（是否掛載 run_skill_code sandbox 工具）
      2. sandbox 內 sys.path 要額外掛載哪些相對目錄（見 tools/skill_code.py）
    不再需要精確對應到個別函式名稱。
    """
    code_dirs: List[str] = []

    root_has_py = any(
        not _is_ignored_path(p.relative_to(skill_dir))
        for p in skill_dir.glob("*.py")
    )
    if root_has_py:
        code_dirs.append(".")

    for child in sorted(skill_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in _NON_LIBRARY_TOP_DIRS or child.name in _IGNORED_DIR_NAMES:
            continue
        if child.name.startswith("."):
            continue

        has_py = any(
            not _is_ignored_path(p.relative_to(skill_dir))
            for p in child.rglob("*.py")
        )
        if has_py:
            code_dirs.append(child.name)

    return code_dirs


def _scan_references(skill_dir: Path) -> List[ReferenceEntry]:
    """直接掃描 skill_dir/references/ 目錄，取代 frontmatter `references:` 宣告清單。

    只要作者把檔案放進 references/，就會自動被收錄，不需要在
    frontmatter 額外重複宣告一次檔名。
    """
    refs_dir = skill_dir / "references"
    if not refs_dir.exists() or not refs_dir.is_dir():
        return []

    entries: List[ReferenceEntry] = []
    for p in sorted(refs_dir.rglob("*")):
        if not p.is_file():
            continue
        rel_path = p.relative_to(skill_dir)
        if _is_ignored_path(rel_path):
            continue
        entries.append(ReferenceEntry(rel_path=str(rel_path), name=p.name))
    return entries


def _list_references(skill_dir: Path, ref_paths: List[str]) -> List[ReferenceEntry]:
    """向後相容 fallback：依 frontmatter 宣告的檔名清單列出 references。

    僅在 _scan_references() 掃描為空、但 frontmatter 仍有宣告時使用，
    處理 references 檔案不在 references/ 目錄底下的邊界情況。
    """
    entries: List[ReferenceEntry] = []
    for ref_rel in ref_paths:
        ref_path = skill_dir / ref_rel
        if not ref_path.exists():
            print(f"[SkillLoader] reference not found: {ref_path}")
            continue
        entries.append(ReferenceEntry(rel_path=ref_rel, name=ref_path.name))
    return entries


def _build_reference_index_text(entries: List[ReferenceEntry]) -> str:
    """將 reference 清單格式化為索引文字，附加在 prompt_context 末尾。"""
    if not entries:
        return ""
    lines = "\n".join(f"- {e.name}" for e in entries)
    return (
        "\n\n## 可查閱的參考資料\n\n"
        f"{lines}\n\n"
        "如需查閱以上任一份文件的完整內容，請呼叫 read_skill_reference 工具。"
    )


# ══════════════════════════════════════════════════════════════
# Functions — frontmatter 精細宣告（選用加分項）
# ══════════════════════════════════════════════════════════════

def _build_function_index_text(specs: List[SkillFunctionSpec]) -> str:
    """將 frontmatter 宣告的函式清單格式化為索引文字，附加在 prompt_context 末尾。

    此清單只用於告知 LLM「sandbox 內有哪些函式可以 import 呼叫」，
    實際執行由 run_skill_code 工具在 e2b sandbox 內完成
    （見 tools/skill_code.py）。frontmatter 未宣告時改用
    _build_generic_sandbox_hint()。
    """
    if not specs:
        return ""
    lines = "\n".join(
        f"- `{s.function}`（{s.script}） — {s.description}" if s.description
        else f"- `{s.function}`（{s.script}）"
        for s in specs
    )
    return (
        "\n\n## 此 Skill 沙盒內可直接 import 的函式\n\n"
        f"{lines}\n\n"
        "呼叫 run_skill_code 執行程式碼時，這些模組已可直接 import 使用"
        "（不需自行處理 sys.path）。"
    )


def _build_generic_sandbox_hint(code_dirs: List[str]) -> str:
    """frontmatter 未宣告 tools 時的通用 sandbox 提示（新增）。

    此時 prompt_context 已包含 SKILL.md body 全文（通常含使用範例與
    檔案結構說明），LLM 可直接依該說明撰寫程式碼；這裡只需告知
    「整個目錄已掛載，且相關程式碼目錄已在 Python path 上」，
    不需要重複列出精確函式清單。
    """
    dir_names = [d for d in code_dirs if d != "."]
    dir_hint  = "、".join(f"`{d}/`" for d in dir_names) if dir_names else "skill 根目錄"
    return (
        "\n\n## Sandbox 使用提示\n\n"
        f"此 Skill 的完整目錄（含 {dir_hint} 等）已掛載到沙盒環境，"
        "其中的程式碼目錄已加入 Python path，可依照上方說明直接 import "
        "對應模組或執行程式碼，不需要額外處理路徑設定。"
    )


# ══════════════════════════════════════════════════════════════
# GCS 輔助下載函數
# ══════════════════════════════════════════════════════════════
import zipfile
import os
from pathlib import Path
from google.cloud import storage

def _download_gcs_skill_file(bucket_name: str, gcs_blob_path: str, local_dir: Path) -> bool:
    """下載 .skill (ZIP) 檔案，並自動剔除 ZIP 內部的第一層根目錄，直接解壓到 local_dir"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_path)
        
        if not blob.exists():
            print(f"[SkillLoader] GCS file not found: {gcs_blob_path}")
            return False
            
        # 1. 下載為本地暫存的 zip 檔
        zip_tmp_path = local_dir.with_suffix('.zip')
        local_dir.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(zip_tmp_path))
        
        # 2. 解壓縮並動態修正路徑
        with zipfile.ZipFile(zip_tmp_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # 剔除路徑中的第一層資料夾名稱
                # 例如 "dsge-model/references/equations.md" -> 拆成 ["dsge-model", "references", "equations.md"]
                parts = Path(member.filename).parts
                
                # 如果只有一層（代表是外層根目錄本身，例如 "dsge-model/"），直接跳過不建立
                if len(parts) <= 1:
                    continue
                
                # 重新組合路徑：去掉第一層 (parts[0])，保留後續所有結構
                relative_path = Path(*parts[1:])
                target_path = local_dir / relative_path
                
                if member.is_dir():
                    # 如果是目錄，建立目錄即可
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    # 如果是檔案，確保上層目錄存在，然後寫入檔案
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                        
        # 3. 刪除暫存的 zip 檔
        if zip_tmp_path.exists():
            zip_tmp_path.unlink()
            
        return True
    except Exception as e:
        print(f"[SkillLoader] GCS .skill Download & Flatten Error: {e}")
        return False




# ══════════════════════════════════════════════════════════════
# SkillLoader (重構版)
# ══════════════════════════════════════════════════════════════
class SkillLoader:
    def __init__(self, bucket_name: str, gcs_prefix: str = "skills"):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix.strip("/")
        self._local_base_dir = Path("/tmp/cloud_run_skills")
        self._local_base_dir.mkdir(parents=True, exist_ok=True)

        # 手動快取：只快取成功結果，失敗（GCS 抖動、SKILL.md 缺失等）不快取，
        # 讓下一次呼叫可以重試，而不是被永久記成「沒有 skill」。
        self._cache:      Dict[str, RCSkill] = {}
        self._cache_lock  = Lock()

    def load(self, rc_name: str) -> Optional[RCSkill]:
        with self._cache_lock:
            cached = self._cache.get(rc_name)
        if cached is not None:
            return cached

        skill = self._load_uncached(rc_name)

        if skill is not None:
            with self._cache_lock:
                self._cache[rc_name] = skill

        return skill

    def _load_uncached(self, rc_name: str) -> Optional[RCSkill]:
        slug = self._slug(rc_name)

        gcs_skill_folder = f"{self.gcs_prefix}/{rc_name}.skill"
        local_skill_dir = self._local_base_dir / slug

        print(f"[SkillLoader] Fetching '{rc_name}' from GCS bucket '{self.bucket_name}'...")
        success = _download_gcs_skill_file(self.bucket_name, gcs_skill_folder, local_skill_dir)

        if not success or not local_skill_dir.exists():
            print(f"[SkillLoader] no skill found in GCS for rc_name='{rc_name}' (slug='{slug}')")
            return None

        skill_md_path = local_skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            print(f"[SkillLoader] SKILL.md not found in {local_skill_dir}")
            return None

        try:
            raw_text          = skill_md_path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(raw_text)
        except Exception as exc:
            print(f"[SkillLoader] failed to read SKILL.md for '{rc_name}': {exc}")
            return None

        # ── references：優先掃描實際目錄，frontmatter 宣告清單僅作向後相容 fallback ──
        reference_entries = _scan_references(local_skill_dir)
        if not reference_entries and frontmatter.get("references"):
            ref_paths_raw = [
                r.get("value", r) if isinstance(r, dict) else r
                for r in frontmatter.get("references", [])
            ]
            reference_entries = _list_references(local_skill_dir, ref_paths_raw)

        # ── functions：優先掃描目錄結構，決定是否需要掛載 sandbox ──
        code_dirs = _scan_code_dirs(local_skill_dir)

        # frontmatter 的 tools: 宣告若存在，仍解析出來，
        # 用於在 prompt 中提供更精細的函式說明（選用加分項，
        # 不影響 has_functions／是否掛載 sandbox 的判斷）
        function_specs: List[SkillFunctionSpec] = []
        for t in frontmatter.get("tools", []):
            if not isinstance(t, dict):
                continue
            if not t.get("script") or not t.get("function"):
                continue
            function_specs.append(SkillFunctionSpec(
                name=t.get("name", t["function"]),
                script=t["script"],
                function=t["function"],
                description=t.get("description", ""),
            ))

        prompt_context = body
        if function_specs:
            prompt_context += _build_function_index_text(function_specs)
        elif code_dirs:
            prompt_context += _build_generic_sandbox_hint(code_dirs)
        prompt_context += _build_reference_index_text(reference_entries)

        actual_rc_name = frontmatter.get("rc_name", rc_name)

        print(f"[SkillLoader] successfully loaded from GCS '{actual_rc_name}': "
              f"code_dirs={code_dirs}, "
              f"{len(function_specs)} declared functions, "
              f"{len(reference_entries)} references, "
              f"{len(prompt_context)} chars of context")

        return RCSkill(
            rc_name=actual_rc_name,
            prompt_context=prompt_context,
            function_specs=function_specs,
            reference_paths=reference_entries,
            code_dirs=code_dirs,
            skill_dir=local_skill_dir,
        )

    @staticmethod
    def _slug(rc_name: str) -> str:
        s = rc_name.lower()
        s = re.sub(r"[^a-z0-9\s_]", "", s)
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"_+", "_", s)
        return s