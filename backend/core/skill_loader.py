# -*- coding: utf-8 -*-


"""skill_loader.py — RC Skill 載入器 (GCS 雲端相容版)

References 採 lazy loading：
  - SKILL.md frontmatter 的 references 只解析出「檔名清單」放進 RCSkill.reference_paths
  - 不在此處讀取檔案內容，避免每次注入完整 prompt_context 造成 token 浪費
  - 實際內容改由 core/main_agent.py 掛載的 read_skill_reference 工具按需讀取
    （見 tools/knowledge.py 的 make_read_skill_reference_tool）

Functions（原 tools）採純 metadata 設計：
  - SKILL.md frontmatter 的 tools[] 只解析為 SkillFunctionSpec 清單（純文字描述），
    不再動態 importlib 執行 skill 內程式碼、不再包成 StructuredTool。
  - 實際執行改由 MainAgent 掛載的泛用 run_skill_code 工具，
    在 e2b sandbox 內直接 import 這些函式執行
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
    """SKILL.md frontmatter 中單一函式的描述（純文件用途，不再動態 import）。

    此函式實際存在於 skill_dir 底下的 script 檔案中，
    由 LLM 透過 run_skill_code 工具在 sandbox 內自行 import 呼叫，
    此處只用來組裝 prompt_context 中的說明文字。
    """
    name:        str   # 顯示名稱（供 prompt 說明用，預設等同 function）
    script:      str   # 相對於 skill_dir 的路徑
    function:    str   # 函數名稱
    description: str = ""


@dataclass
class ReferenceEntry:
    """SKILL.md frontmatter 中單一 reference 的設定（僅檔名/路徑，不含內容）。"""
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
    function_specs:  List[SkillFunctionSpec] = field(default_factory=list)  # 此 skill 提供的函式索引（純文件）
    reference_paths: List[ReferenceEntry]    = field(default_factory=list)  # references 索引（lazy）
    skill_dir:       Optional[Path]          = None      # 本地 skill 目錄，供 run_skill_code / read_skill_reference 使用

    @property
    def has_functions(self) -> bool:
        return len(self.function_specs) > 0

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
# References — lazy loading（只列清單，不讀內容）
# ══════════════════════════════════════════════════════════════

def _list_references(skill_dir: Path, ref_paths: List[str]) -> List[ReferenceEntry]:
    """列出 references 的檔名索引，不讀取檔案內容。

    內容改由 tools/knowledge.py 的 read_skill_reference 工具按需讀取，
    避免每次都把所有 references 全文塞進 prompt_context。
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
# Functions — 純 metadata（只列清單，不動態 import 執行）
# ══════════════════════════════════════════════════════════════

def _build_function_index_text(specs: List[SkillFunctionSpec]) -> str:
    """將函式清單格式化為索引文字，附加在 prompt_context 末尾。

    此清單只用於告知 LLM「sandbox 內有哪些函式可以 import 呼叫」，
    實際執行由 run_skill_code 工具在 e2b sandbox 內完成
    （見 tools/skill_code.py）。
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

        ref_paths_raw = [r.get("value", r) if isinstance(r, dict) else r
                          for r in frontmatter.get("references", [])]
        reference_entries = _list_references(local_skill_dir, ref_paths_raw)

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
        prompt_context += _build_function_index_text(function_specs)
        prompt_context += _build_reference_index_text(reference_entries)

        actual_rc_name = frontmatter.get("rc_name", rc_name)

        print(f"[SkillLoader] successfully loaded from GCS '{actual_rc_name}': "
              f"{len(function_specs)} functions, {len(reference_entries)} references, "
              f"{len(prompt_context)} chars of context")

        return RCSkill(
            rc_name=actual_rc_name,
            prompt_context=prompt_context,
            function_specs=function_specs,
            reference_paths=reference_entries,
            skill_dir=local_skill_dir,
        )

    @staticmethod
    def _slug(rc_name: str) -> str:
        s = rc_name.lower()
        s = re.sub(r"[^a-z0-9\s_]", "", s)
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"_+", "_", s)
        return s