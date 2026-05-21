# -*- coding: utf-8 -*-


"""skill_loader.py — RC Skill 載入器 (GCS 雲端相容版)"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import StructuredTool
from google.cloud import storage  # ── 引入 GCS SDK ──

# YAML / TOML frontmatter 解析（僅依賴標準庫可解析的純文字格式）
# 使用簡易 regex 解析 YAML-like frontmatter，避免額外依賴


# ══════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════

@dataclass
class ToolConfig:
    """SKILL.md frontmatter 中單一工具的設定。"""
    name:        str
    script:      str                    # 相對於 skill_dir 的路徑
    function:    str                    # 函數名稱
    description: str = ""


@dataclass
class RCSkill:
    """載入完成的 RC Skill，可直接注入 MainAgent。"""
    rc_name:        str
    prompt_context: str                     # 注入 system prompt 的動態區塊
    tools:          List[StructuredTool] = field(default_factory=list)

    @property
    def has_tools(self) -> bool:
        return len(self.tools) > 0


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
# Dynamic script importer
# ══════════════════════════════════════════════════════════════

def _import_function(script_path: Path, function_name: str) -> Optional[Callable]:
    """從 .py 檔動態 import 指定函數。"""
    if not script_path.exists():
        print(f"[SkillLoader] script not found: {script_path}")
        return None

    module_name = f"_skill_script_{script_path.stem}_{abs(hash(str(script_path)))}"

    try:
        spec   = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[SkillLoader] failed to import {script_path}: {exc}")
        return None

    func = getattr(module, function_name, None)
    if func is None:
        print(f"[SkillLoader] function '{function_name}' not found in {script_path}")
    return func


def _build_structured_tool(
    skill_dir: Path,
    cfg: ToolConfig,
) -> Optional[StructuredTool]:
    """根據 ToolConfig 建立 LangChain StructuredTool。"""
    script_path = skill_dir / cfg.script
    func = _import_function(script_path, cfg.function)
    if func is None:
        return None

    try:
        return StructuredTool.from_function(
            func=func,
            name=cfg.name,
            description=cfg.description or inspect.getdoc(func) or "",
        )
    except Exception as exc:
        print(f"[SkillLoader] failed to build tool '{cfg.name}': {exc}")
        return None


# ══════════════════════════════════════════════════════════════
# References loader
# ══════════════════════════════════════════════════════════════

def _load_references(skill_dir: Path, ref_paths: List[str]) -> str:
    """載入 references/*.md，串接為補充 context 文字。"""
    chunks: List[str] = []
    for ref_rel in ref_paths:
        ref_path = skill_dir / ref_rel
        if not ref_path.exists():
            print(f"[SkillLoader] reference not found: {ref_path}")
            continue
        try:
            content = ref_path.read_text(encoding="utf-8").strip()
            if content:
                chunks.append(f"### {ref_path.name}\n\n{content}")
        except Exception as exc:
            print(f"[SkillLoader] failed to read {ref_path}: {exc}")

    return "\n\n---\n\n".join(chunks)









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
    """RC Skill 的單例載入器 (GCS 雲端相容版)

    Usage
    -----
    # 建立時改傳入 GCS bucket 名稱
    loader = SkillLoader(bucket_name="my-claud-skills-bucket", gcs_prefix="skills")
    skill  = loader.load("DSGE Model")   # -> 第一次會從 GCS 抓取並快取
    """

    def __init__(self, bucket_name: str, gcs_prefix: str = "skills"):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix.strip("/")
        # 本地暫存目錄設在 Cloud Run 的記憶體磁碟 /tmp
        self._local_base_dir = Path("/tmp/cloud_run_skills")
        self._local_base_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=64)
    def load(self, rc_name: str) -> Optional[RCSkill]:
        slug = self._slug(rc_name)
        
        # 1. 決定 GCS 的路徑與本地 /tmp 的暫存路徑
        gcs_skill_folder = f"{self.gcs_prefix}/{rc_name}.skill"
        local_skill_dir = self._local_base_dir / slug

        # 2. 從 GCS 下載該技能的完整目錄 (SKILL.md, scripts/, references/)
        # 注意：因為有 lru_cache，同一個技能在同一個實例中只會被下載一次
        print(f"[SkillLoader] Fetching '{rc_name}' from GCS bucket '{self.bucket_name}'...")
        success = _download_gcs_skill_file(self.bucket_name, gcs_skill_folder, local_skill_dir)
        
        if not success or not local_skill_dir.exists():
            print(f"[SkillLoader] no skill found in GCS for rc_name='{rc_name}' (slug='{slug}')")
            return None

        skill_md_path = local_skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            print(f"[SkillLoader] SKILL.md not found in {local_skill_dir}")
            return None

        # ── 後續邏輯完全不需要動，直接讀取 local_skill_dir ──
        try:
            raw_text          = skill_md_path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(raw_text)
        except Exception as exc:
            print(f"[SkillLoader] failed to read SKILL.md for '{rc_name}': {exc}")
            return None

        # ── prompt_context：SKILL.md 正文 + references ──
        ref_paths      = [r.get("value", r) if isinstance(r, dict) else r
                          for r in frontmatter.get("references", [])]
        ref_context    = _load_references(local_skill_dir, ref_paths) # 傳入本地路徑
        prompt_context = body
        if ref_context:
            prompt_context += "\n\n## 參考資料\n\n" + ref_context

        # ── tools：從 frontmatter 的 tools[] 動態載入 ──
        tool_cfgs: List[ToolConfig] = []
        for t in frontmatter.get("tools", []):
            if not isinstance(t, dict):
                continue
            if not t.get("name") or not t.get("script") or not t.get("function"):
                continue
            tool_cfgs.append(ToolConfig(
                name=t["name"],
                script=t["script"],
                function=t["function"],
                description=t.get("description", ""),
            ))

        tools: List[StructuredTool] = []
        for cfg in tool_cfgs:
            tool = _build_structured_tool(local_skill_dir, cfg) # 傳入本地路徑
            if tool is not None:
                tools.append(tool)

        actual_rc_name = frontmatter.get("rc_name", rc_name)

        print(f"[SkillLoader] successfully loaded from GCS '{actual_rc_name}': "
              f"{len(tools)} tools, {len(prompt_context)} chars of context")

        return RCSkill(
            rc_name=actual_rc_name,
            prompt_context=prompt_context,
            tools=tools,
        )

    @staticmethod
    def _slug(rc_name: str) -> str:
        s = rc_name.lower()
        s = re.sub(r"[^a-z0-9\s_]", "", s)
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"_+", "_", s)
        return s





