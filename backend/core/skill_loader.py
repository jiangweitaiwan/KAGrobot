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

【新增 — HTML Widget】
  比照 references / code_dirs 的「目錄掃描為主」精神：skill_dir/html/
  底下若存在 manifest.yaml，就解析成 List[HtmlWidgetSpec]，放進
  RCSkill.html_widgets。與 references 不同的地方是 HTML 全文本身也不
  在此處讀取（比 references 更晚才讀）——manifest 只提供 widget_id /
  title / description / compute_functions 這些「索引層」資訊，供
  system prompt 讓 LLM 知道有哪些 widget 可用；實際 HTML 全文要等
  LLM 呼叫 show_interactive_html 工具時才讀取並上傳（見
  tools/skill_widget.py），避免不必要的檔案 I/O 與 prompt token 浪費。

  compute_functions 的白名單（action_name -> "module_path:func_name"）
  也在這裡一併解析好，直接交給 core/widget_token.py::issue_token 當作
  token payload 的一部分，dispatch endpoint（POST /skill/widget/dispatch）
  驗證時不需要重新解析 manifest.yaml。
"""

from __future__ import annotations

import re
import sys
import os
import shutil
from dataclasses import dataclass, field
from threading import Lock   # ← 取代 functools.lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from google.cloud import storage  # ── 引入 GCS SDK ──

# YAML / TOML frontmatter 解析（僅依賴標準庫可解析的純文字格式）
# 使用簡易 regex 解析 YAML-like frontmatter，避免額外依賴
# （注意：html/manifest.yaml 走的是正規 PyYAML，見下方 _scan_html_widgets，
#  與 SKILL.md frontmatter 的 mini parser 是兩套獨立機制，不要混用）


# ══════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════




@dataclass
class ReferenceEntry:
    """單一 reference 的設定（僅檔名/路徑，不含內容）。"""
    rel_path: str    # 相對於 skill_dir 的路徑，例如 "references/equations.md"
    name:     str     # 檔名，例如 "equations.md"（供 read_skill_reference 對照用）


@dataclass
class HtmlWidgetSpec:
    """單一 HTML 互動元件的設定（僅索引資訊，不含 HTML 全文）。

    compute_functions : {action_name: "module_path:func_name"}
        module_path 相對於 skill_dir（例如 "scripts/risk_calc.py"），
        func_name 是該模組內的函式名稱。這份白名單會被原樣放進
        widget_token 的 payload，dispatch endpoint 只允許呼叫這裡列出
        的 action，其餘一律拒絕（見 core/skill_compute.py）。
    """
    widget_id:         str
    rel_path:          str                 # 相對 skill_dir，例如 "html/risk_input.html"
    title:             str
    description:       str
    compute_functions: Dict[str, str] = field(default_factory=dict)


@dataclass
class RCSkill:
    """載入完成的 RC Skill，可直接注入 MainAgent。

    純 metadata 設計：不含任何執行期狀態（無 sandbox_id、無已 import 的函式物件）。
    執行相關的狀態（e2b sandbox_id）由呼叫端逐 session 保存於
    OrchestratorState.skill_sandbox_id，不放進此 dataclass 或 SkillLoader 的快取。
    """
    rc_name:         str
    prompt_context:  str                                 # 注入 system prompt 的動態區塊（不含 references 全文）
    reference_paths: List[ReferenceEntry]    = field(default_factory=list)  # references 索引（lazy）
    code_dirs:       List[str]               = field(default_factory=list)  # 新增：含 .py 檔案的一級子目錄清單（相對 skill_dir），
                                                                              # "." 代表根目錄本身，供 sandbox 掛載 sys.path 使用
    html_widgets:    List[HtmlWidgetSpec]    = field(default_factory=list)  # 新增：html/manifest.yaml 解析出的 widget 索引（lazy）
    skill_dir:       Optional[Path]          = None      # 本地 skill 目錄，供 run_skill_code / read_skill_reference / show_interactive_html 使用

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

    @property
    def has_html(self) -> bool:
        """是否需要掛載 show_interactive_html 工具。"""
        return len(self.html_widgets) > 0


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
# 注意：html/ 底下的 scripts/ 是給 compute_functions 用的獨立子目錄，
# 不受這個排除清單影響——它仍然可能同時是 has_functions 的一部分
# （skill 若同時提供 run_skill_code 用的 scripts/ 與 widget 用的
# compute function，兩者實務上常常就是同一個 scripts/ 目錄）。
_NON_LIBRARY_TOP_DIRS = {"references", "examples", "tests", "test", "html"}


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
      - 若根目錄直接有 .py 檔案（不含 references/examples/tests/html 底下的），
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


def _scan_html_widgets(skill_dir: Path) -> List[HtmlWidgetSpec]:
    """解析 skill_dir/html/manifest.yaml，回傳 widget 索引清單。

    manifest.yaml 格式（見 html_skill 設計文件）：

        widgets:
          - id: risk_input
            file: risk_input.html
            title: 風險參數輸入
            description: 讓使用者用滑桿輸入部位大小、槓桿、停損比例
            compute_functions:
              - name: preview_var
                module_path: scripts/risk_calc.py
                function: preview_var   # 選填，預設同 name

    沒有 html/manifest.yaml、解析失敗、或宣告的 file 實際不存在時，
    一律回傳空清單（best-effort，不讓整個 skill 因為 widget 設定錯誤
    而載入失敗——這與 references/code_dirs 的容錯精神一致）。
    """
    manifest_path = skill_dir / "html" / "manifest.yaml"
    if not manifest_path.exists():
        return []

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[SkillLoader] html/manifest.yaml 解析失敗（{skill_dir}）: {exc}")
        return []

    widgets: List[HtmlWidgetSpec] = []
    for w in (raw.get("widgets") or []):
        try:
            widget_id = w["id"]
            file_name = w["file"]
        except (KeyError, TypeError):
            print(f"[SkillLoader] html/manifest.yaml 內有 widget 缺少 id/file，已跳過: {w}")
            continue

        rel_path = f"html/{file_name}"
        if not (skill_dir / rel_path).exists():
            print(f"[SkillLoader] widget '{widget_id}' 宣告的檔案不存在: {rel_path}，已跳過")
            continue

        compute_functions: Dict[str, str] = {}
        for cf in (w.get("compute_functions") or []):
            try:
                name        = cf["name"]
                module_path = cf["module_path"]
            except (KeyError, TypeError):
                print(f"[SkillLoader] widget '{widget_id}' 的 compute_functions "
                      f"設定缺少 name/module_path，已跳過該項: {cf}")
                continue
            func_name = cf.get("function", name)
            compute_functions[name] = f"{module_path}:{func_name}"

        widgets.append(HtmlWidgetSpec(
            widget_id=widget_id,
            rel_path=rel_path,
            title=w.get("title", widget_id),
            description=w.get("description", ""),
            compute_functions=compute_functions,
        ))

    return widgets


def _build_resource_index_text(
    code_dirs:    List[str],
    references:   List[ReferenceEntry],
    html_widgets: List[HtmlWidgetSpec],
) -> str:
    """列出此 Skill 附帶的資源（純 inventory：只說「有什麼」，
    不描述存取方式或工具名——是否掛載沙盒、怎麼存取，
    由 MainAgent 依實際 deployment 狀態決定，避免兩邊語意打架。
    """
    if not code_dirs and not references and not html_widgets:
        return ""

    lines = ["\n\n## 此 Skill 附帶的資源"]

    if code_dirs:
        dir_names = [d for d in code_dirs if d != "."]
        dir_hint  = "、".join(f"`{d}/`" for d in dir_names) if dir_names else "skill 根目錄"
        lines.append(f"- 程式碼：{dir_hint}")

    if references:
        file_list = "、".join(f"`{r.name}`" for r in references)
        lines.append(f"- 參考文件（references/）：{file_list}")

    if html_widgets:
        widget_list = "、".join(
            f"`{w.widget_id}`（{w.title}）" for w in html_widgets
        )
        lines.append(f"- 互動元件（html/）：{widget_list}")

    return "\n".join(lines)




# ══════════════════════════════════════════════════════════════
# GCS 輔助下載函數
# ══════════════════════════════════════════════════════════════
import zipfile
import os
from pathlib import Path
from google.cloud import storage


def _detect_common_root(infolist: List[zipfile.ZipInfo]) -> Optional[str]:
    """判斷 zip 內所有項目是否共用同一個第一層資料夾（「有根目錄」的打包方式）。

    兩種常見結構：
      1. 有根目錄：所有檔案都包在同一層資料夾底下，
         例如 "black-litterman-skill/SKILL.md"、
              "black-litterman-skill/references/equations.md"
         → 第一層路徑元件只有一種值，回傳該值。
      2. 無根目錄：檔案直接放在 zip 頂層，
         例如 "SKILL.md"、"references/equations.md"、"scripts/xxx.py"
         → 第一層路徑元件有多種值（SKILL.md / references / scripts / examples），
         回傳 None，代表不應剝除任何路徑層。

    只看第一層路徑元件是否唯一，不管檔案深度，足以區分這兩種打包方式，
    也不會誤判「無根目錄但恰好只有一個頂層資料夾」的邊界情況——
    這種情況即使誤判為「有根目錄」而剝除一層，效果與正確判斷也一致
    （因為那個頂層資料夾本來就是唯一內容）。
    """
    tops = set()
    for member in infolist:
        name = member.filename.strip("/")
        if not name:
            continue
        parts = Path(name).parts
        if not parts:
            continue
        tops.add(parts[0])

    if len(tops) == 1:
        return next(iter(tops))
    return None


def _download_gcs_skill_file(bucket_name: str, gcs_blob_path: str, local_dir: Path) -> bool:
    """下載 .skill (ZIP) 檔案並解壓到 local_dir。

    相容兩種 zip 內部結構（見 _detect_common_root 說明）：
      1. 有外層根目錄包住所有檔案 → 自動剔除該層根目錄，
         把內容直接攤平到 local_dir（例如 SKILL.md 落在
         local_dir/SKILL.md，而不是 local_dir/<root>/SKILL.md）。
      2. 沒有外層根目錄，檔案直接在 zip 頂層 → 路徑原樣解壓，
         不做任何剝除，避免像 "SKILL.md"（只有一層路徑）被誤判為
         「根目錄本身的 entry」而整個跳過不解壓。
    """
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

        # 2. 解壓縮並依偵測結果決定是否剝除第一層路徑
        with zipfile.ZipFile(zip_tmp_path, 'r') as zip_ref:
            infolist    = zip_ref.infolist()
            common_root = _detect_common_root(infolist)

            for member in infolist:
                parts = Path(member.filename).parts
                if not parts:
                    continue

                if common_root is not None:
                    # 有根目錄：剔除第一層（root 目錄本身的 entry 直接跳過）
                    if len(parts) <= 1:
                        continue
                    relative_path = Path(*parts[1:])
                else:
                    # 無根目錄：路徑原樣使用，不做任何轉換
                    relative_path = Path(*parts)

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
        # ── skill_exists 專用：輕量 client + 獨立 cache ──────────
        # 與 _cache 分開，因為呼叫頻率與用途不同：單一 rc-paths 請求
        # 內可能對同一個 rc_name 被多個不同 anchor 展開路徑重複詢問。
        self._gcs_client:       Optional[storage.Client] = None
        self._gcs_client_lock   = Lock()
        self._exists_cache:      Dict[str, bool] = {}
        self._exists_cache_lock  = Lock()

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

        # ── references：直接掃描實際目錄 ──
        reference_entries = _scan_references(local_skill_dir)

        # ── functions：掃描目錄結構，決定是否需要掛載 sandbox ──
        code_dirs = _scan_code_dirs(local_skill_dir)

        # ── html widgets：解析 html/manifest.yaml（新增）──
        html_widgets = _scan_html_widgets(local_skill_dir)

        prompt_context = body + _build_resource_index_text(
            code_dirs, reference_entries, html_widgets
        )

        actual_rc_name = frontmatter.get("rc_name", rc_name)

        print(f"[SkillLoader] successfully loaded from GCS '{actual_rc_name}': "
              f"code_dirs={code_dirs}, "
              f"{len(reference_entries)} references, "
              f"{len(html_widgets)} html widgets, "
              f"{len(prompt_context)} chars of context")

        return RCSkill(
            rc_name=actual_rc_name,
            prompt_context=prompt_context,
            reference_paths=reference_entries,
            code_dirs=code_dirs,
            html_widgets=html_widgets,
            skill_dir=local_skill_dir,
        )

    def _get_client(self) -> storage.Client:
        if self._gcs_client is None:
            with self._gcs_client_lock:
                if self._gcs_client is None:
                    self._gcs_client = storage.Client()
        return self._gcs_client

    def skill_exists(self, rc_name: str) -> bool:
        """輕量檢查：此 rc_name 是否有對應的 .skill 存在於 GCS。

        只做 metadata 查詢（blob.exists()），不下載、不解壓，
        成本遠低於 load()。結果快取於記憶體；失敗（GCS 抖動等）
        不快取，允許下次重試,同 load() 的 _cache 設計哲學一致。
        """
        with self._exists_cache_lock:
            cached = self._exists_cache.get(rc_name)
        if cached is not None:
            return cached

        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob   = bucket.blob(f"{self.gcs_prefix}/{rc_name}.skill")
            exists = blob.exists()
        except Exception as exc:
            print(f"[SkillLoader] skill_exists check failed for '{rc_name}': {exc}")
            return False

        with self._exists_cache_lock:
            self._exists_cache[rc_name] = exists
        return exists


    @staticmethod
    def _slug(rc_name: str) -> str:
        s = rc_name.lower()
        s = re.sub(r"[^a-z0-9\s_]", "", s)
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"_+", "_", s)
        return s
