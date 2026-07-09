# -*- coding: utf-8 -*-
"""
tools/params_resolver.py
─────────────────────────────────────────────────────────────
resolve_params 工廠工具：MainAgent 掛給 LLM 使用的參數解析工具。

對應 dsge-model skill 的 references/params_manifest.yaml 協定：
讀 manifest → 依 user_override > tool_retrieval > calibrated_default
解析每個外生參數 → 回傳 resolved / provenance / unresolved 三份清單。

與 dsge-model 自帶的 scripts/resolve_params.py 的差異：
  - 這裡是「後端規則引擎」版本：不透過額外一次 LLM tool call 去呼叫
    retrieve_param，而是直接 import dispatch_retrieve_param() 呼叫，
    省一次來回，也不會因為 LLM 忘記呼叫而漏掉 tool_retrieval。
  - valid_range 檢查採「不擋、只警告」策略：即使使用者輸入超出建議
    範圍，仍採用該值，但在 provenance 附上 warning，交由 LLM 在
    回覆中向使用者說明可靠性風險，不由後端逕自 clip 或拒絕。
  - 本工具與特定 skill 是否提供 params_manifest.yaml 無關，永遠可
    掛載；若 skill_dir 底下找不到 references/params_manifest.yaml，
    直接回傳 ok=False 說明「此 Skill 未提供 manifest」，不影響其餘
    流程（LLM 應照舊自行判斷參數、直接呼叫 run_skill_code）。

user_inputs 的萃取完全交給 LLM 自行從對話中判斷、自行填寫，
本工具不做額外約束層（key 是否對應 manifest 宣告的 name 由 LLM
自律，目前不在後端做名稱校驗）。

結構校準參數（structural_calibrated_parameters）不在此工具處理
範圍內——它們清一色是 calibrated_default 且不查詢、不主動問，一律
留給 skill 自己的 scripts（例如 dsge-model 的
load_structural_defaults()）處理，避免混進來增加不必要的開銷，也
維持與 skill 自身程式碼的分工界線。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import yaml
from langchain_core.tools import StructuredTool

from tools.data_retriever import dispatch_retrieve_param


def make_resolve_params_tool(skill_dir: Path) -> StructuredTool:
    """建立此 Skill 專屬的 resolve_params 工具（closure 綁定 skill_dir）。

    Parameters
    ----------
    skill_dir : 此 skill 在本地（/tmp）解壓後的根目錄
                （對應 RCSkill.skill_dir，manifest 路徑固定為
                skill_dir/references/params_manifest.yaml）
    """
    manifest_path = skill_dir / "references" / "params_manifest.yaml"

    def resolve_params(user_inputs: Optional[Dict[str, Any]] = None) -> str:
        """依此 Skill 的 references/params_manifest.yaml 規則，解析所有外生參數。

        user_inputs 只需填你已從對話中確認、且 manifest 有宣告對應 name
        的參數（例如 {"shock": "monetary_policy"}），其餘（時效性數據
        查詢、校準預設值）交給規則自動處理，不需要你自己判斷優先序
        或呼叫查詢工具。

        回傳三個欄位：
          - resolved   : 可直接餵給模型函式使用的參數值
          - provenance : 每個參數的來源說明（user_override /
                         tool_retrieval / calibrated_default），
                         若數值超出建議範圍會附上 warning，
                         若 tool_retrieval 失敗改用預設值會記在
                         _fallback_log
          - unresolved : 尚待你向使用者確認的問題清單（每項含 name
                         與 question）；非空時應先跟使用者確認，
                         答案齊全後再重新呼叫一次，確認 unresolved
                         為空才可以執行模型。
        """
        if not manifest_path.exists():
            return json.dumps({
                "ok":    False,
                "error": "此 Skill 未提供 references/params_manifest.yaml。",
            }, ensure_ascii=False)

        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return json.dumps({
                "ok":    False,
                "error": f"params_manifest.yaml 解析失敗：{exc}",
            }, ensure_ascii=False)

        resolved, provenance, unresolved = _resolve(manifest or {}, user_inputs or {})

        return json.dumps({
            "ok":         True,
            "resolved":   resolved,
            "provenance": provenance,
            "unresolved": unresolved,
        }, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        func=resolve_params,
        name="resolve_params",
        description=(
            "依此 Skill 的 params_manifest.yaml 解析所有外生參數"
            "（情境定義型 + 時效性數據型）。傳入你已從對話中確認的參數"
            "（key 需對應 manifest 宣告的 name），呼叫後檢查 unresolved："
            "非空則先跟使用者確認清單裡的問題，答案齊全後重新呼叫一次；"
            "為空即可放心把 resolved 的值傳入 run_skill_code 執行模型。"
            "若不確定此 Skill 有沒有提供 manifest，直接呼叫即可，"
            "沒有的話會回傳明確說明，不影響其餘流程。"
        ),
    )


# ──────────────────────────────────────────────────────────
# 私有：規則解析核心
# （與 dsge-model/scripts/resolve_params.py 的 resolve_params()
#  邏輯對應，差異見檔頭說明）
# ──────────────────────────────────────────────────────────

def _resolve(
    manifest:    dict,
    user_inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, str]]]:
    resolved:   Dict[str, Any]        = {}
    provenance: Dict[str, Any]        = {}
    unresolved: List[Dict[str, str]]  = []

    for p in manifest.get("exogenous_parameters", []):
        name = p["name"]

        # Step 1：使用者已明確提供
        if user_inputs.get(name) is not None:
            val     = user_inputs[name]
            warning = _check_range(val, p.get("valid_range"))
            resolved[name]   = val
            provenance[name] = {
                "type": "user_override",
                **({"warning": warning} if warning else {}),
            }
            continue

        # Step 2：需要主動詢問，且無合理預設值 → 交回給 LLM/對話層處理
        if p.get("ask_proactively") and p.get("calibrated_default") is None:
            unresolved.append({
                "name":     name,
                "question": p.get("question") or f"請提供「{p.get('description', name)}」的值",
            })
            continue

        # Step 3：tool_retrieval
        if p.get("source_type") == "tool_retrieval":
            result = dispatch_retrieve_param(
                p["retriever_id"], **(p.get("retriever_kwargs") or {})
            )
            if result.get("success"):
                resolved[name]   = result["value"]
                provenance[name] = {
                    "type":         "tool_retrieval",
                    "source":       result["source"],
                    "as_of_date":   result["as_of_date"],
                    "retrieved_at": result["retrieved_at"],
                }
                continue

            provenance.setdefault("_fallback_log", []).append(
                f"{name}: tool_retrieval 失敗（{result.get('error')}），"
                f"改用 calibrated_default"
            )
            # 查詢失敗 → 落到 Step 4

        # Step 4：calibrated_default（必要 fallback）
        resolved[name]   = p.get("calibrated_default")
        provenance[name] = {
            "type":   "calibrated_default",
            "source": p.get("calibrated_default_source", ""),
        }

    return resolved, provenance, unresolved


def _check_range(val: Any, valid_range: Optional[list]) -> Optional[str]:
    """檢查數值是否落在 manifest 宣告的 valid_range 內。

    採「不擋、只警告」策略：超出範圍仍回傳 warning 字串供上層附註，
    但 resolved 仍採用該值，不因超界而拒絕或 clip。
    valid_range 格式為 [lo, hi]，lo/hi 可為 null（代表該側無上下限）。
    """
    if not valid_range or val is None or len(valid_range) < 2:
        return None

    try:
        numeric_val = float(val)
    except (TypeError, ValueError):
        return None

    lo, hi = valid_range[0], valid_range[1]
    if (lo is not None and numeric_val < lo) or (hi is not None and numeric_val > hi):
        return f"數值 {val} 超出建議範圍 [{lo}, {hi}]，模型結果可靠性可能下降"
    return None
