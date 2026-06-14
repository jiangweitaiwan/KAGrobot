# -*- coding: utf-8 -*-
"""core/registry.py — 全域 Registry 與工具解析函數。

AgentType           : Agent 類型枚舉
AGENTS_REGISTRY     : AgentType → 工具清單
TOOLS_REGISTRY      : tool_name → 工具元描述
CONCEPT_TO_AGENT    : Resulting Concept 名稱 → AgentType

工具解析函數（從 socrates_bot.py 遷移）：
    collect_tools_with_deps()      : 遞迴展開工具依賴
    resolve_agent_type()           : concept 名稱 → AgentType
    resolve_tools_from_candidates(): 候選清單 → (tools, parameters)

新增工具時的步驟：
  1. 在 tools/ 下實作並完成 register_langchain_tool 或 @register_tool
  2. 在本檔 TOOLS_REGISTRY 補充工具描述
  3. 如需對應新 AgentType，在 AGENTS_REGISTRY 加入條目
  4. 如需從 Resulting Concept 名稱自動路由，在 CONCEPT_TO_AGENT 加入映射
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# AgentType
# ═══════════════════════════════════════════════════════════════

class AgentType(Enum):
    GROWTH_SELECT             = "growth_select"
    DCF                       = "dcf"
    COORDINATOR               = "coordinator"
    DSGE_ANALYSIS             = "dsge_analysis"
    NK_BAYESIAN_DSGE_ANALYSIS = "nk_bayesian_dsge_analysis"


# ═══════════════════════════════════════════════════════════════
# CONCEPT_TO_AGENT
# ═══════════════════════════════════════════════════════════════

CONCEPT_TO_AGENT: Dict[str, AgentType] = {
    "growth_select":          AgentType.GROWTH_SELECT,
    "growth stock selection": AgentType.GROWTH_SELECT,
    "dcf":                    AgentType.DCF,
    "discounted cash flow":   AgentType.DCF,
    "Extended New Keynesian DSGE Model with Bayesian Estimation":
        AgentType.NK_BAYESIAN_DSGE_ANALYSIS,
}


# ═══════════════════════════════════════════════════════════════
# AGENTS_REGISTRY
# ═══════════════════════════════════════════════════════════════

AGENTS_REGISTRY: Dict[AgentType, Dict[str, Any]] = {
    AgentType.GROWTH_SELECT: {
        "description": "成長股篩選分析代理",
        "tools": ["growth_select"],
    },
    AgentType.DCF: {
        "description": "DCF 估值分析代理",
        "tools": ["dcf"],
    },
    AgentType.COORDINATOR: {
        "description": "任務協調代理",
        "tools": ["coordinator"],
    },
    AgentType.NK_BAYESIAN_DSGE_ANALYSIS: {
        "description": "DSGE 宏觀經濟分析代理",
        "tools": [
            "identify_shocks",
            "forecast_economy",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# TOOLS_REGISTRY
# ═══════════════════════════════════════════════════════════════

TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ── 財務類 ────────────────────────────────────────────────
    "growth_select": {
        "description": "成長股篩選與分析",
        "skills": ["web_search", "data_collection", "source_validation"],
        "dependencies": ["data_collection"],
        "outputs": ["research_data", "source_list"],
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "股票代碼，例如 'AAPL' 或 'TSLA'，僅支援美股。",
                }
            },
            "required": ["stock_symbol"],
        },
    },
    "dcf": {
        "description": "現金流折現估值分析",
        "skills": ["data_processing", "pattern_recognition", "statistical_analysis"],
        "dependencies": ["data_collection"],
        "outputs": ["analysis_results", "insights"],
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "股票代碼，例如 'AAPL' 或 'TSLA'，僅支援美股。",
                }
            },
            "required": ["stock_symbol"],
        },
    },
    "data_collection": {
        "description": "財務數據收集與整理",
        "skills": ["web_scraping", "api_integration", "data_cleaning"],
        "dependencies": [],
        "outputs": ["financial_data"],
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "股票代碼，例如 'AAPL' 或 'TSLA'，僅支援美股。",
                },
                "data_type": {
                    "type": "string",
                    "description": "資料類型，如 income_statement, balance_sheet, cash_flow",
                },
            },
            "required": ["stock_symbol"],
        },
    },
    "coordinator": {
        "description": "任務協調與資源分配",
        "skills": ["task_coordination", "resource_allocation", "conflict_resolution"],
        "dependencies": [],
        "outputs": ["coordination_plan", "resource_allocation"],
        "parameters": {},
    },

    # ── DSGE 經濟分析類 ───────────────────────────────────────
    "fetch_economic_data": {
        "description": (
            "【數據抓取】自動取得指定國家的最新經濟指標（GDP、通膨、利率、"
            "投資、薪資、就業、消費）。當使用者要求分析特定國家現況或缺乏具體數值時，"
            "**必須優先呼叫此工具**作為分析基礎。"
        ),
        "skills": ["data_retrieval", "api_integration", "economic_analysis"],
        "dependencies": [],
        "outputs": ["economic_indicators", "data_source_info"],
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "國家代碼。支援：US, TW, JP, KR, DE, GB, CN, EU。",
                    "default": "TW",
                },
                "use_realtime": {
                    "type": "boolean",
                    "description": "是否嘗試抓取即時數據（預設 True）。",
                    "default": True,
                },
                "fred_api_key": {
                    "type": "string",
                    "description": "FRED API 金鑰（可選，美國數據更精確）。",
                },
            },
            "required": ["country"],
        },
    },
    "identify_shocks": {
        "description": "識別當前驅動經濟的結構衝擊（技術衝擊、貨幣政策衝擊、供給衝擊等）",
        "skills": ["economic_analysis", "structural_identification", "data_interpretation"],
        "dependencies": [],
        "outputs": ["shock_identification", "economic_state", "chart"],
        "parameters": {
            "type": "object",
            "properties": {
                "gdp_growth":    {"type": "number", "description": "實質 GDP 季成長率 (%)"},
                "inflation":     {"type": "number", "description": "通貨膨脹率，年增率 (%)"},
                "interest_rate": {"type": "number", "description": "政策利率 (%)"},
                "investment":    {"type": "number", "description": "私人投資成長率 (%, 預設 2.0)"},
                "wage_growth":   {"type": "number", "description": "實質薪資年增率 (%, 預設 3.0)"},
                "employment":    {"type": "number", "description": "就業率 (%, 預設 96.0)"},
                "consumption":   {"type": "number", "description": "消費成長率 (%, 預設 1.8)"},
            },
            "required": ["gdp_growth", "inflation", "interest_rate"],
        },
    },
    "forecast_economy": {
        "description": "預測未來 N 季的 GDP、通膨、利率走勢",
        "skills": ["economic_forecasting", "time_series", "macro_modeling"],
        "dependencies": ["fetch_economic_data"],
        "outputs": ["forecast_path", "narrative", "chart"],
        "parameters": {
            "type": "object",
            "properties": {
                "gdp_growth":     {"type": "number", "description": "實質 GDP 季成長率 (%)"},
                "inflation":      {"type": "number", "description": "通貨膨脹率 (%)"},
                "interest_rate":  {"type": "number", "description": "政策利率 (%)"},
                "investment":     {"type": "number", "description": "投資成長率 (%, 預設 2.0)"},
                "wage_growth":    {"type": "number", "description": "薪資年增率 (%, 預設 3.0)"},
                "employment":     {"type": "number", "description": "就業率 (%, 預設 96.0)"},
                "consumption":    {"type": "number", "description": "消費成長率 (%, 預設 1.8)"},
                "horizon":        {"type": "integer", "description": "預測期距（季），建議 4-16（預設 8）"},
                "generate_chart": {"type": "boolean", "description": "是否生成圖表（預設 True）"},
            },
            "required": ["gdp_growth", "inflation", "interest_rate"],
        },
    },
    "scenario_analysis": {
        "description": "模擬政策情境（升降息、財政政策、技術衝擊）對 GDP 和通膨的影響",
        "skills": ["policy_simulation", "scenario_modeling", "impact_analysis"],
        "dependencies": [],
        "outputs": ["scenario_comparison", "impact_metrics", "chart"],
        "parameters": {
            "type": "object",
            "properties": {
                "gdp_growth":       {"type": "number", "description": "實質 GDP 季成長率 (%)"},
                "inflation":        {"type": "number", "description": "通貨膨脹率 (%)"},
                "interest_rate":    {"type": "number", "description": "政策利率 (%)"},
                "rate_change_bps":  {"type": "number", "description": "利率變動（基點，正=升息，預設 0）"},
                "gov_spending_pct": {"type": "number", "description": "政府支出衝擊（%，預設 0）"},
                "tech_shock_pct":   {"type": "number", "description": "技術衝擊（%，預設 0）"},
                "risk_premium_pct": {"type": "number", "description": "風險溢酬衝擊（%，預設 0）"},
                "horizon":          {"type": "integer", "description": "預測期距（季，預設 8）"},
                "generate_chart":   {"type": "boolean", "description": "是否生成圖表（預設 True）"},
            },
            "required": ["gdp_growth", "inflation", "interest_rate"],
        },
    },
    "variance_decomposition": {
        "description": "計算各結構衝擊對 GDP 波動的方差分解（貢獻度）",
        "skills": ["variance_analysis", "risk_attribution", "structural_decomposition"],
        "dependencies": [],
        "outputs": ["variance_shares", "ranked_shocks", "narrative"],
        "parameters": {
            "type": "object",
            "properties": {
                "horizon":   {"type": "integer", "description": "分析期距（季，預設 20）"},
                "variables": {"type": "string",  "description": "分析的變數（目前支援 'gdp'，預設 'gdp'）"},
            },
            "required": [],
        },
    },
    "compare_scenarios": {
        "description": "同時比較多個政策情境，輸出並排對比分析",
        "skills": ["multi_scenario_comparison", "policy_evaluation", "impact_ranking"],
        "dependencies": [],
        "outputs": ["scenario_comparison", "ranking", "chart"],
        "parameters": {
            "type": "object",
            "properties": {
                "gdp_growth":       {"type": "number", "description": "實質 GDP 季成長率 (%)"},
                "inflation":        {"type": "number", "description": "通貨膨脹率 (%)"},
                "interest_rate":    {"type": "number", "description": "政策利率 (%)"},
                "scenario_configs": {
                    "type": "string",
                    "description": (
                        "JSON 字串，包含多個情境設定，"
                        "例如：'[{\"label\":\"降息50bps\",\"rate_change_bps\":-50}]'"
                    ),
                },
                "horizon": {"type": "integer", "description": "預測期距（季，預設 8）"},
            },
            "required": ["gdp_growth", "inflation", "interest_rate"],
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# 工具解析函數（從 socrates_bot.py 遷移）
# ═══════════════════════════════════════════════════════════════

def collect_tools_with_deps(tool_names: List[str]) -> List[str]:
    """遞迴展開工具依賴，回傳依賴優先的有序清單。"""
    visited: set       = set()
    ordered: List[str] = []

    def _walk(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for dep in TOOLS_REGISTRY.get(name, {}).get("dependencies", []):
            _walk(dep)
        ordered.append(name)

    for tn in tool_names:
        _walk(tn)
    return ordered


def resolve_agent_type(
    rc_name:          str,
    concept_to_agent: Optional[Dict[str, AgentType]] = None,
) -> Optional[AgentType]:
    """從 Resulting Concept 名稱解析對應的 AgentType。

    先做精確匹配，再做子字串模糊匹配。
    """
    mapping = concept_to_agent or CONCEPT_TO_AGENT
    name    = rc_name.lower().strip()

    if name in mapping:
        return mapping[name]
    for key, agent in mapping.items():
        if key.lower().strip() in name or name in key.lower().strip():
            return agent
    return None


def resolve_tools_from_candidates(
    candidates:       List[dict],
    concept_to_agent: Optional[Dict[str, AgentType]] = None,
) -> Tuple[List[dict], List[dict]]:
    """將候選清單解析為 (tools, parameters) 兩份清單。

    Returns
    -------
    resolved_tools   : List[dict]  工具描述清單
    resolved_params  : List[dict]  工具參數 schema 清單
    """
    seen_tools:     set        = set()
    resolved_tools: List[dict] = []
    resolved_params: List[dict] = []

    for cand in candidates:
        rc_name = cand.get("rc_name") or ""
        agent_t = resolve_agent_type(rc_name, concept_to_agent)
        if agent_t is None:
            continue

        base_tools = AGENTS_REGISTRY.get(agent_t, {}).get("tools", [])
        all_tools  = collect_tools_with_deps(base_tools)

        for tname in all_tools:
            if tname in seen_tools:
                continue
            seen_tools.add(tname)

            entry = TOOLS_REGISTRY.get(tname)
            if entry is None:
                continue

            is_dep = tname not in base_tools
            resolved_tools.append({
                "tool_name":       tname,
                "description":     entry["description"],
                "skills":          entry["skills"],
                "from_agent":      agent_t.value,
                "matched_concept": rc_name,
                "is_dependency":   is_dep,
            })
            if entry.get("parameters"):
                resolved_params.append({
                    "tool_name":  tname,
                    "parameters": entry["parameters"],
                })

    return resolved_tools, resolved_params
