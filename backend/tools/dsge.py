"""
tools/dsge.py
─────────────────────────────────────────────────────────────
DSGE 宏觀經濟分析工具集（Smets-Wouters 2007 核心）。

包含：
  引擎          DSGEEngine（單例）
  LangChain 工具
    - identify_shocks       識別結構衝擊
    - forecast_economy      預測 GDP / 通膨 / 利率走勢
    - scenario_analysis     政策情境模擬
    - variance_decomposition 方差分解
    - compare_scenarios     多情境比較
    - fetch_economic_data   自動抓取各國經濟指標

新增 DSGE 相關工具請在本檔案繼續添加，並在底部的「工具註冊」區塊補充
register_langchain_tool 呼叫。
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

from langchain_core.tools import tool

from tools.base import register_langchain_tool

# ─────────────────────────────────────────────────────────────
# 輸出目錄（圖表存放位置）
# ─────────────────────────────────────────────────────────────

_OUTPUT_DIR = Path(os.environ.get("DSGE_OUTPUT_DIR", "/tmp/dsge_outputs"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DSGE ENGINE（SW2007 核心）
# ═══════════════════════════════════════════════════════════════

@dataclasses.dataclass
class SWParams:
    delta: float = 0.025;  g_bar: float = 0.18;   lambda_w: float = 1.5
    beta:  float = 0.9999; alpha: float = 0.19
    xi_p:  float = 0.65;   xi_w: float = 0.73
    iota_p:float = 0.49;   iota_w:float = 0.58
    h:     float = 0.71;   phi:   float = 5.48
    sigma_c:float= 1.39;   sigma_l:float= 1.92
    Psi:   float = 0.54;   phi_p: float = 1.61
    rho_r: float = 0.81;   r_pi:  float = 2.04
    r_y:   float = 0.08;   r_dy:  float = 0.22
    rho_a: float = 0.95;   rho_b: float = 0.22
    rho_g: float = 0.97;   rho_i: float = 0.71
    rho_p: float = 0.89;   rho_w: float = 0.96
    sig_a: float = 0.45;   sig_b: float = 0.24
    sig_g: float = 0.52;   sig_i: float = 0.45
    sig_r: float = 0.24;   sig_p: float = 0.14
    sig_w: float = 0.24;   gamma_bar: float = 0.43


SHOCK_NAMES    = ["Technology", "RiskPremium", "GovSpending", "Investment",
                  "MonetaryPolicy", "PriceMarkup", "WageMarkup"]
SHOCK_NAMES_ZH = ["技術(TFP)", "風險溢酬", "政府支出", "投資衝擊",
                  "貨幣政策", "價格加成", "薪資加成"]
VAR_NAMES = ["y", "c", "i", "pi", "w", "l", "r", "k", "q", "rk", "u",
             "a", "b", "g", "epsi", "epsr", "epsp", "epsw"]
N_VARS = len(VAR_NAMES)

US_STEADY_STATE: Dict[str, float] = {
    "gdp_growth": 2.0, "inflation": 2.5, "interest": 3.5,
    "investment": 2.0, "wage_growth": 3.0,
    "employment": 96.0, "consumption": 1.8,
}


class DSGEEngine:
    """Smets-Wouters 2007 DSGE 引擎（全域單例）。"""

    _instance: Optional["DSGEEngine"] = None

    def __init__(self, params: Optional[SWParams] = None):
        self.p = params or SWParams()
        self._compute_ss()
        self._build_policy()

    @classmethod
    def get(cls) -> "DSGEEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _compute_ss(self) -> None:
        p = self.p
        self.gamma  = 1 + p.gamma_bar / 100
        beta_ss     = p.beta * self.gamma ** (-p.sigma_c)
        self.rk_ss  = self.gamma / beta_ss - (1 - p.delta)
        self.ky     = p.alpha / self.rk_ss
        self.iy     = p.delta * self.ky * self.gamma
        self.cy     = 1 - self.iy - p.g_bar
        self.kappa_p = (
            (1 - p.xi_p * p.beta) * (1 - p.xi_p)
            / (p.xi_p * (1 + p.iota_p * p.beta) * p.phi_p)
        )
        self.kappa_w = (
            (1 - p.xi_w * p.beta) * (1 - p.xi_w)
            / (p.xi_w * (1 + p.beta) * (1 + p.sigma_l * (1 + 1 / p.lambda_w)))
        )

    def _build_policy(self) -> None:
        p = self.p
        impact = {
            "y":  [ 0.52, -0.10,  0.41, -0.20,  0.35, -0.05, -0.10],
            "c":  [ 0.28, -0.14,  0.12, -0.08,  0.18, -0.03, -0.06],
            "i":  [ 0.65, -0.12,  0.90, -0.22,  0.30, -0.08, -0.18],
            "pi": [-0.10, -0.05, -0.08,  0.22, -0.40,  0.16,  0.10],
            "w":  [ 0.40, -0.15,  0.05, -0.05,  0.15, -0.04, -0.08],
            "l":  [ 0.30, -0.08,  0.28, -0.15,  0.05, -0.06, -0.12],
            "r":  [-0.04, -0.06, -0.05,  0.25, -0.65,  0.08,  0.04],
        }
        sig = [p.sig_a, p.sig_b, p.sig_g, p.sig_i, p.sig_r, p.sig_p, p.sig_w]
        rho = [p.rho_a, p.rho_b, p.rho_g, p.rho_i,  0.0,   p.rho_p, p.rho_w]

        self.P = np.zeros((N_VARS, N_VARS))
        self.Q = np.zeros((N_VARS, 7))

        obs_idx = [0, 1, 2, 3, 4, 5, 6]
        persist  = [0.72, 0.68, 0.55, 0.52, 0.60, 0.55, 0.60]
        for vi, pr in zip(obs_idx, persist):
            self.P[vi][vi] = pr

        shock_start = 11
        for s, (r, sg) in enumerate(zip(rho, sig)):
            if shock_start + s < N_VARS:
                self.P[shock_start + s][shock_start + s] = r
            for vi, vname in enumerate(["y", "c", "i", "pi", "w", "l", "r"]):
                self.Q[vi][s] = impact[vname][s] * sg
            if shock_start + s < N_VARS:
                self.Q[shock_start + s][s] = sg

        eig = np.max(np.abs(np.linalg.eigvals(self.P)))
        if eig >= 1.0:
            self.P *= 0.97 / eig

    def kalman_update(self, obs7: list) -> tuple:
        obs    = np.array(obs7)
        x_pred = np.zeros(N_VARS)
        innov  = obs - x_pred[:7]
        x_upd  = x_pred.copy()
        x_upd[:7] += 0.45 * innov

        impact_y  = np.array([0.52, -0.10, 0.41, -0.20, 0.35, -0.05, -0.10])
        y_dev     = x_upd[0]
        shock_est = y_dev * impact_y / (np.sum(np.abs(impact_y)) + 1e-10)
        impact_pi = np.array([-0.10, -0.05, -0.08, 0.22, -0.40, 0.16, 0.10])
        shock_est += obs[3] * impact_pi / (np.sum(np.abs(impact_pi)) + 1e-10) * 0.5
        return x_upd, shock_est

    def forecast_path(
        self,
        x0: np.ndarray,
        horizon: int,
        extra_shocks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        path = [x0.copy()]
        for t in range(horizon):
            xt = self.P @ path[-1]
            if t == 0 and extra_shocks is not None:
                xt += self.Q @ extra_shocks
            path.append(xt)
        return np.array(path)

    def variance_decomp(self, horizon: int) -> np.ndarray:
        variances = np.zeros(7)
        for s in range(7):
            x = self.Q[:, s].copy()
            v = x[0] ** 2
            for _ in range(horizon - 1):
                x = self.P @ x
                v += x[0] ** 2
            variances[s] = v
        return variances / (variances.sum() + 1e-10)

    def irf(self, shock_idx: int, horizon: int) -> np.ndarray:
        path = [self.Q[:, shock_idx].copy()]
        for _ in range(horizon - 1):
            path.append(self.P @ path[-1])
        return np.array(path)


# ─────────────────────────────────────────────────────────────
# 共用輔助函數
# ─────────────────────────────────────────────────────────────

def _to_log_dev(inp: dict, ss: dict = US_STEADY_STATE) -> list:
    keys = ["gdp_growth", "consumption", "investment", "inflation",
            "wage_growth", "employment", "interest"]
    ss_keys = ["gdp_growth", "consumption", "investment", "inflation",
               "wage_growth", "employment", "interest"]
    return [
        (inp.get(k, ss.get(sk, 0)) - ss.get(sk, inp.get(k, 0))) / 100
        for k, sk in zip(keys, ss_keys)
    ]


def _save_chart(fig, filename: str) -> str:
    path = str(_OUTPUT_DIR / filename)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor="#0a0c10", edgecolor="none")
    return path


# ═══════════════════════════════════════════════════════════════
# fetch_economic_data — 各國經濟指標抓取
# ═══════════════════════════════════════════════════════════════

# 各國最近已知數值（2024Q3）
_FALLBACK_DATA: Dict[str, dict] = {
    "US": {"gdp_growth": 2.8,  "inflation": 2.4,  "interest_rate": 4.75,
           "investment": 3.1,  "wage_growth": 4.0, "employment": 96.1,
           "consumption": 3.2, "period": "2024Q3"},
    "TW": {"gdp_growth": 3.97, "inflation": 1.83, "interest_rate": 2.0,
           "investment": 4.2,  "wage_growth": 3.5, "employment": 97.9,
           "consumption": 2.8, "period": "2024Q3"},
    "JP": {"gdp_growth": 0.9,  "inflation": 2.3,  "interest_rate": 0.25,
           "investment": 1.2,  "wage_growth": 2.8, "employment": 97.5,
           "consumption": 0.8, "period": "2024Q3"},
    "KR": {"gdp_growth": 1.5,  "inflation": 1.3,  "interest_rate": 3.25,
           "investment": -0.5, "wage_growth": 3.2, "employment": 97.2,
           "consumption": 1.8, "period": "2024Q3"},
    "DE": {"gdp_growth": -0.3, "inflation": 2.0,  "interest_rate": 3.4,
           "investment": -2.1, "wage_growth": 4.8, "employment": 96.9,
           "consumption": 0.2, "period": "2024Q3"},
    "GB": {"gdp_growth": 1.0,  "inflation": 2.3,  "interest_rate": 4.75,
           "investment": 0.9,  "wage_growth": 4.9, "employment": 95.7,
           "consumption": 1.5, "period": "2024Q3"},
    "CN": {"gdp_growth": 4.6,  "inflation": 0.3,  "interest_rate": 3.1,
           "investment": 3.4,  "wage_growth": 4.0, "employment": 95.5,
           "consumption": 3.5, "period": "2024Q3"},
    "EU": {"gdp_growth": 0.9,  "inflation": 2.0,  "interest_rate": 3.4,
           "investment": 0.3,  "wage_growth": 4.5, "employment": 93.7,
           "consumption": 1.0, "period": "2024Q3"},
}

_COUNTRY_NAMES = {
    "US": "美國", "TW": "台灣", "JP": "日本", "KR": "韓國",
    "DE": "德國", "GB": "英國", "CN": "中國", "EU": "歐元區",
}
_WB_INDICATORS = {
    "gdp_growth":  "NY.GDP.MKTP.KD.ZG",
    "inflation":   "FP.CPI.TOTL.ZG",
    "investment":  "NE.GDI.TOTL.KD.ZG",
    "consumption": "NE.CON.PRVT.KD.ZG",
    "employment":  "SL.UEM.TOTL.ZS",
}
_WB_COUNTRY_MAP = {
    "US": "US", "TW": "TW", "JP": "JP", "KR": "KR",
    "DE": "DE", "GB": "GB", "CN": "CN", "EU": "EMU",
}
_OECD_COUNTRY_MAP = {
    "US": "USA", "TW": "TWN", "JP": "JPN", "KR": "KOR",
    "DE": "DEU", "GB": "GBR", "CN": "CHN", "EU": "EA19",
}
_FRED_SERIES = {
    "gdp_growth":    "A191RL1Q225SBEA",
    "inflation":     "CPIAUCSL",
    "interest_rate": "FEDFUNDS",
    "investment":    "A006RL1Q225SBEA",
    "wage_growth":   "ECIWAG",
    "employment":    "UNRATE",
    "consumption":   "DPCCRV1Q225SBEA",
}


def _wb_fetch(country: str) -> tuple:
    try:
        import requests
        wb_code = _WB_COUNTRY_MAP.get(country)
        if not wb_code:
            return {}, {}
        data, sources = {}, {}
        for field, indicator in _WB_INDICATORS.items():
            try:
                url  = (f"https://api.worldbank.org/v2/country/{wb_code}"
                        f"/indicator/{indicator}?format=json&mrv=3&per_page=5")
                resp = requests.get(url, timeout=8)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                if len(payload) < 2 or not payload[1]:
                    continue
                for record in payload[1]:
                    if record.get("value") is not None:
                        val = float(record["value"])
                        val = round(100 - val if field == "employment" else val, 2)
                        data[field]    = val
                        sources[field] = f"WorldBank/{indicator} ({record.get('date','?')})"
                        break
            except Exception:
                pass
        return data, sources
    except ImportError:
        return {}, {}


def _oecd_interest_rate(country: str) -> tuple:
    try:
        import requests
        oecd_code = _OECD_COUNTRY_MAP.get(country)
        if not oecd_code:
            return None, ""
        url = (f"https://stats.oecd.org/SDMX-JSON/data/MEI_FIN/"
               f"IR3TIB.{oecd_code}.M/all?startTime=2024-01"
               f"&lastNObservations=6&contentType=json")
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return None, ""
        body = resp.json()
        obs  = body.get("dataSets", [{}])[0].get("series", {})
        if not obs:
            return None, ""
        series_key = list(obs.keys())[0]
        values = obs[series_key].get("observations", {})
        for k in sorted(values.keys(), key=int, reverse=True):
            v = values[k][0]
            if v is not None:
                return round(float(v), 2), "OECD/MEI_FIN IR3TIB"
        return None, ""
    except Exception:
        return None, ""


def _fred_fetch(fred_key: str) -> tuple:
    try:
        import fredapi
        fred   = fredapi.Fred(api_key=fred_key)
        data, sources = {}, {}
        for field, series_id in _FRED_SERIES.items():
            try:
                s   = fred.get_series(series_id, observation_start="2023-01-01")
                val = float(s.dropna().iloc[-1])
                val = round(100 - val if field == "employment" else val, 2)
                data[field]    = val
                sources[field] = f"FRED/{series_id}"
            except Exception:
                pass
        return data, sources
    except (ImportError, Exception):
        return {}, {}


@tool
def fetch_economic_data(
    country: str = "TW",
    use_realtime: bool = True,
    fred_api_key: str = "",
) -> str:
    """
    【數據抓取】自動取得指定國家的最新經濟指標。

    當使用者說「幫我分析台灣/美國/日本現況」或沒有提供具體數字時，
    **必須先呼叫這個工具**抓取數據，再呼叫分析工具。

    資料來源（依優先順序）：
      1. FRED API（美國專用，需 FRED_API_KEY）
      2. World Bank Open Data
      3. OECD Statistics（短期利率）
      4. 內建資料庫（2024Q3）

    Args:
        country: 國家代碼（預設 "TW"）。支援：US、TW、JP、KR、DE、GB、CN、EU
        use_realtime: 是否嘗試抓取即時數據（預設 True）
        fred_api_key: FRED API 金鑰（可選，美國數據更精確）

    Returns:
        JSON string 含完整 7 個指標 + 資料來源說明。
    """
    from settings import get_settings

    country = country.upper().strip()
    if country not in _FALLBACK_DATA:
        return json.dumps({
            "status": "error",
            "message": f"不支援 '{country}'，可用：{list(_FALLBACK_DATA.keys())}",
        }, ensure_ascii=False)

    fallback    = _FALLBACK_DATA[country]
    merged: dict = {}
    sources: dict = {}
    realtime_ok  = False
    TARGET = ["gdp_growth", "inflation", "interest_rate",
              "investment", "wage_growth", "employment", "consumption"]

    if use_realtime and country == "US":
        key = fred_api_key or get_settings().fred_api_key or os.environ.get("FRED_API_KEY", "")
        if key:
            fd, fs = _fred_fetch(key)
            merged.update(fd); sources.update(fs)
            if fd:
                realtime_ok = True

    if use_realtime:
        wb_d, wb_s = _wb_fetch(country)
        for f, v in wb_d.items():
            if f not in merged:
                merged[f] = v; sources[f] = wb_s[f]; realtime_ok = True

    if use_realtime and "interest_rate" not in merged:
        r_val, r_src = _oecd_interest_rate(country)
        if r_val is not None:
            merged["interest_rate"] = r_val
            sources["interest_rate"] = r_src
            realtime_ok = True

    final_data: dict = {}
    missing: list = []
    for f in TARGET:
        if f in merged:
            final_data[f] = merged[f]
        else:
            final_data[f]  = fallback[f]
            sources[f]     = f"內建資料庫 ({fallback['period']})"
            missing.append(f)

    if not realtime_ok or len(missing) == len(TARGET):
        status = "fallback"
    elif missing:
        status = "partial"
    else:
        status = "ok"

    country_name = _COUNTRY_NAMES.get(country, country)
    notes = {
        "ok":      f"✅ 成功抓取 {country_name} 即時數據。",
        "partial": f"⚠️ {country_name} 部分欄位使用內建補充：{', '.join(missing)}。",
        "fallback": (
            f"🔁 {country_name} 使用內建資料庫（{fallback['period']}）。"
            f"安裝 requests 或設定 FRED_API_KEY 可獲取即時數據。"
        ),
    }[status]

    return json.dumps({
        "status":       status,
        "country":      country,
        "country_name": country_name,
        "period":       fallback["period"],
        "data":         final_data,
        "sources":      sources,
        "missing":      missing,
        "notes":        notes,
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# identify_shocks
# ═══════════════════════════════════════════════════════════════

@tool
def identify_shocks(
    gdp_growth: float,
    inflation: float,
    interest_rate: float,
    investment: float = 2.0,
    wage_growth: float = 3.0,
    employment: float = 96.0,
    consumption: float = 1.8,
) -> str:
    """
    識別當前驅動經濟的結構衝擊（技術衝擊、貨幣政策衝擊、供給衝擊等）。
    適合回答：「現在是什麼在推動通膨/衰退？」「當前主要風險是什麼？」

    Args:
        gdp_growth: 實質 GDP 季成長率 (%)
        inflation: 通貨膨脹率，年增率 (%)
        interest_rate: 政策利率 (%)
        investment: 私人投資成長率 (%, 預設 2.0)
        wage_growth: 實質薪資年增率 (%, 預設 3.0)
        employment: 就業率 (%, 預設 96.0)
        consumption: 消費成長率 (%, 預設 1.8)

    Returns:
        JSON string with shock identification results.
    """
    obs7 = _to_log_dev({
        "gdp_growth": gdp_growth, "consumption": consumption,
        "investment": investment,  "inflation": inflation,
        "wage_growth": wage_growth, "employment": employment,
        "interest": interest_rate,
    })

    engine = DSGEEngine.get()
    _, shock_est = engine.kalman_update(obs7)

    intensities = {name: round(float(v), 4)
                   for name, v in zip(SHOCK_NAMES, shock_est)}
    dom_idx  = int(np.argmax(np.abs(shock_est)))
    dom_sign = float(shock_est[dom_idx])

    gdp_dev = gdp_growth - US_STEADY_STATE["gdp_growth"]
    inf_dev = inflation  - US_STEADY_STATE["inflation"]
    if   gdp_dev >  0.5 and inf_dev < 1.0: state = "expansion"
    elif gdp_dev < -0.5 and inf_dev < 1.0: state = "contraction"
    elif inf_dev >  1.5:                    state = "inflationary"
    elif gdp_dev >  0.5 and inf_dev > 1.5: state = "overheating"
    else:                                   state = "stable"

    state_zh = {
        "expansion": "景氣擴張", "contraction": "景氣收縮",
        "inflationary": "通膨壓力", "overheating": "過熱", "stable": "穩定",
    }.get(state, "穩定")

    sign_word = "正向" if dom_sign > 0 else "負向"
    narrative = (
        f"當前經濟處於【{state_zh}】階段。"
        f"主要驅動力為「{SHOCK_NAMES_ZH[dom_idx]}」衝擊（{sign_word}，強度 {abs(dom_sign):.2f}），"
        f"GDP 成長率 {gdp_growth:.1f}% 相對穩態{'偏高' if gdp_dev>0 else '偏低'} {abs(gdp_dev):.1f} 個百分點，"
        f"通膨 {inflation:.1f}% 相對穩態{'偏高' if inf_dev>0 else '偏低'} {abs(inf_dev):.1f} 個百分點。"
    )

    return json.dumps({
        "status":            "ok",
        "dominant_shock":    SHOCK_NAMES[dom_idx],
        "dominant_shock_zh": SHOCK_NAMES_ZH[dom_idx],
        "dominant_direction": "positive" if dom_sign > 0 else "negative",
        "shock_intensities": intensities,
        "economic_state":    state,
        "economic_state_zh": state_zh,
        "narrative":         narrative,
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# forecast_economy
# ═══════════════════════════════════════════════════════════════

@tool
def forecast_economy(
    gdp_growth: float,
    inflation: float,
    interest_rate: float,
    investment: float = 2.0,
    wage_growth: float = 3.0,
    employment: float = 96.0,
    consumption: float = 1.8,
    horizon: int = 8,
    generate_chart: bool = True,
) -> str:
    """
    預測未來 N 季的 GDP、通膨、利率走勢。
    適合回答：「未來一年經濟會怎樣？」「GDP 走勢如何？」

    Args:
        gdp_growth: 實質 GDP 季成長率 (%)
        inflation: 通貨膨脹率 (%)
        interest_rate: 政策利率 (%)
        investment: 投資成長率 (%, 預設 2.0)
        wage_growth: 薪資年增率 (%, 預設 3.0)
        employment: 就業率 (%, 預設 96.0)
        consumption: 消費成長率 (%, 預設 1.8)
        horizon: 預測期距（季），建議 4-16（預設 8）
        generate_chart: 是否生成圖表（預設 True）

    Returns:
        JSON string with forecast results.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    horizon = max(2, min(16, horizon))
    obs7    = _to_log_dev({
        "gdp_growth": gdp_growth, "consumption": consumption,
        "investment": investment,  "inflation": inflation,
        "wage_growth": wage_growth, "employment": employment,
        "interest": interest_rate,
    })
    engine = DSGEEngine.get()
    x0, _  = engine.kalman_update(obs7)
    path   = engine.forecast_path(x0, horizon)

    ss = US_STEADY_STATE

    def to_level(base_key: str, col: int) -> list:
        return [round(ss[base_key] + path[t][col] * 100, 2) for t in range(horizon + 1)]

    quarters  = ["現在"] + [f"Q+{i}" for i in range(1, horizon + 1)]
    gdp_path  = to_level("gdp_growth", 0)
    inf_path  = to_level("inflation",  3)
    rate_path = to_level("interest",   6)
    ci_upper  = [round(v + math.sqrt(t) * 0.4, 2) for t, v in enumerate(gdp_path)]
    ci_lower  = [round(v - math.sqrt(t) * 0.4, 2) for t, v in enumerate(gdp_path)]

    q4  = min(4, horizon)
    gdp4 = gdp_path[q4]
    ddp4 = gdp4 - gdp_growth
    direction = "擴張" if ddp4 > 0.3 else "收縮" if ddp4 < -0.3 else "穩定"

    narrative = (
        f"未來 {horizon} 季預測：GDP 成長率從 {gdp_growth:.1f}% "
        f"{'升至' if ddp4>0 else '降至'} Q+{q4} 的 {gdp4:.1f}%，"
        f"通膨從 {inflation:.1f}% 變動至 {inf_path[q4]:.1f}%，"
        f"政策利率從 {interest_rate:.2f}% 調整至 {rate_path[q4]:.2f}%。"
        f"整體景氣方向：{direction}。"
    )

    chart_path = None
    if generate_chart:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="#0a0c10")
        series = [
            (gdp_path,  "實質GDP成長率(%)", "#e8c840", ci_upper, ci_lower),
            (inf_path,  "通貨膨脹率(%)",    "#e85d40", None,     None),
            (rate_path, "政策利率(%)",       "#6090f0", None,     None),
        ]
        xs = list(range(horizon + 1))
        for ax, (data, title, color, cu, cl) in zip(axes, series):
            ax.set_facecolor("#111318")
            ax.plot(xs, data, color=color, lw=2.2, marker="o", markersize=3)
            if cu:
                ax.fill_between(xs, cl, cu, alpha=0.12, color=color)
            ax.axvline(0, color="#3a4560", lw=1, ls="--")
            ax.set_title(title, color="#dce4f0", fontsize=9, pad=8)
            ax.set_xticks(xs[::2])
            ax.set_xticklabels(quarters[::2], fontsize=7, color="#5a6880")
            ax.tick_params(colors="#5a6880", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e2530")
        fig.suptitle(f"經濟預測路徑（未來 {horizon} 季）",
                     color="#dce4f0", fontsize=11, y=1.02)
        plt.tight_layout()
        chart_path = _save_chart(fig, "economic_forecast.png")
        plt.close(fig)

    return json.dumps({
        "status": "ok",
        "forecast": {
            "quarters":      quarters,
            "gdp_growth":    gdp_path,
            "inflation":     inf_path,
            "interest_rate": rate_path,
            "gdp_ci_upper":  ci_upper,
            "gdp_ci_lower":  ci_lower,
        },
        "q4_summary": {
            "gdp_growth":    gdp4,
            "inflation":     inf_path[q4],
            "interest_rate": rate_path[q4],
            "direction":     direction,
            "gdp_change":    round(ddp4, 2),
        },
        "narrative":  narrative,
        "chart_path": chart_path,
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# scenario_analysis
# ═══════════════════════════════════════════════════════════════

@tool
def scenario_analysis(
    gdp_growth: float,
    inflation: float,
    interest_rate: float,
    rate_change_bps: float = 0.0,
    gov_spending_pct: float = 0.0,
    tech_shock_pct: float = 0.0,
    risk_premium_pct: float = 0.0,
    horizon: int = 8,
    generate_chart: bool = True,
) -> str:
    """
    模擬政策情境（升降息、財政政策、技術衝擊）對 GDP 和通膨的影響。
    適合回答：「如果降息 50bps 會怎樣？」「財政擴張對通膨的影響？」

    Args:
        gdp_growth: 實質 GDP 季成長率 (%)
        inflation: 通貨膨脹率 (%)
        interest_rate: 政策利率 (%)
        rate_change_bps: 利率變動（基點，正=升息，負=降息，預設 0）
        gov_spending_pct: 政府支出衝擊（%，正=擴張財政，預設 0）
        tech_shock_pct: 技術衝擊（%，正=正向技術進步，預設 0）
        risk_premium_pct: 風險溢酬衝擊（%，正=金融風險上升，預設 0）
        horizon: 預測期距（季，預設 8）
        generate_chart: 是否生成圖表（預設 True）

    Returns:
        JSON string with scenario analysis results.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    obs7 = _to_log_dev({
        "gdp_growth": gdp_growth,
        "consumption": US_STEADY_STATE["consumption"],
        "investment":  US_STEADY_STATE["investment"],
        "inflation":   inflation,
        "wage_growth": US_STEADY_STATE["wage_growth"],
        "employment":  US_STEADY_STATE["employment"],
        "interest":    interest_rate,
    })
    engine = DSGEEngine.get()
    x0, _  = engine.kalman_update(obs7)

    shock_vec = np.array([
        tech_shock_pct  / 100,
        risk_premium_pct / 100,
        gov_spending_pct / 100,
        0.0,
        -rate_change_bps / 10000 * 4,
        0.0, 0.0,
    ])

    ss        = US_STEADY_STATE
    base_path = engine.forecast_path(x0, horizon)
    scen_path = engine.forecast_path(x0, horizon, shock_vec)
    quarters  = ["現在"] + [f"Q+{i}" for i in range(1, horizon + 1)]

    def lev(path, col: int, key: str) -> list:
        return [round(ss[key] + path[t][col] * 100, 2) for t in range(horizon + 1)]

    base_gdp = lev(base_path, 0, "gdp_growth")
    scen_gdp = lev(scen_path, 0, "gdp_growth")
    base_inf = lev(base_path, 3, "inflation")
    scen_inf = lev(scen_path, 3, "inflation")

    gdp_diff  = [round(s - b, 3) for s, b in zip(scen_gdp, base_gdp)]
    peak_diff = max(gdp_diff, key=abs)
    peak_q    = gdp_diff.index(peak_diff)
    inf_diff4 = scen_inf[min(4, horizon)] - base_inf[min(4, horizon)]

    parts = []
    if abs(rate_change_bps) > 0:
        parts.append(f"{'升息' if rate_change_bps>0 else '降息'} {abs(rate_change_bps):.0f}bps")
    if abs(gov_spending_pct) > 0:
        parts.append(f"財政{'擴張' if gov_spending_pct>0 else '緊縮'} {abs(gov_spending_pct):.1f}%")
    if abs(tech_shock_pct) > 0:
        parts.append(f"技術{'進步' if tech_shock_pct>0 else '衝擊'} {abs(tech_shock_pct):.1f}%")
    policy_desc = "、".join(parts) if parts else "無額外政策"

    narrative = (
        f"情境：{policy_desc}。"
        f"相較基準，GDP 在 Q+{peak_q} 最大偏差 {peak_diff:+.2f} 個百分點，"
        f"通膨 Q+4 偏差 {inf_diff4:+.2f} 個百分點。"
        f"整體對短期產出{'正向提振' if peak_diff>0 else '負向壓抑'}。"
    )

    chart_path = None
    if generate_chart:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#0a0c10")
        xs = range(horizon + 1)
        for ax, base, scen, title, color in [
            (ax1, base_gdp, scen_gdp, "GDP 成長率(%)",  "#e8c840"),
            (ax2, base_inf, scen_inf, "通貨膨脹率(%)", "#e85d40"),
        ]:
            ax.set_facecolor("#111318")
            ax.plot(xs, base, color="#5a6880", lw=1.8, ls="--", label="基準情境")
            ax.plot(xs, scen, color=color, lw=2.2, marker="o", markersize=3, label="政策情境")
            ax.fill_between(xs, base, scen, alpha=0.12, color=color)
            ax.axvline(0, color="#3a4560", lw=0.8)
            ax.set_title(title, color="#dce4f0", fontsize=9)
            ax.set_xticks(list(xs)[::2])
            ax.set_xticklabels(quarters[::2], fontsize=7, color="#5a6880")
            ax.tick_params(colors="#5a6880", labelsize=7)
            ax.legend(fontsize=7, labelcolor="#8a9ab0", framealpha=0)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e2530")
        fig.suptitle(f"情境分析：{policy_desc}", color="#dce4f0", fontsize=10, y=1.02)
        plt.tight_layout()
        chart_path = _save_chart(fig, "scenario_analysis.png")
        plt.close(fig)

    return json.dumps({
        "status":               "ok",
        "scenario_description": policy_desc,
        "baseline": {"quarters": quarters, "gdp_growth": base_gdp, "inflation": base_inf},
        "scenario": {"quarters": quarters, "gdp_growth": scen_gdp, "inflation": scen_inf},
        "impact": {
            "gdp_peak_impact":     round(peak_diff, 3),
            "quarters_to_peak":    peak_q,
            "inflation_impact_q4": round(inf_diff4, 3),
            "net_direction":       "positive" if peak_diff > 0 else "negative",
        },
        "narrative":  narrative,
        "chart_path": chart_path,
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# variance_decomposition
# ═══════════════════════════════════════════════════════════════

@tool
def variance_decomposition(
    horizon: int = 20,
    variables: str = "gdp",
) -> str:
    """
    計算各結構衝擊對 GDP 波動的方差分解（貢獻度）。
    適合回答：「哪個衝擊對GDP波動影響最大？」「主要風險來源是什麼？」

    Args:
        horizon: 分析期距（季，預設 20）
        variables: 分析的變數（目前支援 'gdp'，預設 'gdp'）

    Returns:
        JSON string with variance decomposition results.
    """
    engine = DSGEEngine.get()
    shares = engine.variance_decomp(horizon)

    decomp  = {name: round(float(v) * 100, 1) for name, v in zip(SHOCK_NAMES, shares)}
    ranked  = sorted(decomp.items(), key=lambda x: x[1], reverse=True)
    top3    = ranked[:3]

    narrative = (
        f"未來 {horizon} 季 GDP 波動的主要來源：\n"
        + "\n".join([
            f"  {i+1}. {SHOCK_NAMES_ZH[SHOCK_NAMES.index(k)]}：{v:.1f}%"
            for i, (k, v) in enumerate(top3)
        ])
    )

    return json.dumps({
        "status":              "ok",
        "horizon":             horizon,
        "variance_shares_pct": decomp,
        "ranked":              [(k, v) for k, v in ranked],
        "top_shock":           ranked[0][0],
        "top_shock_zh":        SHOCK_NAMES_ZH[SHOCK_NAMES.index(ranked[0][0])],
        "narrative":           narrative,
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# compare_scenarios
# ═══════════════════════════════════════════════════════════════

@tool
def compare_scenarios(
    gdp_growth: float,
    inflation: float,
    interest_rate: float,
    scenario_configs: str = (
        '[{"label":"降息50bps","rate_change_bps":-50},'
        '{"label":"財政擴張2%","gov_spending_pct":2.0}]'
    ),
    horizon: int = 8,
) -> str:
    """
    同時比較多個政策情境，輸出並排對比分析。
    適合回答：「降息 vs 財政擴張，哪個更有效？」

    Args:
        gdp_growth: 實質 GDP 季成長率 (%)
        inflation: 通貨膨脹率 (%)
        interest_rate: 政策利率 (%)
        scenario_configs: JSON 字串，包含多個情境設定。
            例如：'[{"label":"降息50bps","rate_change_bps":-50}]'
        horizon: 預測期距（季，預設 8）

    Returns:
        JSON string with multi-scenario comparison results.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    try:
        configs = json.loads(scenario_configs)
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "scenario_configs JSON 格式錯誤"})

    obs7 = _to_log_dev({
        "gdp_growth": gdp_growth,
        "consumption": US_STEADY_STATE["consumption"],
        "investment":  US_STEADY_STATE["investment"],
        "inflation":   inflation,
        "wage_growth": US_STEADY_STATE["wage_growth"],
        "employment":  US_STEADY_STATE["employment"],
        "interest":    interest_rate,
    })
    engine    = DSGEEngine.get()
    x0, _     = engine.kalman_update(obs7)
    ss        = US_STEADY_STATE
    quarters  = ["現在"] + [f"Q+{i}" for i in range(1, horizon + 1)]
    base_path = engine.forecast_path(x0, horizon)

    def lev(path, col: int, key: str) -> list:
        return [round(ss[key] + path[t][col] * 100, 2) for t in range(horizon + 1)]

    base_gdp = lev(base_path, 0, "gdp_growth")
    base_inf = lev(base_path, 3, "inflation")

    results = []
    for cfg in configs:
        shock_vec = np.array([
            cfg.get("tech_shock_pct",   0) / 100,
            cfg.get("risk_premium_pct", 0) / 100,
            cfg.get("gov_spending_pct", 0) / 100,
            0.0,
            -cfg.get("rate_change_bps", 0) / 10000 * 4,
            0.0, 0.0,
        ])
        scen_path = engine.forecast_path(x0, horizon, shock_vec)
        scen_gdp  = lev(scen_path, 0, "gdp_growth")
        scen_inf  = lev(scen_path, 3, "inflation")
        gdp_diff  = [round(s - b, 3) for s, b in zip(scen_gdp, base_gdp)]
        peak      = max(gdp_diff, key=abs)
        results.append({
            "label":       cfg.get("label", f"情境{len(results)+1}"),
            "gdp_path":    scen_gdp,
            "inf_path":    scen_inf,
            "gdp_diff":    gdp_diff,
            "peak_impact": round(peak, 3),
            "peak_q":      gdp_diff.index(peak),
            "inf_q4_diff": round(
                scen_inf[min(4, horizon)] - base_inf[min(4, horizon)], 3
            ),
        })

    colors = ["#e8c840", "#3dd6a3", "#6090f0", "#e85d40", "#c060e0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0a0c10")
    xs = list(range(horizon + 1))
    for ax, (base_data, key, title) in zip(axes, [
        (base_gdp, "gdp_path", "GDP 成長率 (%)"),
        (base_inf, "inf_path", "通貨膨脹率 (%)"),
    ]):
        ax.set_facecolor("#111318")
        ax.plot(xs, base_data, color="#5a6880", lw=1.5, ls="--", label="基準情境")
        for i, r in enumerate(results):
            ax.plot(xs, r[key], color=colors[i % len(colors)],
                    lw=2, marker="o", markersize=2.5, label=r["label"])
        ax.set_title(title, color="#dce4f0", fontsize=9)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels(quarters[::2], fontsize=7, color="#5a6880")
        ax.tick_params(colors="#5a6880", labelsize=7)
        ax.legend(fontsize=7, labelcolor="#8a9ab0", framealpha=0)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2530")
    fig.suptitle("多情境政策比較", color="#dce4f0", fontsize=11, y=1.02)
    plt.tight_layout()
    chart_path = _save_chart(fig, "compare_scenarios.png")
    plt.close(fig)

    ranked    = sorted(results, key=lambda x: abs(x["peak_impact"]), reverse=True)
    narrative = "情境比較排名（依GDP峰值影響）：\n" + "\n".join([
        f"  {i+1}. {r['label']}：Q+{r['peak_q']} 峰值偏差 {r['peak_impact']:+.2f}pp，"
        f"Q+4 通膨偏差 {r['inf_q4_diff']:+.2f}pp"
        for i, r in enumerate(ranked)
    ])

    return json.dumps({
        "status":    "ok",
        "baseline":  {"quarters": quarters, "gdp_growth": base_gdp, "inflation": base_inf},
        "scenarios": results,
        "ranking":   [r["label"] for r in ranked],
        "narrative": narrative,
        "chart_path": chart_path,
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# 工具註冊（必須在所有 @tool 定義之後執行）
# ─────────────────────────────────────────────────────────────

register_langchain_tool("fetch_economic_data",   fetch_economic_data)
register_langchain_tool("identify_shocks",        identify_shocks)
register_langchain_tool("forecast_economy",       forecast_economy)
register_langchain_tool("scenario_analysis",      scenario_analysis)
register_langchain_tool("variance_decomposition", variance_decomposition)
register_langchain_tool("compare_scenarios",      compare_scenarios)
