# -*- coding: utf-8 -*-
"""
tools/connectors/fred.py — FRED（St. Louis Fed）單一序列查詢 connector。

retriever_id 格式：fred.{series_id}，例如 fred.CPIAUCSL、fred.FEDFUNDS。

支援的 retriever_kwargs：
  - units             : FRED 的資料轉換參數，例如 "pc1"（年增率%）、
                        "pch"（環比%）。省略則回傳原始水準值。
  - observation_start : 起始日期字串，如 "2023-01-01"（預設近一年）。
  - api_key           : 覆寫使用的 FRED API key（一般不需要，
                        預設讀 settings.fred_api_key / 環境變數 FRED_API_KEY）。

需要 FRED_API_KEY；未設定或 fredapi 套件未安裝時，回傳 success=False，
呼叫方（resolve_params）會自動 fallback 到 calibrated_default。
"""

from __future__ import annotations

from typing import Any, Dict

from tools.connectors.base import error_result, get_setting, success_result


def fetch(series_id: str, **kwargs: Any) -> Dict[str, Any]:
    api_key = kwargs.pop("api_key", None) or get_setting("fred_api_key", "FRED_API_KEY")
    if not api_key:
        return error_result(series_id, "缺少 FRED_API_KEY，請於 .env 或環境變數設定", source="FRED")

    try:
        import fredapi
    except ImportError:
        return error_result(series_id, "未安裝 fredapi 套件（pip install fredapi）", source="FRED")

    observation_start = kwargs.pop("observation_start", "2023-01-01")

    try:
        fred = fredapi.Fred(api_key=api_key)
        series = fred.get_series(
            series_id,
            observation_start=observation_start,
            **kwargs,  # 例如 units="pc1"
        )
        series = series.dropna()
        if series.empty:
            return error_result(series_id, "查無有效資料（全為缺值）", source="FRED")

        value    = float(series.iloc[-1])
        as_of    = series.index[-1]
        as_of_date = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)

        return success_result(
            value=round(value, 4),
            source="FRED",
            as_of_date=as_of_date,
            series_id=series_id,
        )
    except Exception as exc:
        return error_result(series_id, exc, source="FRED")
