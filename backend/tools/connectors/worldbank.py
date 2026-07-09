# -*- coding: utf-8 -*-
"""
tools/connectors/worldbank.py — World Bank Open Data 單一指標查詢 connector。

retriever_id 格式：worldbank.{indicator_code}，
例如 worldbank.NY.GDP.MKTP.KD.ZG（GDP 成長率）、
     worldbank.FP.CPI.TOTL.ZG（CPI 年增率）。

必要 retriever_kwargs：
  - country : ISO 國碼，例如 "TW"、"US"、"EMU"（歐元區）。
              World Bank 依國家分資料，這裡不猜測，未提供時直接失敗。

選用 retriever_kwargs：
  - mrv : 取最近幾筆資料嘗試找非缺值（預設 5）。

不需要 API key（World Bank Open Data 為公開 API）。
"""

from __future__ import annotations

from typing import Any, Dict

from tools.connectors.base import error_result, success_result


def fetch(series_id: str, **kwargs: Any) -> Dict[str, Any]:
    country = kwargs.pop("country", None)
    if not country:
        return error_result(
            series_id,
            "缺少必要參數 country（ISO 國碼），World Bank connector 不猜測國家",
            source="WorldBank",
        )

    mrv = kwargs.pop("mrv", 5)

    try:
        import requests
    except ImportError:
        return error_result(series_id, "未安裝 requests 套件", source="WorldBank")

    try:
        url = (
            f"https://api.worldbank.org/v2/country/{country}"
            f"/indicator/{series_id}?format=json&mrv={mrv}&per_page=5"
        )
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return error_result(series_id, f"HTTP {resp.status_code}", source="WorldBank")

        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            return error_result(series_id, "查無資料", source="WorldBank")

        for record in payload[1]:
            if record.get("value") is not None:
                return success_result(
                    value=round(float(record["value"]), 4),
                    source="WorldBank",
                    as_of_date=record.get("date", "?"),
                    series_id=series_id,
                )

        return error_result(series_id, f"近 {mrv} 筆資料皆為缺值", source="WorldBank")
    except Exception as exc:
        return error_result(series_id, exc, source="WorldBank")
