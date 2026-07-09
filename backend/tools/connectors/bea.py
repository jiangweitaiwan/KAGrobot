# -*- coding: utf-8 -*-
"""
tools/connectors/bea.py — BEA（Bureau of Economic Analysis）單一序列查詢 connector。

retriever_id 格式：bea.{table_name}，例如 bea.T10101（NIPA Table 1.1.1）。

必要 retriever_kwargs：
  - line_number : BEA 表格內的行號（例如 GDP 通常是 "1"）。
                  BEA 一張表格含多行不同定義的數據，本 connector
                  不嘗試猜測或回傳整張表格——寫 params_manifest.yaml
                  時務必先去 BEA 官網或內部文件查清楚對應的 line_number。

選用 retriever_kwargs：
  - frequency  : "Q"（季）或 "A"（年），預設 "Q"。
  - year       : 查詢年份範圍，預設 "LAST5"（BEA API 慣例值，近 5 筆）。
  - dataset    : BEA DataSetName，預設 "NIPA"。
  - api_key    : 覆寫使用的 BEA API key（一般不需要，
                 預設讀 settings.bea_api_key / 環境變數 BEA_API_KEY）。

需要 BEA_API_KEY；未設定時回傳 success=False，
呼叫方（resolve_params）會自動 fallback 到 calibrated_default。
"""

from __future__ import annotations

from typing import Any, Dict

from tools.connectors.base import error_result, get_setting, success_result


def fetch(series_id: str, **kwargs: Any) -> Dict[str, Any]:
    line_number = kwargs.pop("line_number", None)
    if not line_number:
        return error_result(
            series_id,
            "缺少必要參數 line_number，BEA connector 不猜測要回傳哪一行數據",
            source="BEA",
        )

    api_key = kwargs.pop("api_key", None) or get_setting("bea_api_key", "BEA_API_KEY")
    if not api_key:
        return error_result(series_id, "缺少 BEA_API_KEY，請於 .env 或環境變數設定", source="BEA")

    frequency = kwargs.pop("frequency", "Q")
    year      = kwargs.pop("year", "LAST5")
    dataset   = kwargs.pop("dataset", "NIPA")

    try:
        import requests
    except ImportError:
        return error_result(series_id, "未安裝 requests 套件", source="BEA")

    try:
        resp = requests.get(
            "https://apps.bea.gov/api/data",
            params={
                "UserID":       api_key,
                "method":       "GetData",
                "DataSetName":  dataset,
                "TableName":    series_id,
                "Frequency":    frequency,
                "Year":         year,
                "ResultFormat": "JSON",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return error_result(series_id, f"HTTP {resp.status_code}", source="BEA")

        data    = resp.json()
        results = data.get("BEAAPI", {}).get("Results", {})

        if isinstance(results, dict) and results.get("Error"):
            return error_result(series_id, results["Error"], source="BEA")

        rows = [
            r for r in (results.get("Data") or [])
            if str(r.get("LineNumber")) == str(line_number)
        ]
        if not rows:
            return error_result(
                series_id, f"查無 LineNumber={line_number} 的資料", source="BEA"
            )

        rows.sort(key=lambda r: r.get("TimePeriod", ""), reverse=True)
        latest     = rows[0]
        value_raw  = (latest.get("DataValue") or "").replace(",", "")
        if not value_raw:
            return error_result(series_id, "最新一筆資料為缺值", source="BEA")

        return success_result(
            value=round(float(value_raw), 4),
            source="BEA",
            as_of_date=latest.get("TimePeriod", "?"),
            series_id=series_id,
        )
    except Exception as exc:
        return error_result(series_id, exc, source="BEA")
