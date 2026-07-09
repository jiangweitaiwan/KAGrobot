# -*- coding: utf-8 -*-
"""
tools/connectors/oecd.py — OECD Statistics（SDMX-JSON）單一序列查詢 connector。

retriever_id 格式：oecd.{series}，例如 oecd.IR3TIB（短期利率）。

必要 retriever_kwargs：
  - country : OECD 3 碼國別代碼，例如 "TWN"、"USA"、"EA19"（歐元區）。
              不猜測，未提供時直接失敗。

選用 retriever_kwargs：
  - dataset               : SDMX dataset 代碼（預設 "MEI_FIN"）。
  - freq                  : 頻率代碼（預設 "M"，月頻）。
  - start_time            : 起始期間，如 "2024-01"（預設 "2024-01"）。
  - last_n_observations   : 取最近幾筆觀測值嘗試找非缺值（預設 6）。

不需要 API key（OECD Statistics 為公開 API）。
"""

from __future__ import annotations

from typing import Any, Dict

from tools.connectors.base import error_result, success_result


def fetch(series_id: str, **kwargs: Any) -> Dict[str, Any]:
    country = kwargs.pop("country", None)
    if not country:
        return error_result(
            series_id,
            "缺少必要參數 country（OECD 3 碼國別代碼），OECD connector 不猜測國家",
            source="OECD",
        )

    dataset             = kwargs.pop("dataset", "MEI_FIN")
    freq                = kwargs.pop("freq", "M")
    start_time          = kwargs.pop("start_time", "2024-01")
    last_n_observations = kwargs.pop("last_n_observations", 6)

    try:
        import requests
    except ImportError:
        return error_result(series_id, "未安裝 requests 套件", source="OECD")

    try:
        url = (
            f"https://stats.oecd.org/SDMX-JSON/data/{dataset}/"
            f"{series_id}.{country}.{freq}/all?startTime={start_time}"
            f"&lastNObservations={last_n_observations}&contentType=json"
        )
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return error_result(series_id, f"HTTP {resp.status_code}", source="OECD")

        body   = resp.json()
        series = body.get("dataSets", [{}])[0].get("series", {})
        if not series:
            return error_result(series_id, "查無資料", source="OECD")

        series_key = list(series.keys())[0]
        values     = series[series_key].get("observations", {})

        for k in sorted(values.keys(), key=int, reverse=True):
            v = values[k][0]
            if v is not None:
                return success_result(
                    value=round(float(v), 4),
                    source=f"OECD/{dataset}",
                    as_of_date=k,
                    series_id=series_id,
                )

        return error_result(series_id, "近期觀測值皆為缺值", source="OECD")
    except Exception as exc:
        return error_result(series_id, exc, source="OECD")
