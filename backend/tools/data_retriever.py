# -*- coding: utf-8 -*-
"""
tools/data_retriever.py
─────────────────────────────────────────────────────────────
統一的經濟數據檢索層，對應 dsge-model skill 依賴的
data-retriever 生態系角色，但整合進 KAGrobot 後端，
供兩種呼叫方式使用：

  1. dispatch_retrieve_param()：純 Python 函式，供
     tools/params_resolver.py 的 resolve_params 直接 import 呼叫
     （不透過額外一次 LLM tool call，省一次來回，
     也不會因為 LLM 忘記呼叫而漏掉 tool_retrieval）。



新增資料源步驟見 tools/connectors/__init__.py 的說明。
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.tools import tool

from tools.base import register_langchain_tool
from tools.connectors import bea, fred, oecd, worldbank

# retriever_id 前綴 → connector 的 fetch() 函式
_CONNECTORS = {
    "fred":      fred.fetch,
    "worldbank": worldbank.fetch,
    "oecd":      oecd.fetch,
    "bea":       bea.fetch,
}


def dispatch_retrieve_param(retriever_id: str, **kwargs: Any) -> Dict[str, Any]:
    """依 retriever_id 前綴分派給對應 connector。

    retriever_id 格式須為 "{來源}.{序列代碼}"，例如 "fred.CPIAUCSL"。
    絕不 raise exception，任何失敗都回傳 success=False + error 說明。
    """
    source, sep, series_id = retriever_id.partition(".")

    if not sep:
        return {
            "success": False, "value": None, "unit": None,
            "source": "", "series_id": retriever_id, "as_of_date": None,
            "retrieved_at": None,
            "error": f"retriever_id 格式錯誤，須為 '{{來源}}.{{序列代碼}}'：'{retriever_id}'",
        }

    fn = _CONNECTORS.get(source)
    if fn is None:
        return {
            "success": False, "value": None, "unit": None,
            "source": source, "series_id": series_id, "as_of_date": None,
            "retrieved_at": None,
            "error": f"未知資料源 '{source}'，可用：{list(_CONNECTORS.keys())}",
        }

    try:
        return fn(series_id, **kwargs)
    except Exception as exc:
        return {
            "success": False, "value": None, "unit": None,
            "source": source, "series_id": series_id, "as_of_date": None,
            "retrieved_at": None, "error": str(exc),
        }


@tool
def retrieve_param(retriever_id: str, kwargs_json: str = "{}") -> str:
    """
    查詢單一經濟數據序列的最新值，支援 FRED、World Bank、OECD、BEA 四種來源。

    retriever_id 格式為 "{來源}.{序列代碼}"：
      - fred.{series_id}       美國 FRED，例如 fred.CPIAUCSL、fred.FEDFUNDS
      - worldbank.{indicator}  World Bank，例如 worldbank.NY.GDP.MKTP.KD.ZG
                                （kwargs_json 需附 country，如 '{"country":"TW"}'）
      - oecd.{series}          OECD SDMX，例如 oecd.IR3TIB
                                （kwargs_json 需附 country，如 '{"country":"TWN"}'）
      - bea.{table_name}       BEA NIPA，例如 bea.T10101
                                （kwargs_json 需附 line_number，不猜測）

    絕不拋出例外；任何失敗（key 未設、網路擋、序列錯誤、資料缺值）
    都回傳 success=False + error 說明，可安全檢查 success 並自行 fallback。

    Args:
        retriever_id: 資料源與序列代碼，例如 'fred.CPIAUCSL'
        kwargs_json: JSON 字串，傳給對應 connector 的額外參數，
            例如 '{"country": "TW"}'。無額外參數時傳 '{}'（預設值）。

    Returns:
        JSON string，含 success / value / unit / source / series_id /
        as_of_date / retrieved_at / error。
    """
    try:
        extra_kwargs = json.loads(kwargs_json) if kwargs_json else {}
    except json.JSONDecodeError:
        return json.dumps(
            {"success": False, "error": "kwargs_json 格式錯誤，需為合法 JSON 字串"},
            ensure_ascii=False,
        )

    result = dispatch_retrieve_param(retriever_id, **extra_kwargs)
    return json.dumps(result, ensure_ascii=False, indent=2)



