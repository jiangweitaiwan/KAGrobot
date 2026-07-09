# -*- coding: utf-8 -*-
"""
tools/connectors/base.py
─────────────────────────────────────────────────────────────
所有 connector（fred / worldbank / oecd / bea）共用的標準回傳格式。

設計原則（比照 data-retriever skill 的 connectors/base.py）：
  - 絕不 raise exception。任何失敗（key 未設、網路擋、序列錯誤、
    資料缺值）都回傳 success=False + error 說明，呼叫方永遠可以
    安全地檢查 result["success"] 並自行 fallback。
  - 回傳格式固定，讓 tools/data_retriever.py 與
    tools/params_resolver.py 不需要為不同來源另外處理欄位差異。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def success_result(
    *,
    value:      Any,
    source:     str,
    as_of_date: str,
    series_id:  str,
    unit:       str = "as_published",
) -> Dict[str, Any]:
    """組裝成功查詢的標準回傳格式。"""
    return {
        "success":      True,
        "value":        value,
        "unit":         unit,
        "source":       source,
        "series_id":    series_id,
        "as_of_date":   as_of_date,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "error":        None,
    }


def error_result(
    series_id: str,
    error:     Any,
    source:    str = "",
) -> Dict[str, Any]:
    """組裝失敗查詢的標準回傳格式。error 可為例外物件或字串，一律轉字串。"""
    return {
        "success":      False,
        "value":        None,
        "unit":         None,
        "source":       source,
        "series_id":    series_id,
        "as_of_date":   None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "error":        str(error),
    }


def get_setting(attr_name: str, env_name: str) -> Optional[str]:
    """統一的 API key 讀取順序：settings.py（.env） → 環境變數。

    Parameters
    ----------
    attr_name : Settings 物件上的屬性名稱，例如 "fred_api_key"
    env_name  : 環境變數名稱，例如 "FRED_API_KEY"
    """
    import os

    try:
        from settings import get_settings
        val = getattr(get_settings(), attr_name, "") or ""
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(env_name, "") or None
