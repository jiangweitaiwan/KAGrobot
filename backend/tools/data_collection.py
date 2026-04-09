"""
tools/data_collection.py
─────────────────────────────────────────────────────────────
通用財務數據收集工具。

目前包含：
  - data_collection：股票財務數據收集（佔位實作，可接入真實資料源）

新增財務類工具請在本檔案繼續添加，並在底部完成註冊。
"""

from __future__ import annotations

import json

from tools.base import register_tool


# ─────────────────────────────────────────────
# data_collection
# ─────────────────────────────────────────────

@register_tool("data_collection")
def _data_collection(stock_symbol: str, data_type: str = "all") -> str:
    """財務數據收集與整理：從資料源收集指定股票的財務數據。

    Args:
        stock_symbol: 股票代碼，例如 'AAPL'。
        data_type: 資料類型，如 income_statement / balance_sheet / cash_flow / all。

    Returns:
        JSON 字串，含收集狀態與數據摘要。
    """
    return json.dumps(
        {
            "tool": "data_collection",
            "stock_symbol": stock_symbol,
            "data_type": data_type,
            "status": "placeholder",
            "message": (
                f"資料收集 ({stock_symbol}, {data_type})"
                " — 尚未接入真實資料源"
            ),
        },
        ensure_ascii=False,
    )
