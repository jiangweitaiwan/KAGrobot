"""
tools/sandbox.py
─────────────────────────────────────────────────────────────
GCP Cloud Function Python 沙盒執行工具。

由 core/main_agent.py 在每次 _build_tools 時動態建立，
透過 settings.python_sandbox_url 連線至遠端沙盒服務。
"""

from __future__ import annotations

import requests
from langchain_core.tools import StructuredTool

from settings import get_settings


def make_execute_python_in_sandbox_tool() -> StructuredTool:
    """建立 execute_python_in_sandbox 工具。

    每次呼叫都從 settings 讀取沙盒 URL，不持有連線狀態。

    Returns:
        LangChain StructuredTool。
    """

    def execute_python_in_sandbox(code: str) -> str:
        """當需要計算矩陣、模擬 DSGE 模型或執行 Python 程式碼時呼叫此工具。"""
        sandbox_url = get_settings().python_sandbox_url.strip()
        if not sandbox_url:
            return "沙盒服務未設定。請在環境變數 PYTHON_SANDBOX_URL 設定 GCP Cloud Function URL。"

        try:
            response = requests.post(
                sandbox_url, json={"code": code}, timeout=15
            )
            res_data = response.json()

            if res_data.get("success"):
                return f"沙盒執行成功。輸出結果：\n{res_data.get('stdout')}"
            return f"沙盒執行失敗。錯誤訊息：\n{res_data.get('stderr')}"
        except Exception as e:
            return f"無法連線至沙盒服務：{str(e)}"

    return StructuredTool.from_function(
        func=execute_python_in_sandbox,
        name="execute_python_in_sandbox",
        description=(
            "在遠端安全沙盒中執行 Python 程式碼。"
            "當需要計算矩陣、模擬 DSGE 模型、數值運算或驗證公式時呼叫此工具。"
            "傳入完整可執行的 Python 程式碼字串。"
        ),
    )
