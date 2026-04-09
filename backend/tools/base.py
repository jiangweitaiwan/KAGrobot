"""
tools/base.py
─────────────────────────────────────────────────────────────
工具註冊機制。

提供兩種註冊方式：
  1. @register_tool("tool_name")  — 裝飾一般 Python 函式
  2. register_langchain_tool(...)  — 直接登記 LangChain BaseTool
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.tools import BaseTool

# 全域實作字典：tool_name → Callable 或 BaseTool
_TOOL_IMPLEMENTATIONS: Dict[str, Any] = {}


def register_tool(tool_name: str) -> Callable:
    """裝飾器：將函式註冊為某個 tool name 的實作。

    tool_name 須與 core/registry.py 中 TOOLS_REGISTRY 的 key 一致。

    Example::

        @register_tool("my_tool")
        def my_tool_impl(param: str) -> str:
            ...
    """
    def _decorator(func: Callable[..., str]) -> Callable[..., str]:
        _TOOL_IMPLEMENTATIONS[tool_name] = func
        return func
    return _decorator


def register_langchain_tool(tool_name: str, tool_obj: BaseTool) -> None:
    """直接登記一個 LangChain BaseTool（如 @tool 裝飾的函式）。

    Example::

        from langchain_core.tools import tool

        @tool
        def my_lc_tool(x: float) -> str: ...

        register_langchain_tool("my_lc_tool", my_lc_tool)
    """
    _TOOL_IMPLEMENTATIONS[tool_name] = tool_obj
