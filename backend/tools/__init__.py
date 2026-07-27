"""
tools 套件：所有可用工具的集中匯出點。

新增工具步驟：
  1. 在 tools/ 下建立新的 .py 檔案（參考 dsge.py 格式）
  2. 在該檔案底部用 register_langchain_tool / @register_tool 完成註冊
  3. 在本 __init__.py 中 import 你的新模組（觸發副作用式註冊）

注意：tools/params_resolver.py 不在此列——它是工廠函數
（make_resolve_params_tool），需要 skill_dir 才能建立，比照
tools/knowledge.py 的模式由 core/main_agent.py 在 _build_tools 內
直接呼叫，不透過 _TOOL_IMPLEMENTATIONS 全域字典管理。
"""

# 觸發各工具模組的副作用式註冊
from tools import data_collection  # noqa: F401
from tools import data_retriever   # noqa: F401
from tools import dsge              # noqa: F401
from tools import knowledge         # noqa: F401