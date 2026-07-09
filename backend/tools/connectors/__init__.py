# -*- coding: utf-8 -*-
"""
tools/connectors — 各官方經濟數據源的統一 connector 實作。

每個 connector 模組（fred.py / worldbank.py / oecd.py / bea.py）皆須實作：

    fetch(series_id: str, **kwargs) -> dict

回傳格式一律遵守 base.py 的 success_result() / error_result()，
絕不 raise exception。

新增資料源步驟（比照 data-retriever skill）：
  1. 在此目錄新增 {source}.py，實作 fetch(series_id, **kwargs) -> dict
  2. 在 tools/data_retriever.py 的 _CONNECTORS 字典註冊 "{source}": {source}.fetch
  3. 若有專屬參數（如 country、line_number），在該 connector 的
     docstring 說明清楚，供撰寫 params_manifest.yaml 的 retriever_kwargs 參考

新增資料源不需要改動任何一個已產出的 RC Skill——它們的
params_manifest.yaml 只認 retriever_id，只要格式對，底層新增
什麼資料源都與它們無關。
"""
