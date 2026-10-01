# -*- coding: utf-8 -*-
"""core/skill_compute.py — HTML Widget 即時試算的 subprocess 隔離執行器。

設計取捨（已與使用者確認：維持獨立 subprocess，不重用 e2b sandbox）：
  - Widget 的「即時試算」（例如拖曳 slider 當下算 VaR）預期呼叫頻率高、
    要求低延遲，若每次都連線 e2b sandbox（網路往返 + sandbox 冷啟動）
    使用者體驗會變差。改用本機 subprocess，用 spawn（非 fork）避免
    子行程繼承父行程的 event loop / DB 連線等狀態。
  - 這是比 e2b 更弱的信任層級：只有 RLIMIT_AS（記憶體上限）與
    wall-clock timeout 兩道防線，沒有真正的 VM/容器隔離，網路與檔案
    系統 namespace 仍與 host 共用。因此：
      1. module_path 一律驗證落在 skill_dir 之內（防 path traversal）。
      2. 只允許呼叫 widget manifest.yaml 明確宣告、且已編碼進
         widget_token 的 compute_functions 白名單，不接受任意路徑。
      3. Skill 作者文件應明確要求 compute function 不得有網路 I/O、
         不得讀寫 skill_dir 以外路徑（見 html/manifest.yaml 撰寫規範）。
      4. skills_bucket_name 的寫入權限需嚴格管控在內部發布流程，
         不對外開放任意上傳——這一層是「作者可信但比後端 repo 弱」的
         中介信任層級，需要在部署文件上特別標注。
  - spawn 模式在容器環境（Cloud Run）的冷啟動延遲，第一次呼叫可能
    比預期高（需要重新 import 整個 Python 直譯器），實際部署前建議
    量測一次；若延遲過高，可以評估改用一個常駐的 worker pool，
    但這超出本次變更範圍，先以正確性與安全邊界為優先。
"""

from __future__ import annotations

import asyncio
import importlib.util
import resource
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict


def _child_entry(
    skill_dir_str: str,
    module_path:   str,
    func_name:     str,
    parameters:    Dict[str, Any],
    mem_limit_mb:  int,
    result_queue,
) -> None:
    """在子行程（spawn）內執行：套用記憶體上限、驗證路徑、動態 import、呼叫函式。

    這個函式本身會被序列化到子行程執行，不能依賴任何跨行程共享的物件
    （不能傳 Path 物件本身、不能傳 queue 以外的複雜物件），因此參數
    一律是基本型別（str / dict / int），符合 multiprocessing spawn 的限制。
    """
    try:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (mem_limit_mb * 1024 * 1024, mem_limit_mb * 1024 * 1024),
            )
        except (ValueError, resource.error):
            # 某些平台（例如部分容器環境）可能不支援調整 RLIMIT_AS，
            # 降級為不設限但仍執行——timeout 仍是最後一道防線。
            pass

        skill_dir = Path(skill_dir_str).resolve()
        target    = (skill_dir / module_path).resolve()

        if skill_dir != target and skill_dir not in target.parents:
            result_queue.put({"ok": False, "error": "module_path 越界，拒絕執行"})
            return
        if not target.exists():
            result_queue.put({"ok": False, "error": f"找不到模組檔案：{module_path}"})
            return

        spec = importlib.util.spec_from_file_location("skill_compute_mod", target)
        if spec is None or spec.loader is None:
            result_queue.put({"ok": False, "error": f"無法載入模組：{module_path}"})
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func = getattr(module, func_name, None)
        if func is None or not callable(func):
            result_queue.put({"ok": False, "error": f"模組內找不到可呼叫的函式：{func_name}"})
            return

        result = func(**parameters)
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:  # noqa: BLE001 — 子行程任何例外都要回報，不能讓它默默死掉
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


async def run_compute_function(
    skill_dir:    Path,
    fn_ref:       str,
    parameters:   Dict[str, Any],
    *,
    timeout_s:    int = 5,
    mem_limit_mb: int = 256,
) -> Dict[str, Any]:
    """在隔離 subprocess 內執行一個 compute function，回傳 {"ok": bool, ...}。

    Parameters
    ----------
    fn_ref : "module_path:func_name"，來自 HtmlWidgetSpec.compute_functions
             的其中一個 value（已經過 widget_token 白名單驗證）。
    """
    module_path, sep, func_name = fn_ref.partition(":")
    if not sep:
        return {"ok": False, "error": f"compute function 參照格式錯誤：{fn_ref}"}

    ctx   = get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_child_entry,
        args=(str(skill_dir), module_path, func_name, parameters, mem_limit_mb, queue),
        daemon=True,
    )
    proc.start()

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, queue.get), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.terminate()
        return {"ok": False, "error": f"執行逾時（超過 {timeout_s} 秒）"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"執行失敗：{exc}"}
    finally:
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)

    if not isinstance(result, dict) or "ok" not in result:
        return {"ok": False, "error": "compute function 未回傳合法結果"}
    return result
