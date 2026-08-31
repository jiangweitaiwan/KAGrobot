# -*- coding: utf-8 -*-
"""core/walk_stream.py — Random Walk Mode 的 SSE 事件轉譯層（改版）。

【變更說明 — 改吃 orchestrator.astream() 的 chunk，而非 WalkTurnResult】
舊版 to_sse_event() 的輸入是 core/walk_bot.py::WalkTurnResult，一個手刻的
dataclass，由 WalkTurnRunner.run_turn() 每輪明確組出來。現在 walk 的推進
邏輯搬進了 orchestrator/walk_nodes.py 的 LangGraph node，驅動方式也從
「呼叫端手動 for 迴圈呼叫 run_turn()」改成「orchestrator.astream(...,
stream_mode='updates')」，每個 tick 吐出的是 `{node_name: state_update}`
這種形狀的 chunk，或是暫停時的 `{"__interrupt__": (Interrupt(...),)}`。

這裡的 to_sse_events() 負責把這些 chunk 翻譯成前端 RandomWalkPanel.tsx
早已在消費的同一套 event schema（frontier_ready / frontier_expanded /
path_dead_end / user_question / walk_complete），front-end 完全不需要
改動。

翻譯規則（見各 helper function 的說明）：
  - "__interrupt__" 出現 → user_question（不看是哪個 node 產生的）
  - {"walk_fetch_frontier": update} → frontier_ready 或 path_dead_end
    （frontier 抓回來發現是空的，代表這條 path 已經走到底）
  - {"walk_decide": update} → 兩種情況：
      a. update 裡帶了 walk_pending_question → 這個 tick 本身不對應任何
         使用者可見事件；真正的 user_question 事件會在緊接著的
         "__interrupt__" chunk 出現，這裡刻意不重複發送。
      b. 否則代表 LLM 決定推進了 → frontier_expanded；如果推進後這條
         path 因為達到 max_depth 而直接被標記終止（PathState.advance()
         內部行為），這個 tick不會再進 walk_fetch_frontier（scheduler
         之後會跳過已 dead_end 的 path），所以「終止」這件事必須在這裡
         一併補發一個 path_dead_end，否則這種終止方式永遠不會被前端
         看到（這是相對舊版 WalkTurnResult 設計補上的一個小改進，
         no_downstream 終止方式一直都有對應事件，max_depth 終止方式
         舊版其實也沒有，這裡一併修正）。
  - {"walk_ask": update} → 沒有對應事件（resume 後只是把答案記錄進
    context_summary，使用者早就知道自己回答了什麼，不需要额外通知）。
  - {"walk_finish": update} → walk_complete。

path_id 的取得：WalkSession.start() 建立時 path_id 就等於它在
session.paths 這個 list 裡的 index（graph_service/walk_state.py），而
WalkScheduler.next_path() 選中某條 path 時，會把該 index 存進
self._pointer（也就是 node 回傳、我們存進 state 的 walk_pointer）。
因此「這個 tick 是在處理哪條 path」，直接讀 update["walk_pointer"] 就是
該 path 在 session["paths"] 陣列裡的 index，不需要另外比對前後兩份
session 的差異。
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from graph_service.walk_state import WalkSession

_DEFAULT_MAX_CHUNKS = 400   # 安全上限，避免圖邏輯有誤造成無限串流


# ══════════════════════════════════════════════════════════════
# Chunk → SSE event(s) 翻譯
# ══════════════════════════════════════════════════════════════

def to_sse_events(chunk: Dict[str, Any]) -> List[dict]:
    """把 astream(stream_mode='updates') 吐出的一個 chunk 轉成 0～N 個
    前端可消費的 event dict（多數情況是 0 或 1 個，只有 walk_decide 在
    「推進後直接因 max_depth 終止」這個邊界情況會是 2 個）。
    """
    if "__interrupt__" in chunk:
        return _events_from_interrupt(chunk["__interrupt__"])

    if len(chunk) != 1:
        # walk 圖沒有平行分支（刻意設計，見與使用者的討論），理論上
        # 每個 chunk 只會有一個 node 的 key；防禦性地忽略不符預期的形狀。
        return []

    node_name, update = next(iter(chunk.items()))

    if node_name == "walk_fetch_frontier":
        return _events_from_fetch_frontier(update)
    if node_name == "walk_decide":
        return _events_from_decide(update)
    if node_name == "walk_finish":
        return _events_from_finish(update)

    # walk_ask、walk_confirm（不會出現在這裡，見檔尾說明）等其餘 node
    # 沒有對應的使用者可見事件。
    return []


def _events_from_interrupt(interrupts: tuple) -> List[dict]:
    events = []
    for itr in interrupts:
        payload = getattr(itr, "value", itr) or {}
        events.append({
            "type":     "user_question",
            "path_id":  payload.get("path_id"),
            "question": payload.get("question", ""),
        })
    return events


def _events_from_fetch_frontier(update: Dict[str, Any]) -> List[dict]:
    session_dict = update.get("walk_session")
    path_id      = update.get("walk_pointer")
    if session_dict is None or path_id is None:
        return []  # 防禦分支（node 內部的早期 return），理論上不會發生

    path = _path_at(session_dict, path_id)
    if path is None:
        return []

    if path["status"] == "dead_end":
        return [{
            "type":    "path_dead_end",
            "path_id": path_id,
            "reason":  path.get("dead_end_reason"),
        }]

    return [{
        "type":     "frontier_ready",
        "path_id":  path_id,
        "candidates": [
            {
                "eid":   c["eid"],
                "label": c["label"],
                "is_rc": "resulting" in (c.get("node_type") or "").lower(),
            }
            for c in (path.get("frontier") or [])
        ],
    }]


def _events_from_decide(update: Dict[str, Any]) -> List[dict]:
    if update.get("walk_pending_question") is not None:
        # 真正的 user_question 事件會在緊接著的 "__interrupt__" chunk
        # 出現，這裡不重複發送。
        return []

    session_dict = update.get("walk_session")
    path_id      = update.get("walk_pointer")
    if session_dict is None or path_id is None:
        return []  # 防禦分支

    path = _path_at(session_dict, path_id)
    if path is None:
        return []

    decision_log = path.get("decision_log") or []
    if not decision_log:
        return []  # 防禦：理論上剛推進過一定有 log

    last    = decision_log[-1]
    visited = path.get("visited") or []
    from_eid = visited[-2]["eid"] if len(visited) >= 2 else path.get("start_eid")

    events = [{
        "type":     "frontier_expanded",
        "path_id":  path_id,
        "from_eid": from_eid,
        "to_eid":   path.get("current_eid"),
        "to_label": path.get("current_label"),
        "reason":   last.get("reason", ""),
    }]

    # advance() 若因為達到 max_depth 而直接終止，這個 tick 不會再進
    # walk_fetch_frontier（見本檔頭說明），"終止" 這件事必須在這裡
    # 一併回報。
    if path["status"] == "dead_end":
        events.append({
            "type":    "path_dead_end",
            "path_id": path_id,
            "reason":  path.get("dead_end_reason"),
        })

    return events


def _events_from_finish(update: Dict[str, Any]) -> List[dict]:
    session_dict = update.get("walk_session") or {"session_id": "", "paths": []}
    session = WalkSession.from_dict(session_dict)
    return [{"type": "walk_complete", "ranking": session.all_foundation_eids()}]


def _path_at(session_dict: Dict[str, Any], path_id: int) -> Optional[dict]:
    paths = session_dict.get("paths") or []
    if not (0 <= path_id < len(paths)):
        return None
    return paths[path_id]


def format_sse(event: dict) -> str:
    """組成 SSE wire format 字串。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ══════════════════════════════════════════════════════════════
# Streaming driver
# ══════════════════════════════════════════════════════════════

async def stream_walk_via_orchestrator(
    orchestrator: Any,
    config:       dict,
    invoke_input: Any,
    max_chunks:   int = _DEFAULT_MAX_CHUNKS,
) -> AsyncGenerator[str, None]:
    """驅動 orchestrator.astream() 跑 random walk 子圖，逐 chunk 轉成 SSE
    字串。invoke_input 可以是：

      - 一份完整的初始 state dict（第一次呼叫，見 api/routes/routes_walk.py
        的 /walk/start + /walk/stream 兩段式流程）
      - langgraph.types.Command(resume=answer)（使用者回答問題後，見
        /walk/answer；也對應 core/walk_bot.py 舊版 stream_walk_turns() 的
        「resume = 開一段新 stream」設計精神，只是現在 resume 是 LangGraph
        原生機制，不需要呼叫端手動把答案摺進 context_summary）

    連線在圖真正跑到 END（walk_finish 之後）或再次卡在 interrupt() 時
    自然結束——不需要像舊版那樣手動判斷「這輪 kind 是不是 question /
    walk_complete」才 return，astream() 本身在遇到這兩種情況都會自然
    停止 yield。
    """
    chunk_count = 0
    async for chunk in orchestrator.astream(invoke_input, config=config, stream_mode="updates"):
        chunk_count += 1
        if chunk_count > max_chunks:
            yield format_sse({
                "type":    "walk_error",
                "message": f"超過 max_chunks={max_chunks}，強制中止（可能有邏輯錯誤造成無限迴圈）",
            })
            return

        for event in to_sse_events(chunk):
            yield format_sse(event)
