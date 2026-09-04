# -*- coding: utf-8 -*-
"""core/walk_stream.py — Random Walk（慢想）Mode 的 SSE 事件轉譯層。

【重構說明 — 併入 /chat 主流程後的變化】
Walk 不再有獨立的 walk_id 與 /walk/start REST endpoint：一趟慢想的
起點選取（原本 StartConceptPool.retrieve 的呼叫）現在是圖內的第一個
node —— walk_seed（見 orchestrator/walk_nodes.py），由 socratic 判斷
語意已收斂且 walk_enabled=True 時，把 phase 轉為 "walk_pending" 觸發
進圖。因此 SSE 事件流現在多一個起點：

  {"walk_seed": update} → walk_started（成功，附上三條 path 的起點）
                          或 walk_aborted（需求不足 / 找不到起點，
                          walk_seed 已把 phase 轉回 "socratic"）

walk_finish 完成後不再產生 ranking——排名完全交給既有的
/brain/rc-semantic-scores（walk 只是把 WalkSession.context_summary
寫進 OrchestratorState.thinking_process，讓語意排名多吃一份探索過程
的語料，見 core/semantic_context.py）。walk_complete 事件因此只帶
walk_finish 產生的說明文字，前端收到後另外重新拉一次語意排名 API
即可（見 page.tsx::handleWalkEvent 的 "walk_complete" 分支）。

【path_id 的歸屬 — 循序制排程後不再含糊】
graph_service/walk_state.py::WalkScheduler 已改為循序制：任一時刻
最多只有一條 path 處於 ACTIVE/PENDING，其餘皆已 DEAD_END 或尚未輪到。
這代表：
  - frontier_ready / frontier_expanded / path_dead_end / user_question
    這幾個事件的 path_id，永遠對應「當下唯一正在被推進的那條 path」，
    不會出現兩條 path 同時等待使用者回答的情況。
  - 前端不需要額外邏輯去判斷「這個問題是哪條 path 問的」——它就是
    目前畫面上唯一活躍的 path。path_id 欄位依然保留在事件裡，
    是為了讓前端能明確標示（例如「path 1 提問」），而非用於消歧義。

翻譯規則（其餘與前一版相同）：
  - "__interrupt__" 出現 → user_question（不看是哪個 node 產生的）
  - {"walk_seed": update} → walk_started 或 walk_aborted（見上）
  - {"walk_fetch_frontier": update} → frontier_ready 或 path_dead_end
    （frontier 抓回來發現是空的，代表這條 path 已經走到底）
  - {"walk_decide": update} → 兩種情況：
      a. update 裡帶了 walk_pending_question → 這個 tick 本身不對應
         任何使用者可見事件；真正的 user_question 事件會在緊接著的
         "__interrupt__" chunk 出現，這裡刻意不重複發送。
      b. 否則代表 LLM 決定推進了 → frontier_expanded；如果推進後這條
         path 因為達到 max_depth 而直接被標記終止（PathState.advance()
         內部行為），這個 tick 不會再進 walk_fetch_frontier（循序制
         scheduler 之後會直接跳去下一條 path），所以「終止」這件事
         必須在這裡一併補發一個 path_dead_end。
  - {"walk_ask": update} → 沒有對應事件（resume 後只是把答案記錄進
    context_summary，使用者早就知道自己回答了什麼，不需要额外通知）。
  - {"walk_finish": update} → walk_complete（帶 summary 文字，不帶
    ranking）。

path_id 的取得：見 graph_service/walk_state.py::WalkSession.start()
與 WalkScheduler 的說明，直接讀 update["walk_pointer"] 即可對應該
path 在 session["paths"] 陣列裡的 index。
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

    if node_name == "walk_seed":
        return _events_from_seed(update)
    if node_name == "walk_fetch_frontier":
        return _events_from_fetch_frontier(update)
    if node_name == "walk_decide":
        return _events_from_decide(update)
    if node_name == "walk_finish":
        return _events_from_finish(update)

    # walk_ask（resume 後只記錄不通知，見檔頭說明）等其餘 node
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


def _message_text(update: Dict[str, Any]) -> str:
    msgs = update.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content) if content is not None else ""


def _events_from_seed(update: Dict[str, Any]) -> List[dict]:
    """walk_seed 的兩種結果：

    成功 → update 帶有 walk_session，代表三條 path 的起點已選定，
           對應事件 walk_started，附上每條 path 的起點供前端顯示。
    失敗 → update 沒有 walk_session（需求不夠具體、或找不到合適起點），
           walk_seed 已經把 phase 轉回 "socratic" 並附上一則說明訊息，
           對應事件 walk_aborted，前端應將其當作一般 AI 回覆呈現，
           並結束這段 SSE（walk_seed 沒有出圖到 walk_fetch_frontier，
           stream 會自然在這裡結束）。
    """
    session_dict = update.get("walk_session")
    if session_dict is None:
        return [{
            "type":    "walk_aborted",
            "message": _message_text(update),
        }]

    session = WalkSession.from_dict(session_dict)
    return [{
        "type": "walk_started",
        "start_concepts": [
            {"path_id": p.path_id, "eid": p.start_eid, "label": p.start_label}
            for p in session.paths
        ],
    }]


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

    # advance() 若因為達到 max_depth 而直接終止，循序制排程下這條 path
    # 不會再進 walk_fetch_frontier（scheduler 會直接換到下一條 path），
    # "終止" 這件事必須在這裡一併回報。
    if path["status"] == "dead_end":
        events.append({
            "type":    "path_dead_end",
            "path_id": path_id,
            "reason":  path.get("dead_end_reason"),
        })

    return events


def _events_from_finish(update: Dict[str, Any]) -> List[dict]:
    """walk_finish 完成：不再附帶 ranking——排名交給
    /brain/rc-semantic-scores（thinking_process 已寫入
    OrchestratorState，見 orchestrator/walk_nodes.py::walk_finish）。
    這裡只把說明文字透傳出去，前端收到後另外重新拉一次語意排名。
    """
    return [{
        "type":    "walk_complete",
        "summary": _message_text(update),
    }]


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
    """驅動 orchestrator.astream() 跑 random walk 子流程，逐 chunk 轉成
    SSE 字串。invoke_input 可以是：

      - 一份最小 state dict（例如 {"messages": []}），單純用來觸發
        entry_router 重新進圖——phase 早已由 /chat 那次 ainvoke 寫進
        checkpoint 為 "walk_pending"（見 api/routes/routes_walk.py 的
        GET /walk/stream/{session_id}）
      - langgraph.types.Command(resume=answer)（使用者回答問題後，
        見 POST /walk/answer/{session_id}）

    連線在圖真正跑到 END（walk_finish 或 walk_seed 失敗提前結束之後）
    或再次卡在 interrupt() 時自然結束——不需要手動判斷「這輪 kind 是
    不是 question / walk_complete」才 return，astream() 本身在遇到
    這兩種情況都會自然停止 yield。
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