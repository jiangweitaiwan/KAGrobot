# -*- coding: utf-8 -*-
"""orchestrator/routers.py — LangGraph 條件路由函數。

【新增 — Random Walk Mode】
entry_router 現在依 state["mode"] 分派到三條路徑之一：
  - phase 已是 path_confirmed/main_agent → 一律進 main_agent
    （不論 mode，這是既有的「已確認路徑，繼續分析對話」邏輯，
    dialogue 與 random walk 收斂到同一個交接點）
  - mode == "random_walk"：
      - 使用者剛從排名結果選定一個 RC（selected_rc_eid 有值，但 phase
        尚未轉為 path_confirmed）→ 進 walk_confirm（比照 dialogue mode
        socratic 節點裡「selected_rc_eid is not None → confirm_path」
        的那段邏輯，只是搬到獨立節點，見 orchestrator/walk_nodes.py）
      - 否則 → 進 walk 迴圈本身該去的下一個節點（_next_walk_node）
  - 其餘（mode == "dialogue" 或未設定）→ 沿用原本的 socratic

walk_router / walk_decide_router 是 walk 迴圈內部「跑完一個 node 之後
該去哪」的判斷，邏輯與 entry_router 內用的 _next_walk_node 共用同一份
實作，避免兩處各自維護一份、日後改 WalkScheduler 用法時漏改其中一處。
"""

from __future__ import annotations

from typing import Optional

from graph_service.walk_state import PathStatus, WalkScheduler, WalkSession
from orchestrator.state import OrchestratorState


# ══════════════════════════════════════════════════════════════
# Random Walk 迴圈內部路由（共用邏輯）
# ══════════════════════════════════════════════════════════════

def _next_walk_node(state: OrchestratorState) -> str:
    """依 WalkSession 目前狀態，決定 walk 迴圈下一個該進哪個 node。

    刻意用一個「用完即丟」的 WalkScheduler 讀 state 現有的
    walk_pointer 來判斷——不會修改、也不需要把這次算出的 pointer 存回
    state（該次呼叫若真的選中了某條 path 而推進 pointer，只有實際執行
    副作用的 node（walk_fetch_frontier / walk_decide）才需要把新 pointer
    寫回 state；純路由判斷這裡呼叫 next_path() 純粹是為了讀出「現在該去
    哪」，重算一次是安全且 idempotent 的，因為 pointer 只有在真正執行的
    node 裡才會被持久化。
    """
    session_dict = state.get("walk_session")
    if session_dict is None:
        return "walk_finish"

    # state["walk_session"] 是 WalkSession.to_dict() 的輸出（見
    # orchestrator/state.py 的欄位說明），router 只是讀取判斷、不修改，
    # 每次還原一份新的 WalkSession 是安全且低成本的。
    session = WalkSession.from_dict(session_dict)
    if session.all_done():
        return "walk_finish"

    scheduler = WalkScheduler(session, state.get("walk_pointer", -1))
    path = scheduler.next_path()
    if path is None:
        return "walk_finish"

    return "walk_fetch_frontier" if path.status == PathStatus.PENDING else "walk_decide"


def walk_router(state: OrchestratorState) -> str:
    """walk_fetch_frontier / walk_ask 跑完後的一般 loop-back 路由。"""
    return _next_walk_node(state)


def walk_decide_router(state: OrchestratorState) -> str:
    """walk_decide 跑完後的路由：優先檢查是否需要暫停詢問使用者。"""
    if state.get("walk_pending_question") is not None:
        return "walk_ask"
    return _next_walk_node(state)


# ══════════════════════════════════════════════════════════════
# 進入點與 dialogue mode 既有路由
# ══════════════════════════════════════════════════════════════

def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    mode  = state.get("mode") or "dialogue"

    if phase in ("path_confirmed", "main_agent"):
        return "main_agent"

    if mode == "random_walk":
        if state.get("selected_rc_eid") is not None:
            return "walk_confirm"
        return _next_walk_node(state)

    return "socratic"


def socratic_router(state: OrchestratorState) -> str:
    """socratic 與 walk_confirm 共用同一個出口路由：兩者都只是把 phase
    轉成 path_confirmed 與否，其餘語意完全一致，不需要各自維護一份。"""
    phase = state.get("phase") or "socratic"
    return "main_agent" if phase == "path_confirmed" else "end"
