# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Optional

from graph_service.walk_state import PathStatus, WalkScheduler, WalkSession
from orchestrator.state import OrchestratorState



def _next_walk_node(state: OrchestratorState) -> str:
    session_dict = state.get("walk_session")
    if session_dict is None:
        return "walk_finish"

    session = WalkSession.from_dict(session_dict)
    if session.all_done():
        return "walk_finish"

    scheduler = WalkScheduler(session, state.get("walk_pointer", -1))
    path = scheduler.next_path()
    if path is None:
        return "walk_finish"

    return "walk_fetch_frontier" if path.status == PathStatus.PENDING else "walk_decide"


def walk_router(state: OrchestratorState) -> str:
    return _next_walk_node(state)


def walk_decide_router(state: OrchestratorState) -> str:
    if state.get("walk_pending_question") is not None:
        return "walk_ask"
    return _next_walk_node(state)


def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"

    if phase in ("path_confirmed", "main_agent"):
        return "main_agent"

    if phase == "walk_pending":
        if state.get("walk_session") is None:
            return "walk_seed"
        return _next_walk_node(state)

    return "socratic"


def socratic_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase == "path_confirmed":
        return "main_agent"
    # walk_pending 不在這裡直接接 walk_seed —— /chat 這次呼叫只負責
    # 把 phase 轉場，真正驅動 walk_seed 進圖交給 GET /walk/stream 的
    # entry_router（見 orchestrator/routers.py::entry_router），
    # 避免 walk_seed ~ walk_ask 在同一次 ainvoke() 裡悄悄跑掉、
    # 產生沒有經過 astream() 的孤兒 interrupt。
    return "end"