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
    if phase == "walk_pending":
        return "walk_seed"
    return "end"