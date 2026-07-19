# -*- coding: utf-8 -*-
"""orchestrator/routers.py — LangGraph 條件路由函數。"""

from __future__ import annotations

from orchestrator.state import OrchestratorState


def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    return "main_agent" if phase in ("path_confirmed", "main_agent") else "socratic"


def socratic_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    return "main_agent" if phase == "path_confirmed" else "end"
