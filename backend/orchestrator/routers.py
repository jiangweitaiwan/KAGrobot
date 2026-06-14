# -*- coding: utf-8 -*-
"""orchestrator/routers.py — LangGraph 條件路由函數。"""

from __future__ import annotations

from orchestrator.state import OrchestratorState


def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase in ("socratic", "concept_resolved"):
        return "socratic"
    if phase in ("path_confirmed", "main_agent"):
        return "main_agent"
    return "socratic"


def socratic_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase == "path_confirmed":
        return "main_agent"
    return "end"
