# -*- coding: utf-8 -*-
"""orchestrator/graph.py — LangGraph workflow 組裝。

build_orchestrator() 接受已初始化的 OrchestratorNodes，
回傳 compiled LangGraph app，由 lifespan 呼叫一次。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from orchestrator.nodes   import OrchestratorNodes
from orchestrator.routers import entry_router, socratic_router
from orchestrator.state   import OrchestratorState


def build_orchestrator(nodes: OrchestratorNodes):
    """組裝並編譯 LangGraph orchestrator。

    Parameters
    ----------
    nodes : OrchestratorNodes（已注入所有依賴）

    Returns
    -------
    CompiledGraph（帶 MemorySaver checkpointer）
    """
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("socratic",   nodes.socratic)
    workflow.add_node("main_agent", nodes.main_agent)

    workflow.set_conditional_entry_point(
        entry_router,
        {
            "socratic":   "socratic",
            "main_agent": "main_agent",
        },
    )
    workflow.add_conditional_edges(
        "socratic",
        socratic_router,
        {
            "main_agent": "main_agent",
            "end":        END,
        },
    )
    workflow.add_edge("main_agent", END)

    return workflow.compile(checkpointer=MemorySaver())
