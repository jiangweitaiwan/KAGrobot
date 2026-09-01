# -*- coding: utf-8 -*-


from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from orchestrator.nodes      import OrchestratorNodes
from orchestrator.walk_nodes import WalkOrchestratorNodes
from orchestrator.routers    import (
    entry_router,
    socratic_router,
    walk_decide_router,
    walk_router,
)
from orchestrator.state      import OrchestratorState

def build_orchestrator(nodes: OrchestratorNodes, walk_nodes: WalkOrchestratorNodes):
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("socratic",   nodes.socratic)
    workflow.add_node("main_agent", nodes.main_agent)

    workflow.add_node("walk_seed",           walk_nodes.walk_seed)
    workflow.add_node("walk_fetch_frontier", walk_nodes.walk_fetch_frontier)
    workflow.add_node("walk_decide",         walk_nodes.walk_decide)
    workflow.add_node("walk_ask",            walk_nodes.walk_ask)
    workflow.add_node("walk_finish",         walk_nodes.walk_finish)

    workflow.set_conditional_entry_point(
        entry_router,
        {
            "socratic":            "socratic",
            "main_agent":          "main_agent",
            "walk_seed":           "walk_seed",
            "walk_fetch_frontier": "walk_fetch_frontier",
            "walk_decide":         "walk_decide",
            "walk_finish":         "walk_finish",
        },
    )

    workflow.add_conditional_edges(
        "socratic",
        socratic_router,
        {
            "main_agent": "main_agent",
            "walk_seed":  "walk_seed",
            "end":        END,
        },
    )

    _WALK_LOOP_TARGETS = {
        "walk_fetch_frontier": "walk_fetch_frontier",
        "walk_decide":         "walk_decide",
        "walk_finish":         "walk_finish",
    }

    workflow.add_conditional_edges("walk_seed",           walk_router, _WALK_LOOP_TARGETS)
    workflow.add_conditional_edges("walk_fetch_frontier",  walk_router, _WALK_LOOP_TARGETS)
    workflow.add_conditional_edges("walk_ask",              walk_router, _WALK_LOOP_TARGETS)
    workflow.add_conditional_edges(
        "walk_decide",
        walk_decide_router,
        {**_WALK_LOOP_TARGETS, "walk_ask": "walk_ask"},
    )

    workflow.add_edge("walk_finish", END)
    workflow.add_edge("main_agent",  END)

    return workflow.compile(checkpointer=MemorySaver())
