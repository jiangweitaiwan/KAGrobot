# -*- coding: utf-8 -*-
"""orchestrator/graph.py — LangGraph workflow 組裝。

build_orchestrator() 接受已初始化的 OrchestratorNodes 與
WalkOrchestratorNodes，回傳 compiled LangGraph app，由 lifespan 呼叫一次。

【新增 — Random Walk Mode】
Random walk 的五個節點（walk_fetch_frontier / walk_decide / walk_ask /
walk_finish / walk_confirm，見 orchestrator/walk_nodes.py）與既有的
socratic / main_agent 節點掛在同一張圖上，靠 entry_router（讀
state["mode"]）在對話一開始就分流，兩套流程互不干擾，但共用：
  - 同一個 MemorySaver checkpointer（同一個 thread_id 慣例）
  - main_agent 交接點（walk_confirm 與 socratic 一樣，最終都是把
    phase 設為 "path_confirmed" 後把球交給 main_agent）

walk 迴圈本身的形狀：

    entry ─┐
           ├─(mode=random_walk)─▶ walk_fetch_frontier ──┐
           │                                            │
           │        ┌───────────────────────────────────┘
           │        ▼
           │   walk_decide ──(需要問使用者)──▶ walk_ask ──┐
           │        │                                    │
           │        └──(已推進 / 無需問)────┐             │
           │                               ▼             │
           │                        （walk_router 判斷）◀─┘
           │                               │
           │                 ┌─────────────┼─────────────┐
           │                 ▼             ▼             ▼
           │        walk_fetch_frontier  walk_decide  walk_finish → END
           │
           └─(選定 RC)──▶ walk_confirm ──▶ main_agent → END

walk_fetch_frontier / walk_ask 跑完後一律用 walk_router 判斷下一步；
walk_decide 跑完後用 walk_decide_router（多一個「是否有待問問題」的
優先判斷）。兩個 router 對外可能路由到的節點名稱集合不同（walk_decide_router
多一個 "walk_ask"），因此各自用獨立的 mapping dict 註冊。
"""

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

# walk_fetch_frontier 與 walk_ask 共用的一般 loop-back 目標集合。
_WALK_LOOP_TARGETS = {
    "walk_fetch_frontier": "walk_fetch_frontier",
    "walk_decide":         "walk_decide",
    "walk_finish":         "walk_finish",
}


def build_orchestrator(nodes: OrchestratorNodes, walk_nodes: WalkOrchestratorNodes):
    """組裝並編譯 LangGraph orchestrator。

    Parameters
    ----------
    nodes      : OrchestratorNodes（dialogue mode，已注入所有依賴）
    walk_nodes : WalkOrchestratorNodes（random walk mode，已注入
                 frontier_fetcher / llm_caller）

    Returns
    -------
    CompiledGraph（帶 MemorySaver checkpointer）
    """
    workflow = StateGraph(OrchestratorState)

    # ── Dialogue mode（既有）───────────────────────────────────
    workflow.add_node("socratic",   nodes.socratic)
    workflow.add_node("main_agent", nodes.main_agent)

    # ── Random walk mode（新增）─────────────────────────────────
    workflow.add_node("walk_fetch_frontier", walk_nodes.walk_fetch_frontier)
    workflow.add_node("walk_decide",         walk_nodes.walk_decide)
    workflow.add_node("walk_ask",            walk_nodes.walk_ask)
    workflow.add_node("walk_finish",         walk_nodes.walk_finish)
    workflow.add_node("walk_confirm",        walk_nodes.walk_confirm)

    # ── 進入點：依 mode 與 phase 分流 ───────────────────────────
    workflow.set_conditional_entry_point(
        entry_router,
        {
            "socratic":            "socratic",
            "main_agent":          "main_agent",
            "walk_fetch_frontier": "walk_fetch_frontier",
            "walk_decide":         "walk_decide",
            "walk_finish":         "walk_finish",
            "walk_confirm":        "walk_confirm",
        },
    )

    # ── Dialogue mode 出口（既有）───────────────────────────────
    workflow.add_conditional_edges(
        "socratic",
        socratic_router,
        {
            "main_agent": "main_agent",
            "end":        END,
        },
    )

    # ── walk_confirm 與 socratic 共用同一套出口語意 ─────────────
    # （兩者都只是把 phase 設成 path_confirmed 與否，見
    # orchestrator/routers.py::socratic_router 的說明）
    workflow.add_conditional_edges(
        "walk_confirm",
        socratic_router,
        {
            "main_agent": "main_agent",
            "end":        END,
        },
    )

    # ── Random walk 迴圈 ─────────────────────────────────────────
    workflow.add_conditional_edges(
        "walk_fetch_frontier", walk_router, _WALK_LOOP_TARGETS,
    )
    workflow.add_conditional_edges(
        "walk_ask", walk_router, _WALK_LOOP_TARGETS,
    )
    workflow.add_conditional_edges(
        "walk_decide",
        walk_decide_router,
        {**_WALK_LOOP_TARGETS, "walk_ask": "walk_ask"},
    )

    workflow.add_edge("walk_finish", END)
    workflow.add_edge("main_agent",  END)

    return workflow.compile(checkpointer=MemorySaver())
