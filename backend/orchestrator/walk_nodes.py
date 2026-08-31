# -*- coding: utf-8 -*-
"""orchestrator/walk_nodes.py — Random Walk Mode 的 LangGraph 節點。

取代 core/walk_bot.py::WalkTurnRunner.run_turn() 的單一大 dispatch，改成
五個獨立、細粒度的 LangGraph node，理由見與使用者的討論：

  1. 每個 node 對應「一次有意義的狀態變化」，天然適合 orchestrator.astream()
     逐 tick 吐出更新，前端 GlobalBrain 的即時 highlight 直接消費這些
     state update，不需要另外手刻一套 SSE 迴圈。
  2. interrupt() 語意上要求「resume 時整個 node 從頭重跑」，所以有副作用
     的呼叫（Neo4j / LLM）跟 interrupt() 本身必須分屬不同 node，否則
     resume 會不必要地重打一次 LLM。見 walk_decide vs walk_ask 的拆分。

Node 對應關係：
  walk_fetch_frontier : 取得 scheduler 選中的 path 的下一輪候選（呼叫
                         frontier_fetcher，內部走 AdjacencyCache，會打
                         Neo4j 但有快取）。對應原本 run_turn() 的
                         _fetch_frontier()。
  walk_decide          : 呼叫 LLM 判斷要不要推進；決定推進就直接
                         advance()；決定要問使用者，則只把問題存進
                         walk_pending_question，不在這裡呼叫 interrupt()。
  walk_ask              : 唯一呼叫 interrupt() 的地方，把
                         walk_pending_question 丟出去暫停，resume 後把
                         answer 摺進 context_summary。
  walk_finish            : 所有 path 皆終止後的收尾，產生一則對話訊息，
                         phase 轉為 "walk_complete"（前端據此顯示排名 UI，
                         等使用者選定 RC）。
  walk_confirm            : 使用者從排名結果選定 RC 後呼叫，比照
                         core/socrates_bot.py::confirm_path 的語意，把
                         phase 轉為 "path_confirmed"，並用
                         walk_session.context_summary 組一份
                         main_agent 看得懂的 socrates_state。

WalkScheduler 的 pointer 存續規則見 graph_service/walk_state.py 檔頭
說明：每個真正執行副作用的 node，離開前都必須把 scheduler.pointer 寫回
state["walk_pointer"]。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from core.chat_state import ChatState, ConversationPhase
from core.walk_bot import (
    _SYSTEM_PROMPT_TEMPLATE,
    _build_advance_tool,
    _format_frontier,
)
from graph_service.walk_state import FrontierCandidate, PathStatus, WalkScheduler, WalkSession
from orchestrator.state import OrchestratorState


class WalkOrchestratorNodes:
    """封裝 random walk 五個節點的業務邏輯。

    Parameters
    ----------
    frontier_fetcher : Callable[[str, str], List[FrontierCandidate]]
                       見 graph_service/walk_frontier.py::make_frontier_fetcher()
                       的輸出，簽章與 core/walk_bot.py::WalkTurnRunner 完全相同，
                       可直接沿用 app.py 既有的組裝方式。
    llm_caller       : Callable[[str, type], Any]，同上，簽章與
                       WalkTurnRunner 相同。
    """

    def __init__(
        self,
        frontier_fetcher: Callable[[str, str], List[FrontierCandidate]],
        llm_caller:       Callable[[str, type], Any],
    ) -> None:
        self._frontier_fetcher = frontier_fetcher
        self._llm_caller       = llm_caller

    # ------------------------------------------------------------------
    # Node: walk_fetch_frontier
    # ------------------------------------------------------------------

    async def walk_fetch_frontier(self, state: OrchestratorState) -> OrchestratorState:
        session, scheduler, path = self._resolve_current_path(state)
        if path is None or path.status != PathStatus.PENDING:
            # 理論上 walk_decide_router / walk_router 已保證進來時符合
            # 這個狀態；防禦性地只同步 pointer，不做其他事。
            return {"walk_pointer": scheduler.pointer}

        candidates = await asyncio.to_thread(
            self._frontier_fetcher, path.current_eid, session.context_summary
        )
        path.set_frontier(candidates)

        return {"walk_session": session.to_dict(), "walk_pointer": scheduler.pointer}

    # ------------------------------------------------------------------
    # Node: walk_decide（呼叫 LLM；決定推進或標記待問問題，不呼叫 interrupt）
    # ------------------------------------------------------------------

    async def walk_decide(self, state: OrchestratorState) -> OrchestratorState:
        session, scheduler, path = self._resolve_current_path(state)
        if path is None or path.status != PathStatus.ACTIVE:
            return {"walk_pointer": scheduler.pointer, "walk_pending_question": None}

        tool_schema = _build_advance_tool(path)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            path_id=path.path_id,
            visited_path=" → ".join(path.visited_labels_only()),
            frontier_desc=_format_frontier(path.frontier),
            context_summary=session.context_summary or "（尚無累積脈絡）",
        )

        ai_msg = await asyncio.to_thread(self._llm_caller, system_prompt, tool_schema)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            question = getattr(ai_msg, "content", "") or ""
            return {
                "walk_session":          session.to_dict(),
                "walk_pointer":          scheduler.pointer,
                "walk_pending_question": {"path_id": path.path_id, "question": question},
            }

        call       = tool_calls[0]
        args       = call.get("args", {})
        target_eid = args.get("target_eid", "")
        reason     = args.get("reason", "")

        # PathState.advance() 內部會再驗證一次 target_eid 落在 frontier
        # 內，即使上方 Literal schema 已限制過，這裡不因此省略防禦。
        path.advance(target_eid, reason=reason)
        session.record_decision(reason)

        return {
            "walk_session":          session.to_dict(),
            "walk_pointer":          scheduler.pointer,
            "walk_pending_question": None,
        }

    # ------------------------------------------------------------------
    # Node: walk_ask（唯一呼叫 interrupt() 的地方）
    # ------------------------------------------------------------------

    async def walk_ask(self, state: OrchestratorState) -> OrchestratorState:
        pending      = state.get("walk_pending_question")
        session_dict = state.get("walk_session")

        if pending is None or session_dict is None:
            # 防禦性：理論上 walk_decide_router 已保證不會空手進來。
            return {"walk_pending_question": None}

        session = WalkSession.from_dict(session_dict)

        # 第一次執行：raise GraphInterrupt，暫停並把 pending 丟給前端。
        # resume 時：這個 node 會整個從頭重跑（包含上面 from_dict()
        # 這幾行，這是刻意的、廉價的重跑），但完全不重打 LLM／Neo4j——
        # 這正是把 walk_decide（呼叫 LLM）跟 walk_ask（只呼叫
        # interrupt()）拆成兩個 node 的理由，見 orchestrator/state.py
        # 對 walk_pending_question 欄位的說明。
        answer = interrupt(pending)

        session.record_decision(f"[path {pending['path_id']} 提問]：{pending['question']}")
        session.record_decision(f"[path {pending['path_id']} 使用者回覆]：{answer}")

        return {"walk_session": session.to_dict(), "walk_pending_question": None}

    # ------------------------------------------------------------------
    # Node: walk_finish
    # ------------------------------------------------------------------

    async def walk_finish(self, state: OrchestratorState) -> OrchestratorState:
        session = WalkSession.from_dict(state["walk_session"])

        lines = [
            f"· path {p.path_id}：{p.current_label}"
            f"（{'RC' if p.final_is_rc else '中繼概念'}）"
            for p in session.paths
        ]
        reply = (
            "本輪探索已完成，各路徑走到的位置：\n" + "\n".join(lines)
            + "\n\n請從左側／Personal Brain 的推薦排名中選定一個 Resulting Concept 開始分析。"
        )

        return OrchestratorState(
            messages=[AIMessage(content=reply)],
            phase="walk_complete",
            walk_session=session.to_dict(),
            attachments=[],
        )

    # ------------------------------------------------------------------
    # Node: walk_confirm
    # ------------------------------------------------------------------

    async def walk_confirm(self, state: OrchestratorState) -> OrchestratorState:
        """比照 core/socrates_bot.py::confirm_path 的語意：使用者選定 RC
        後，把 phase 轉為 path_confirmed。與 dialogue mode 的差異在於
        socrates_state 不是真的跑過 socratic 對話產生的，而是用
        walk_session.context_summary（整趟 walk 累積的決策理由摘要）
        組出一份形狀相容的 ChatState，讓 main_agent 的
        「使用者需求」prompt 區塊有內容可用，不會顯示「（未提供）」。
        """
        session_dict = state.get("walk_session")
        context_summary = session_dict.get("context_summary", "") if session_dict else ""

        new_socrates_state: ChatState = ChatState(
            messages=[],
            agent_messages=[],
            phase=ConversationPhase.PATH_CONFIRMED.value,
            context_summary=context_summary,
        )

        return OrchestratorState(
            messages=[AIMessage(content="路徑已確認，開始執行分析…")],
            phase="path_confirmed",
            socrates_state=new_socrates_state,
            attachments=[],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_current_path(state: OrchestratorState):
        """共用的「讀 state → 還原 WalkSession → 重建 scheduler → 取得
        目前該處理的 path」邏輯，避免 walk_fetch_frontier / walk_decide
        兩處各自重寫一次。

        state["walk_session"] 存的是 WalkSession.to_dict() 的輸出（見
        orchestrator/state.py 的欄位說明），這裡還原成真正的 dataclass
        才能呼叫 set_frontier() / advance() 等方法；node 回傳前記得
        呼叫 session.to_dict() 轉回去，不要把 dataclass 實例直接放回
        state（會被 checkpointer 的 msgpack 序列化層擋下來）。

        回傳 (session, scheduler, path)；path 可能為 None（理論上不會，
        因為 router 已先判斷過，這裡保留防禦）。
        """
        session = WalkSession.from_dict(state["walk_session"])
        scheduler = WalkScheduler(session, state.get("walk_pointer", -1))
        path = scheduler.next_path()
        return session, scheduler, path
