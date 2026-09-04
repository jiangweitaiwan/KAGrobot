# -*- coding: utf-8 -*-


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

from graph_service.walk_start_pool import StartConceptPool
from typing import Awaitable


class WalkOrchestratorNodes:
    def __init__(
        self,
        frontier_fetcher: Callable[[str, str], List[FrontierCandidate]],
        llm_caller:       Callable[[str, type], Any],
        start_pool:       StartConceptPool,
        brain_weights_loader: Callable[[Optional[str]], Awaitable[Dict[str, float]]],
        max_depth:        int = 8,
    ) -> None:
        self._frontier_fetcher     = frontier_fetcher
        self._llm_caller           = llm_caller
        self._start_pool           = start_pool
        self._brain_weights_loader = brain_weights_loader
        self._max_depth            = max_depth

    # ------------------------------------------------------------------
    # Node: walk_seed（新增，取代舊版獨立的 /walk/start endpoint）
    # ------------------------------------------------------------------

    async def walk_seed(self, state: OrchestratorState) -> OrchestratorState:
        socrates_state = state.get("socrates_state") or {}
        query = (socrates_state.get("context_summary") or "").strip()
        uid   = state.get("uid")

        if not query:
            # 理論上不會發生：entry_router 只有在 socratic 剛把
            # ready_for_search=True 時才會進來，此時 context_summary
            # 必然有值。防禦性地直接結束，回到 socratic 繼續對話。
            return OrchestratorState(
                messages=[AIMessage(content="需求還不夠明確，我們繼續聊聊你想分析的方向。")],
                phase="socratic",
            )

        brain_weights = await self._brain_weights_loader(uid)
        start_concepts = await self._start_pool.retrieve(query, brain_weights, 3)
        if not start_concepts:
            return OrchestratorState(
                messages=[AIMessage(content="找不到合適的起點概念，我們換個角度聊聊？")],
                phase="socratic",
            )

        session = WalkSession.start(
            session_id=f"{uid}:{state.get('session_id') or 'unknown'}",
            start_concepts=start_concepts,
            max_depth=self._max_depth,
        )

        return {
            "walk_session":  session.to_dict(),
            "walk_pointer":  -1,
            "phase":         "walk_pending",
        }

    # ------------------------------------------------------------------
    # Node: walk_fetch_frontier（不變）
    # ------------------------------------------------------------------

    async def walk_fetch_frontier(self, state: OrchestratorState) -> OrchestratorState:
        session, scheduler, path = self._resolve_current_path(state)
        if path is None or path.status != PathStatus.PENDING:
            return {"walk_pointer": scheduler.pointer}

        candidates = await asyncio.to_thread(
            self._frontier_fetcher, path.current_eid, session.context_summary
        )
        path.set_frontier(candidates)

        return {"walk_session": session.to_dict(), "walk_pointer": scheduler.pointer}

    # ------------------------------------------------------------------
    # Node: walk_decide（不變，仍是「決定推進或標記待問」，不呼叫 interrupt）
    # ------------------------------------------------------------------

    async def walk_decide(self, state: OrchestratorState) -> OrchestratorState:
        session, scheduler, path = self._resolve_current_path(state)
        if path is None or path.status != PathStatus.ACTIVE:
            return {"walk_pointer": scheduler.pointer, "walk_pending_question": None}

        # 唯一候選：不需要 LLM 判斷，直接前進
        if len(path.frontier) == 1:
            only = path.frontier[0]
            path.advance(only.eid, reason="唯一候選，直接前進")
            return {
                "walk_session":          session.to_dict(),
                "walk_pointer":          scheduler.pointer,
                "walk_pending_question": None,
            }

        socrates_state = state.get("socrates_state") or {}
        user_goal = socrates_state.get("context_summary") or "（使用者尚未提供具體目的）"

        tool_schema   = _build_advance_tool(path)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            path_id=path.path_id,
            user_goal=user_goal,
            thinking_chain=path.thinking_chain(last_n=3),
            frontier_desc=_format_frontier(path.frontier),
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
        path.advance(target_eid, reason=reason)

        return {
            "walk_session":          session.to_dict(),
            "walk_pointer":          scheduler.pointer,
            "walk_pending_question": None,
        }

    # ------------------------------------------------------------------
    # Node: walk_ask（不變，唯一呼叫 interrupt() 的地方）
    # ------------------------------------------------------------------

    async def walk_ask(self, state: OrchestratorState) -> OrchestratorState:
        pending      = state.get("walk_pending_question")
        session_dict = state.get("walk_session")

        if pending is None or session_dict is None:
            return {"walk_pending_question": None}

        session = WalkSession.from_dict(session_dict)
        answer  = interrupt(pending)

        path = session.get_path(pending["path_id"])
        path.pending_qa = {"question": pending["question"], "answer": str(answer)}

        return {"walk_session": session.to_dict(), "walk_pending_question": None}

    # ------------------------------------------------------------------
    # Node: walk_finish（簡化版 — 不再交接 main_agent，只寫回 thinking_process）
    # ------------------------------------------------------------------

    async def walk_finish(self, state: OrchestratorState) -> OrchestratorState:
        session = WalkSession.from_dict(state["walk_session"])

        lines = [
            f"· {p.current_label}（{'RC' if p.final_is_rc else '中繼概念'}）"
            for p in session.paths
        ]
        reply = (
            "這輪慢想探索完成，走到的位置：\n" + "\n".join(lines)
            + "\n\n已更新推薦排名，可以在 Personal Brain 看到更新後的候選。"
        )

        return OrchestratorState(
            messages=[AIMessage(content=reply)],
            phase="socratic",
            thinking_process=session.full_thinking_process(),   # ← 改這裡
            walk_session=session.to_dict(),
            walk_pending_question=None,
            attachments=[],
        )

    # ------------------------------------------------------------------
    # Private helpers（不變）
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_current_path(state: OrchestratorState):
        session   = WalkSession.from_dict(state["walk_session"])
        scheduler = WalkScheduler(session, state.get("walk_pointer", -1))
        path      = scheduler.next_path()
        return session, scheduler, path
