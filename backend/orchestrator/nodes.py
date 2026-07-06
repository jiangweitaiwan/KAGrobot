# -*- coding: utf-8 -*-
"""orchestrator/nodes.py — LangGraph 節點邏輯。

OrchestratorNodes 將 socratic_node 與 main_agent_node 封裝為
instance method，所有外部依賴在建構時注入，方便測試替換。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from core.socrates_bot import SocratesChatBot
from core.chat_state   import ConversationPhase

from core.main_agent        import MainAgent
from core.skill_loader      import SkillLoader
from graph_service.expander  import GraphExpander
from graph_service.context_builder import GraphContextBuilder
from graph_service.projector import GraphProjector
from orchestrator.state     import OrchestratorState


class OrchestratorNodes:
    """封裝 LangGraph 兩個主要節點的業務邏輯。

    Parameters
    ----------
    socrates_bot     : SocratesChatBot
    main_agent       : MainAgent
    graph_expander   : GraphExpander
    context_builder  : GraphContextBuilder
    skill_loader     : SkillLoader
    """

    def __init__(
        self,
        socrates_bot:    SocratesChatBot,
        main_agent:      MainAgent,
        graph_expander:  GraphExpander,
        context_builder: GraphContextBuilder,
        skill_loader:    SkillLoader,
    ) -> None:
        self._socrates_bot    = socrates_bot
        self._main_agent      = main_agent
        self._graph_expander  = graph_expander
        self._context_builder = context_builder
        self._skill_loader    = skill_loader
        self._projector       = GraphProjector()

    # ------------------------------------------------------------------
    # Node: socratic
    # ------------------------------------------------------------------

    async def socratic(
        self, state: OrchestratorState
    ) -> OrchestratorState:
        messages         = state.get("messages") or []
        user_msg         = self._last_human_content(messages)
        selected_path_id = state.get("selected_path_id")
        current_phase    = state.get("phase") or "socratic"

        if not user_msg:
            return {}

        old_socrates_state = (
            state.get("socrates_state") or SocratesChatBot.init_state()
        )

        # 前端已選定路徑 → 確認並切換至 main_agent
        if selected_path_id is not None and current_phase == "concept_resolved":
            new_socrates_state = self._socrates_bot.confirm_path(old_socrates_state)
            return OrchestratorState(
                messages       = [AIMessage(content="路徑已確認，開始執行分析…")],
                phase          = "path_confirmed",
                socrates_state = new_socrates_state,
            )

        graph_context = (
            state.get("graph_context")
            if current_phase == "concept_resolved"
            else None
        )

        new_socrates_state = self._socrates_bot.process_message(
            old_socrates_state,
            user_msg,
            graph_context=graph_context,
        )

        inner_phase = new_socrates_state.get("phase")
        reply_text  = self._socrates_bot.get_last_reply(new_socrates_state)

        if inner_phase == ConversationPhase.CONCEPT_RESOLVED.value:
            candidates   = new_socrates_state.get("candidates") or []
            intent_types = new_socrates_state.get("intent_types") or []

            all_graphs = await self._graph_expander.expand_all(
                candidates, intent_types=intent_types or None
            )
            new_graph_context = self._context_builder.build(all_graphs)

            return OrchestratorState(
                messages       = [AIMessage(content=reply_text)],
                phase          = "concept_resolved",
                socrates_state = new_socrates_state,
                all_graphs     = all_graphs,
                graph_context  = new_graph_context,
            )

        return OrchestratorState(
            messages       = [AIMessage(content=reply_text)],
            phase          = current_phase,
            socrates_state = new_socrates_state,
        )

    # ------------------------------------------------------------------
    # Node: main_agent
    # ------------------------------------------------------------------


    async def main_agent(
        self, state: OrchestratorState
    ) -> OrchestratorState:
        socrates_state = state.get("socrates_state")
        if not socrates_state:
            return OrchestratorState(
                messages=[AIMessage(
                    content="目前尚未完成需求釐清，請先重新描述你的需求。"
                )],
                phase="socratic",
            )

        selected_path_id  = state.get("selected_path_id")
        confirmed_path_id = state.get("confirmed_path_id")
        all_graphs        = state.get("all_graphs")

        is_reselect = (
            selected_path_id is not None
            and selected_path_id != confirmed_path_id
        )

        selected_path = self._find_path(all_graphs, selected_path_id)
        skill         = self._load_skill(selected_path)

        # 重新選路時捨棄舊 sandbox：新 skill 需要全新的隔離環境，
        # 不應延用上一個分析方法遺留的變數/檔案狀態。
        sandbox_id = None if is_reselect else state.get("skill_sandbox_id")

        if is_reselect:
            user_msg        = socrates_state.get("query") or "請開始執行分析。"
            history:  List[BaseMessage] = []
            path_label      = (selected_path or {}).get("final_label") \
                              or f"路徑 {selected_path_id}"
            reselect_notice = AIMessage(
                content=f"已切換至新路徑「{path_label}」，重新開始分析。"
            )
        else:
            user_msg        = self._last_human_content(
                state.get("messages") or []
            )
            history         = list(state.get("main_state") or [])
            reselect_notice = None

        if not user_msg:
            return {}

        result = await asyncio.to_thread(
            self._main_agent.chat,
            user_msg,
            socrates_state,
            history,
            skill=skill,
            sandbox_id=sandbox_id,
        )

        output_messages: List[BaseMessage] = []
        if reselect_notice:
            output_messages.append(reselect_notice)
        output_messages.append(AIMessage(content=result["content"]))

        return OrchestratorState(
            messages          = output_messages,
            phase             = "main_agent",
            socrates_state    = socrates_state,
            handoff_task      = None,
            main_state        = result.get("messages") or [],
            confirmed_path_id = (
                selected_path_id if is_reselect else confirmed_path_id
            ),
            skill_sandbox_id  = result.get("sandbox_id"),
        )





    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_path(
        self,
        all_graphs:      Optional[Dict[int, Any]],
        selected_path_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if all_graphs is None or selected_path_id is None:
            return None
        for g in all_graphs.values():
            if not g:
                continue
            for path in g.get("paths", []):
                if path.get("path_id") == selected_path_id:
                    return path
        return None

    def _load_skill(self, selected_path: Optional[Dict[str, Any]]):
        if not selected_path:
            return None
        final_rc_name = selected_path.get("final_label") or ""
        if not final_rc_name:
            return None
        skill = self._skill_loader.load(final_rc_name)
        if skill:
            print(f"[OrchestratorNodes] skill loaded: '{skill.rc_name}' "
                  f"(tools={len(skill.tools)})")
        else:
            print(f"[OrchestratorNodes] no skill for rc='{final_rc_name}', "
                  f"using fallback")
        return skill

    @staticmethod
    def _last_human_content(
        messages: List[BaseMessage],
    ) -> Optional[str]:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None
