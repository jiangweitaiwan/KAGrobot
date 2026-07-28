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



from orchestrator.state     import OrchestratorState
from graph_service.bfs_coordinator import BFSCoordinator   # ← 新增 import

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
        socrates_bot:        SocratesChatBot,
        main_agent:          MainAgent,
        skill_loader:        SkillLoader,
        sandbox_provisioner: "SkillSandboxProvisioner",
        coordinator:         BFSCoordinator,
    ) -> None:
        self._socrates_bot        = socrates_bot
        self._main_agent          = main_agent
        self._skill_loader        = skill_loader
        self._sandbox_provisioner = sandbox_provisioner
        self._coordinator         = coordinator


    # ------------------------------------------------------------------
    # Node: socratic
    # ------------------------------------------------------------------

    async def socratic(self, state: OrchestratorState) -> OrchestratorState:
        messages        = state.get("messages") or []
        user_msg        = self._last_human_content(messages)
        selected_rc_eid = state.get("selected_rc_eid")

        if not user_msg:
            return {}

        old_socrates_state = state.get("socrates_state") or SocratesChatBot.init_state()

        if selected_rc_eid is not None:
            new_socrates_state = self._socrates_bot.confirm_path(old_socrates_state)
            return OrchestratorState(
                messages       = [AIMessage(content="路徑已確認，開始執行分析…")],
                phase          = "path_confirmed",
                socrates_state = new_socrates_state,
                # 進入 main_agent 前的過渡訊息不帶任何附件；
                # 明確清空，避免前一輪 main_agent 產生的 attachments
                # 殘留在 checkpoint state 裡被這一輪誤附加顯示。
                attachments    = [],
            )

        new_socrates_state = self._socrates_bot.process_message(old_socrates_state, user_msg)
        reply_text = self._socrates_bot.get_last_reply(new_socrates_state)

        return OrchestratorState(
            messages       = [AIMessage(content=reply_text)],
            phase          = "socratic",
            socrates_state = new_socrates_state,
            # 同上：socratic 階段的回覆不含 attachments，明確重置，
            # 避免使用者切回上一個已確認路徑再問問題時，看到舊圖表重複出現。
            attachments    = [],
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
                attachments=[],
            )

        selected_rc_eid   = state.get("selected_rc_eid")
        selected_rc_label = state.get("selected_rc_label") or ""
        confirmed_rc_eid  = state.get("confirmed_rc_eid")

        # skill 綁定的是 RC，不是 path——同一個 RC 不管走哪條 path
        # 到達，都是同一顆 skill，不該觸發 sandbox 重建。
        is_reselect = (
            selected_rc_eid is not None
            and selected_rc_eid != confirmed_rc_eid
        )

        # skill_loader.load() 內部是同步 GCS 下載 + zip 解壓，避免卡住 event loop
        skill = await asyncio.to_thread(self._load_skill, selected_rc_label)

        # 重新選路時捨棄舊 sandbox：best-effort 通知 e2b 直接 kill，
        # 失敗（sandbox 已過期/不存在）不影響流程，純粹避免堆積孤兒 sandbox
        old_sandbox_id = state.get("skill_sandbox_id")
        if is_reselect and old_sandbox_id:
            await asyncio.to_thread(self._sandbox_provisioner.kill, old_sandbox_id)

        sandbox_id = None if is_reselect else old_sandbox_id

        if is_reselect:
            user_msg        = socrates_state.get("query") or "請開始執行分析。"
            history:  List[BaseMessage] = []
            reselect_notice = AIMessage(
                content=f"已切換至「{selected_rc_label or selected_rc_eid}」，重新開始分析。"
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
            confirmed_rc_eid  = (
                selected_rc_eid if is_reselect else confirmed_rc_eid
            ),
            skill_sandbox_id  = result.get("sandbox_id"),
            attachments       = result.get("attachments") or [],
        )





    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------



    def _load_skill(self, rc_label: str):
        if not rc_label:
            return None
        skill = self._skill_loader.load(rc_label)
        if skill:
            print(f"[OrchestratorNodes] skill loaded: '{skill.rc_name}' ")
        else:
            print(f"[OrchestratorNodes] no skill for rc='{rc_label}', "
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