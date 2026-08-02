# -*- coding: utf-8 -*-
"""orchestrator/state.py — LangGraph Orchestrator 狀態定義。"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    messages:          Annotated[List[BaseMessage], add_messages]
    phase:             str
    socrates_state:    Any
    main_state:        Annotated[List[BaseMessage], add_messages]
    handoff_task:      Optional[str]
    selected_rc_eid:   Optional[str]
    selected_rc_label: Optional[str]
    confirmed_rc_eid:  Optional[str]
    skill_sandbox_id:  Optional[str]
    attachments:       List[Dict[str, Any]]
    # 已驗證的 Firebase uid（見 api/routes/chat.py），由呼叫端在第一次
    # invoke 時放進 invoke_input；LangGraph 節點回傳值中若未包含此 key，
    # 該 channel 值會維持不變，因此不需要在 nodes.py 每個 return 都重複
    # 帶上 uid，只需要在 main_agent 節點讀取時 state.get("uid") 即可。
    uid:               Optional[str]
