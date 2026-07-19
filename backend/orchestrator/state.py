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
    selected_path_id:  Optional[int]
    selected_rc_eid:   Optional[str]
    selected_rc_label: Optional[str]
    confirmed_path_id: Optional[int]
    confirmed_rc_eid:  Optional[str]
    skill_sandbox_id:  Optional[str]