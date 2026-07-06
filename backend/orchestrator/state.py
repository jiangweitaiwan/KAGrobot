# -*- coding: utf-8 -*-
"""orchestrator/state.py — LangGraph Orchestrator 狀態定義。"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    messages:          Annotated[List[BaseMessage], add_messages]
    phase:             str
    socrates_state:    Any          # ChatState（避免循環 import 使用 Any）
    main_state:        Annotated[List[BaseMessage], add_messages]
    handoff_task:      Optional[str]
    selected_path_id:  Optional[int]
    confirmed_path_id: Optional[int]
    all_graphs:        Optional[Dict[int, Any]]
    graph_context:     Optional[str]
    skill_sandbox_id:  Optional[str]   # 新增：per-session 保存的 e2b sandbox id    
