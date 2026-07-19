# -*- coding: utf-8 -*-
"""core/chat_state.py — 對話狀態定義與輔助函數。

ConversationPhase : 對話階段枚舉
ChatState         : SocratesChatBot 的對話狀態 TypedDict
extract_selected_chain() : 從 state 取出已選定的候選
format_selected_chain()  : 將候選格式化為可讀字串
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage


# ═══════════════════════════════════════════════════════════════
# ConversationPhase
# ═══════════════════════════════════════════════════════════════

class ConversationPhase(Enum):
    RUNNING        = "running"
    PATH_CONFIRMED = "path_confirmed"


# ═══════════════════════════════════════════════════════════════
# ChatState
# ═══════════════════════════════════════════════════════════════

class ChatState(TypedDict, total=False):
    messages:        List[BaseMessage]
    agent_messages:  List[BaseMessage]
    phase:           str
    context_summary: Optional[str]

# ═══════════════════════════════════════════════════════════════
# 公開輔助函數
# ═══════════════════════════════════════════════════════════════
