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
    RUNNING          = "running"
    CONCEPT_RESOLVED = "concept_resolved"
    PATH_CONFIRMED   = "path_confirmed"


# ═══════════════════════════════════════════════════════════════
# ChatState
# ═══════════════════════════════════════════════════════════════

class ChatState(TypedDict, total=False):
    messages:       List[BaseMessage]
    agent_messages: List[BaseMessage]
    phase:          str
    query:          Optional[str]
    candidates:     Optional[List[dict]]
    selected_index: Optional[int]
    tools:          Optional[List[dict]]
    parameters:     Optional[List[dict]]
    intent_types:   Optional[List[str]]


# ═══════════════════════════════════════════════════════════════
# 公開輔助函數
# ═══════════════════════════════════════════════════════════════

def extract_selected_chain(state: ChatState) -> Optional[dict]:
    """從 state 取出已選定的候選 concept dict，未選定回傳 None。"""
    candidates: List[dict] = state.get("candidates") or []
    idx: Optional[int]     = state.get("selected_index")
    if idx is None or not candidates:
        return None
    if idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


def format_selected_chain(candidate: Optional[dict]) -> str:
    """將候選 concept 格式化為可讀字串，供 logging / debug 使用。"""
    if candidate is None:
        return "（未選定）"
    rc_name = candidate.get("rc_name") or "（未命名）"
    score   = candidate.get("score", "")
    score_s = f"（相似度 {score}）" if score else ""
    return f"選定分析概念：{rc_name}{score_s}"
