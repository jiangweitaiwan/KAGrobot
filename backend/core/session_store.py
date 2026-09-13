# -*- coding: utf-8 -*-
"""core/session_store.py — 使用者對話 session 的 Firestore 持久化備援。

背景問題：
  orchestrator/orchestrator_graph.py 的 LangGraph 用 MemorySaver()，
  純記憶體 checkpointer，process 重啟（部署更新／Cloud Run 冷啟動／
  scale-to-zero）就會整個遺失所有進行中的對話狀態，且完全不區分使用者。

此模組的角色：
  在每次 /chat 呼叫「結束後」，把 OrchestratorState 中「可安全序列化、
  足以恢復對話語意」的子集，額外備份一份到 Firestore：

      users/{uid}/sessions/{session_id}

  下次呼叫時，若 MemorySaver 對該 thread_id 已無任何記錄（代表遇到
  冷啟動或重啟），就從這裡讀回最後一次的狀態，重新塞進 LangGraph 的
  invoke_input 當作起點，讓對話可以延續，而不是被迫從頭開始。

刻意不做的事（設計取捨，見上一輪討論）：
  - 不試圖完整還原 LangGraph 內部 checkpoint（含工具呼叫中途狀態）。
    那需要實作完整的 BaseCheckpointSaver + serde，複雜度與風險都高，
    「分析對話中途因重啟而遺失一次工具呼叫的中間過程」屬於可接受的
    降級，之後若有需要可以升級成正式的 Firestore-backed
    BaseCheckpointSaver 取代 MemorySaver。
  - 還原 messages 時，只保留「純文字的人類/AI 對話」，捨棄任何帶有
    tool_calls 的 AIMessage——避免恢復後缺少對應的 tool response，
    導致下一輪呼叫 LLM API 時因為「懸空的 tool_call」而報錯。

【新增 — HTML Widget】
  save_session_snapshot 的 turn_messages 參數（由 api/routes/chat.py 與
  api/routes/routes_walk.py 組裝）現在可能在某則 AI turn message 上多帶
  一個 "widget" 欄位（{widget_id, title, url, created_at,
  expires_in_hours}），代表這輪對話顯示過一個互動元件。這裡刻意不特別
  解析或驗證這個欄位——turn_messages 本來就是「呼叫端已經組好的顯示用
  資料」，直接原樣併入 display_messages 陣列存檔即可，前端重播時依
  expires_in_hours + created_at 自行判斷是否過期（比照 AttachmentCard
  的 isAttachmentExpired 邏輯，見 api/routes/users.py::
  SessionDisplayMessage 的 widget 欄位定義）。不需要跟 pending_widget
  一樣「可續答」——重播模式下這個 widget 只是歷史紀錄，iframe 若還沒
  過期仍可正常互動，但即時試算（dispatch）與正常流程無異，提交後一樣
  走全新一輪 /chat；只是過期後前端顯示灰階，不呈現 iframe。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_db = None
_MAX_DISPLAY_MESSAGES = 120  # 前端渲染用的訊息上限，避免單一 session 文件無限膨脹

def _get_db():
    global _db
    if _db is None:
        from google.cloud import firestore
        _db = firestore.AsyncClient(project = "project-33725f55-5932-4674-9bd",database="kagrobot-state")
    return _db


# ── Message (de)serialization（保守版：只保留純文字對話）───────────

def _serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if getattr(m, "tool_calls", None):
            # 帶有 tool_calls 的 AIMessage 不安全還原，整則跳過
            continue
        if not isinstance(m, (HumanMessage, AIMessage)):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if not content.strip():
            continue
        out.append({"type": m.type, "content": content})
    return out


def _deserialize_messages(raw: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for item in raw or []:
        content = item.get("content", "")
        if item.get("type") == "human":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


# ── Public API ───────────────────────────────────────────────────

async def save_session_snapshot(
    uid: str,
    session_id: str,
    state: Dict[str, Any],
    *,
    turn_messages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    try:
        socrates_state = state.get("socrates_state") or {}
        now = int(time.time() * 1000)

        payload: Dict[str, Any] = {
            "phase":                     state.get("phase", "socratic"),
            "selected_rc_eid":           state.get("selected_rc_eid"),
            "selected_rc_label":         state.get("selected_rc_label"),
            "confirmed_rc_eid":          state.get("confirmed_rc_eid"),
            "skill_sandbox_id":          state.get("skill_sandbox_id"),
            "thinking_process":          state.get("thinking_process") or "",
            "socrates_phase":            socrates_state.get("phase"),
            "socrates_intent_types":     socrates_state.get("intent_types") or [],
            "socrates_context_summary":  socrates_state.get("context_summary") or "",
            "main_state_messages":       _serialize_messages(state.get("main_state") or []),
            # ← 新增：walk 的展示用狀態（唯讀重播，不代表可續答）
            "walk_session":              state.get("walk_session"),
            "walk_pending_question":     state.get("walk_pending_question"),
            "updated_at":                now,
        }

        db  = _get_db()
        ref = (
            db.collection("users").document(uid)
              .collection("sessions").document(session_id)
        )
        doc = await ref.get()

        if not doc.exists:
            payload["created_at"] = now

        if turn_messages:
            existing = (doc.to_dict() or {}).get("display_messages") or []
            payload["display_messages"] = (existing + turn_messages)[-_MAX_DISPLAY_MESSAGES:]

        await ref.set(payload, merge=True)
    except Exception as exc:
        print(f"[session_store] save_session_snapshot failed "
              f"(uid={uid}, session={session_id}): {exc}")


async def load_session_snapshot(
    uid:        str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """讀回最後一次備份的狀態，供 MemorySaver 遺失該 thread 時重新 seed。

    回傳 None 代表沒有備份，或讀取失敗——呼叫端應視同全新 session 處理。
    """
    try:
        db  = _get_db()
        ref = (
            db.collection("users").document(uid)
              .collection("sessions").document(session_id)
        )
        doc = await ref.get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        return {
            "phase":             data.get("phase", "socratic"),
            "selected_rc_eid":   data.get("selected_rc_eid"),
            "selected_rc_label": data.get("selected_rc_label"),
            "confirmed_rc_eid":  data.get("confirmed_rc_eid"),
            "skill_sandbox_id":  data.get("skill_sandbox_id"),
            "thinking_process":  data.get("thinking_process") or "", 
            "socrates_state": {
                "messages":        [],
                "agent_messages":  [],
                "phase":           data.get("socrates_phase", "running"),
                "intent_types":    data.get("socrates_intent_types") or [],
                "context_summary": data.get("socrates_context_summary") or "",
            },
            "main_state": _deserialize_messages(data.get("main_state_messages") or []),
        }
    except Exception as exc:
        print(f"[session_store] load_session_snapshot failed "
              f"(uid={uid}, session={session_id}): {exc}")
        return None

async def load_session_display(uid: str, session_id: str) -> Optional[Dict[str, Any]]:
    try:
        db  = _get_db()
        ref = (
            db.collection("users").document(uid)
              .collection("sessions").document(session_id)
        )
        doc = await ref.get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        return {
            "session_id":            session_id,
            "phase":                 data.get("phase", "socratic"),
            "display_messages":      data.get("display_messages") or [],
            "selected_rc_eid":       data.get("selected_rc_eid"),
            "selected_rc_label":     data.get("selected_rc_label"),
            "updated_at":            data.get("updated_at"),
            "created_at":            data.get("created_at"),
            "walk_session":          data.get("walk_session"),
            "walk_pending_question": data.get("walk_pending_question"),
        }
    except Exception as exc:
        print(f"[session_store] load_session_display failed "
              f"(uid={uid}, session={session_id}): {exc}")
        return None


async def list_sessions(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
    """列出使用者的對話 session（供「我的對話紀錄」列表使用）。"""
    try:
        from google.cloud import firestore

        db    = _get_db()
        query = (
            db.collection("users").document(uid).collection("sessions")
              .order_by("updated_at", direction=firestore.Query.DESCENDING)
              .limit(limit)
        )

        results: List[Dict[str, Any]] = []
        async for doc in query.stream():
            d = doc.to_dict() or {}
            results.append({
                "session_id": doc.id,
                "phase":      d.get("phase"),
                "summary":    d.get("socrates_context_summary", ""),
                "updated_at": d.get("updated_at"),
                "created_at": d.get("created_at"),
            })
        return results
    except Exception as exc:
        print(f"[session_store] list_sessions failed (uid={uid}): {exc}")
        return []
