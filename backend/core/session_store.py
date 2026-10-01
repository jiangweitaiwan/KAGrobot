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

刻意不做的事（設計取捨）：
  - 不試圖完整還原 LangGraph 內部 checkpoint（含工具呼叫中途狀態）。
    那需要實作完整的 BaseCheckpointSaver + serde，複雜度與風險都高，
    「分析對話中途因重啟而遺失一次工具呼叫的中間過程」屬於可接受的
    降級，之後若有需要可以升級成正式的 Firestore-backed
    BaseCheckpointSaver 取代 MemorySaver。
  - 還原 messages 時，只保留「純文字的人類/AI 對話」，捨棄任何帶有
    tool_calls 的 AIMessage——避免恢復後缺少對應的 tool response，
    導致下一輪呼叫 LLM API 時因為「懸空的 tool_call」而報錯。

【HTML Widget】
  save_session_snapshot 的 turn_messages 參數可能在某則 AI turn message 上
  多帶一個 "widget" 欄位，這裡不解析、原樣併入 display_messages，
  前端重播時依 expires_in_hours + created_at 自行判斷是否過期。

【Phase 0 — Firestore client 統一】
  sessions 仍存放在 kagrobot-state 資料庫（既有資料不搬），但 client 改由
  core/firestore_db.get_state_db() 提供，不再自己建。

【Phase 2 — 會員中心】
  - title：使用者自訂的對話名稱（rename_session）。沒有 title 時前端退回
    顯示需求摘要。save_session_snapshot 用 merge=True，不會覆蓋 title。
  - delete_session / delete_all_sessions：刪除對話（後者供刪帳號使用）。

【RC Briefing】
  - rc_briefings.{eid}：該 session 內某 RC 最後一次生成的全局說明（含 context_hash），
    供多 worker handoff 與避免重複計費。
  - display_messages 可帶 briefing 欄（比照 widget），供歷史對話重播卡片。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.firestore_db import get_state_db as _get_db

_MAX_DISPLAY_MESSAGES = 120  # 前端渲染用的訊息上限，避免單一 session 文件無限膨脹


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


def _session_ref(uid: str, session_id: str):
    return (
        _get_db().collection("users").document(uid)
                 .collection("sessions").document(session_id)
    )


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
            # walk 的展示用狀態（唯讀重播，不代表可續答）
            "walk_session":              state.get("walk_session"),
            "walk_pending_question":     state.get("walk_pending_question"),
            "updated_at":                now,
        }

        ref = _session_ref(uid, session_id)
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
        doc = await _session_ref(uid, session_id).get()
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
        doc = await _session_ref(uid, session_id).get()
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

        query = (
            _get_db().collection("users").document(uid).collection("sessions")
              .order_by("updated_at", direction=firestore.Query.DESCENDING)
              .limit(limit)
        )

        results: List[Dict[str, Any]] = []
        async for doc in query.stream():
            d = doc.to_dict() or {}
            results.append({
                "session_id": doc.id,
                "title":      d.get("title", ""),
                "phase":      d.get("phase"),
                "summary":    d.get("socrates_context_summary", ""),
                "updated_at": d.get("updated_at"),
                "created_at": d.get("created_at"),
            })
        return results
    except Exception as exc:
        print(f"[session_store] list_sessions failed (uid={uid}): {exc}")
        return []


# ── 會員中心：改名 / 刪除 ──────────────────────────────────────────

async def rename_session(uid: str, session_id: str, title: str) -> bool:
    """回傳 False 代表找不到該對話。title 為空字串 = 清除自訂名稱。"""
    ref = _session_ref(uid, session_id)
    doc = await ref.get()
    if not doc.exists:
        return False
    await ref.set({"title": title}, merge=True)
    return True


async def delete_session(uid: str, session_id: str) -> None:
    await _session_ref(uid, session_id).delete()


async def delete_all_sessions(uid: str) -> None:
    """刪帳號用：分批刪除該使用者所有 sessions（Firestore 沒有原生的刪除整個 collection）。"""
    col = _get_db().collection("users").document(uid).collection("sessions")
    while True:
        count = 0
        async for doc in col.limit(100).stream():
            await doc.reference.delete()
            count += 1
        if count < 100:
            break


# ── RC Briefing（Firestore 持久化）────────────────────────────────

def _briefing_display_entry(
    *,
    eid: str,
    rc_label: str,
    ctx_hash: str,
    briefing: Dict[str, Any],
    labels: Dict[str, str],
    status: str = "ready",
) -> Dict[str, Any]:
    return {
        "eid": eid,
        "label": rc_label,
        "status": status,
        "context_hash": ctx_hash,
        "data": briefing,
        "labels": labels,
    }


def _display_has_briefing(
    messages: List[Dict[str, Any]], eid: str, ctx_hash: str,
) -> bool:
    for m in messages:
        b = m.get("briefing") or {}
        if b.get("eid") == eid and b.get("context_hash") == ctx_hash:
            return True
    return False


async def save_rc_briefing(
    uid: str,
    session_id: str,
    eid: str,
    ctx_hash: str,
    rc_label: str,
    briefing: Dict[str, Any],
    labels: Dict[str, str],
) -> None:
    """寫入 rc_briefings[eid]（同 session 內每 RC 保留最新一筆）。"""
    try:
        now = int(time.time() * 1000)
        ref = _session_ref(uid, session_id)
        doc = await ref.get()
        data = doc.to_dict() or {}
        briefings = dict(data.get("rc_briefings") or {})
        briefings[eid] = {
            "context_hash": ctx_hash,
            "rc_label":     rc_label,
            "briefing":     briefing,
            "labels":       labels,
            "updated_at":   now,
        }
        payload: Dict[str, Any] = {"rc_briefings": briefings, "updated_at": now}
        if not doc.exists:
            payload["created_at"] = now
        await ref.set(payload, merge=True)
    except Exception as exc:
        print(f"[session_store] save_rc_briefing failed "
              f"(uid={uid}, session={session_id}, eid={eid}): {exc}")


async def load_rc_briefing_for_context(
    uid: str,
    session_id: str,
    eid: str,
    ctx_hash: str,
) -> Optional[Dict[str, Any]]:
    """精確比對 context_hash；命中回傳 {briefing, labels}（briefing 為 dict）。"""
    try:
        doc = await _session_ref(uid, session_id).get()
        if not doc.exists:
            return None
        entry = (doc.to_dict() or {}).get("rc_briefings", {}).get(eid)
        if not entry or entry.get("context_hash") != ctx_hash:
            return None
        return {"briefing": entry["briefing"], "labels": entry.get("labels") or {}}
    except Exception as exc:
        print(f"[session_store] load_rc_briefing_for_context failed "
              f"(uid={uid}, session={session_id}): {exc}")
        return None


async def load_rc_briefing_latest(
    uid: str,
    session_id: str,
    eid: str,
) -> Optional[Dict[str, Any]]:
    """main_agent handoff：不需 context_hash，取該 RC 在 session 內最後存檔。"""
    try:
        doc = await _session_ref(uid, session_id).get()
        if not doc.exists:
            return None
        entry = (doc.to_dict() or {}).get("rc_briefings", {}).get(eid)
        if not entry:
            return None
        return {"briefing": entry["briefing"], "labels": entry.get("labels") or {}}
    except Exception as exc:
        print(f"[session_store] load_rc_briefing_latest failed "
              f"(uid={uid}, session={session_id}): {exc}")
        return None


async def append_briefing_display_message(
    uid: str,
    session_id: str,
    *,
    eid: str,
    rc_label: str,
    ctx_hash: str,
    briefing: Dict[str, Any],
    labels: Dict[str, str],
) -> None:
    """在 display_messages 追加一則 briefing 卡片（同 eid+context_hash 去重）。"""
    try:
        now = int(time.time() * 1000)
        ref = _session_ref(uid, session_id)
        doc = await ref.get()
        data = doc.to_dict() or {}
        existing = list(data.get("display_messages") or [])
        if _display_has_briefing(existing, eid, ctx_hash):
            return
        row: Dict[str, Any] = {
            "role":        "ai",
            "content":     "",
            "attachments": [],
            "created_at":  now,
            "briefing":    _briefing_display_entry(
                eid=eid, rc_label=rc_label, ctx_hash=ctx_hash,
                briefing=briefing, labels=labels, status="ready",
            ),
        }
        payload: Dict[str, Any] = {
            "display_messages": (existing + [row])[-_MAX_DISPLAY_MESSAGES:],
            "updated_at":       now,
        }
        if not doc.exists:
            payload["created_at"] = now
        await ref.set(payload, merge=True)
    except Exception as exc:
        print(f"[session_store] append_briefing_display_message failed "
              f"(uid={uid}, session={session_id}): {exc}")


async def mark_briefing_adopted_in_display(
    uid: str,
    session_id: str,
    eid: str,
) -> None:
    """採用方案後，把 display_messages 內該 RC 的 briefing 標記為 adopted。"""
    try:
        ref = _session_ref(uid, session_id)
        doc = await ref.get()
        if not doc.exists:
            return
        data = doc.to_dict() or {}
        messages = list(data.get("display_messages") or [])
        changed = False
        for m in messages:
            b = m.get("briefing")
            if b and b.get("eid") == eid and b.get("status") != "adopted":
                b["status"] = "adopted"
                changed = True
        if not changed:
            return
        await ref.set({
            "display_messages": messages,
            "updated_at":       int(time.time() * 1000),
        }, merge=True)
    except Exception as exc:
        print(f"[session_store] mark_briefing_adopted_in_display failed "
              f"(uid={uid}, session={session_id}): {exc}")
