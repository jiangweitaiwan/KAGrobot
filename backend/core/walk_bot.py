# -*- coding: utf-8 -*-
"""core/walk_bot.py — Random Walk Mode 的單輪 LLM 決策迴圈。

設計原則（對應討論定案）：
  - 一次 LLM 呼叫只處理「scheduler 選中的那一條 path」，不是 socrates_bot.py
    那種多輪 agent↔tool_executor 迴圈——因此這裡不需要 LangGraph，
    單一 LLM 呼叫 + 純 Python 判斷輸出即可，結構比 SocratesChatBot 簡單。
  - LLM 只有兩種輸出形式：
      1. 呼叫 advance_path(target_eid, reason) → 推進這條 path
      2. 純文字（不呼叫工具）→ 視為要問使用者的問題，本輪 path 保持 ACTIVE
         （frontier 不清空），等使用者回答後，下一次 run_turn() 用同一個
         path 重新決策（相當於 resume，不是重新從起點問一次）。
  - advance_path 的 target_eid 用 Literal 動態限制在「當前 frontier 的
    eid 集合」內（見 _build_advance_tool），讓 LLM 在 schema 層面就只能
    選合法選項；PathState.advance() 內部仍會再驗證一次，雙重保險。
  - LLM 呼叫本身透過 llm_caller 注入（Callable[[messages, tool], AIMessage]），
    讓這支模組完全不依賴真的打 API，可以用假 LLM 函式做整合測試
    （對應 walk_state.py 已經用假資料測過 path 狀態機本身）。
  - frontier 的取得同樣透過 frontier_fetcher 注入，介面為
    (current_eid, context_query) -> List[FrontierCandidate]；
    帶上 context_query 是因為排序候選需要依語意脈絡，不能只靠 eid。

【併入 orchestrator 後的狀態】
  _SYSTEM_PROMPT_TEMPLATE / _build_advance_tool / _format_frontier 這三個
  仍是 orchestrator/walk_nodes.py::WalkOrchestratorNodes 直接重用的部分
  （prompt 組裝與 schema 動態建構的邏輯只寫一份）。

  WalkTurnRunner 這個 class 本身已被 orchestrator/walk_nodes.py 的五個
  LangGraph node 取代（run_turn() 的三種分支對應 walk_fetch_frontier /
  walk_decide / walk_ask 三個 node），只剩 api/routes/routes_walk.py 的
  舊版實作還在用它。等 routes_walk.py 遷移到透過 orchestrator.astream()
  驅動後，這個 class 就可以整個刪除，屆時本檔案只保留上面三個共用的
  prompt/schema 組裝函式。

與既有程式碼的介接點：
  - frontier_fetcher 的真正實作見 graph_service/walk_frontier.py 的
    make_frontier_fetcher()——接 AdjacencyCache 取一跳下游鄰居、批次
    補齊 definition/mechanism、依語意脈絡做 sbert prefilter。
  - llm_caller 的真正實作：比照 core/main_agent.py 的
    `self._base_llm.bind_tools(tools)` 模式，用 core/google_llm.py 的
    create_chat_google_generative_ai 建立底層 LLM。
  - session.context_summary 的角色與 core/semantic_context.py 的
    SemanticContext.summary 一致，最終應該併入同一份 embedding_text，
    供 CandidateRetriever / FoundationRecommender 做語意排名。
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, create_model

from graph_service.walk_state import (
    FrontierCandidate, PathState, PathStatus, WalkScheduler, WalkSession,
)

# ══════════════════════════════════════════════════════════════
# System prompt
# ══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    你是一位知識圖譜探索助手，正在帶領一條推理路徑，從一個起點概念
    一步步往下游（downstream）走，最終希望走到一個 Resulting Concept（RC）。

    ## 目前狀態
    這條路徑（path_id={path_id}）已經走過：{visited_path}

    ## 目前可選的下一步（frontier）
    {frontier_desc}

    ## 對話累積的語意脈絡
    {context_summary}

    ## 你的任務
    從上方 frontier 中選一個最合理的下一步，呼叫 advance_path 工具，
    並用一句話（15 字以內的繁體中文）說明為什麼選這個方向。

    如果 frontier 選項都無法判斷該往哪走——例如需要使用者提供關鍵資訊
    才能決定方向——不要呼叫工具，直接用繁體中文向使用者提出一個
    簡短、具體的問題。使用者回答後，你會在下一輪重新看到同樣的
    frontier 並可以再次決策。
""")


def _format_frontier(frontier: List[FrontierCandidate]) -> str:
    lines = []
    for c in frontier:
        tag = "（RC）" if c.is_rc else ""
        desc = c.definition or c.mechanism or "（無描述）"
        lines.append(f"- eid={c.eid} | {c.label}{tag} | {desc}")
    return "\n".join(lines) if lines else "（無候選）"


# ══════════════════════════════════════════════════════════════
# 動態 advance_path 工具（target_eid 限制在當前 frontier 內）
# ══════════════════════════════════════════════════════════════

def _build_advance_tool(path: PathState) -> type[BaseModel]:
    """為當前 path 動態建立 advance_path 的 pydantic schema。

    target_eid 用 Literal 限制在「這一輪 frontier 的 eid 集合」內，
    讓 LLM 在 schema 層面就只能選合法選項（多數模型會遵守 enum 限制），
    PathState.advance() 內部仍會再驗證一次，不因為 schema 限制就省略。
    """
    eids = tuple(c.eid for c in path.frontier)
    if not eids:
        raise ValueError(f"path {path.path_id} 的 frontier 為空，不應呼叫 _build_advance_tool")

    from typing import Literal  # noqa: PLC0415  local import：Literal 需要動態 eids

    return create_model(
        "advance_path",
        __doc__=(
            "推進目前這條探索路徑到 frontier 中選定的節點。"
            "target_eid 必須是目前 frontier 清單裡的其中一個 eid，"
            "reason 是一句話（15 字以內）說明選擇理由。"
        ),
        target_eid=(Literal[eids], Field(description="選定的下一個節點 eid，必須來自目前 frontier")),
        reason=(str, Field(description="一句話說明為什麼選這個方向（15 字以內，繁體中文）")),
    )


# ══════════════════════════════════════════════════════════════
# Turn 結果
# ══════════════════════════════════════════════════════════════

@dataclass
class WalkTurnResult:
    """run_turn() 的回傳結果，供呼叫端（之後的 SSE 層）轉譯成 event。"""
    kind: str
    # kind 的可能值：
    #   "frontier_ready"   — 剛取得某條 path 的 frontier，尚未做決策（下一輪才問 LLM）
    #   "path_terminated"  — 某條 path 剛終止（no_downstream / max_depth）
    #   "advanced"         — LLM 決策完成，path 已推進一步
    #   "question"         — LLM 選擇不推進，向使用者提問
    #   "walk_complete"    — 所有 path 皆已終止

    path_id:      Optional[int] = None
    from_eid:     Optional[str] = None   # advanced 事件用：推進前的節點 eid
    target_eid:   Optional[str] = None
    target_label: Optional[str] = None
    reason:       Optional[str] = None
    question:     Optional[str] = None
    dead_end_reason: Optional[str] = None
    ranking_hint: Optional[dict] = None
    # frontier_ready 事件用：讓前端可以預先顯示「候選節點」的虛線預覽，
    # 呼應 KnowledgeGraph.tsx 既有 foundation dashed-line 的視覺語言。
    frontier: Optional[List[dict]] = None


# ══════════════════════════════════════════════════════════════
# WalkTurnRunner（legacy — 見檔頭「併入 orchestrator 後的狀態」說明）
# ══════════════════════════════════════════════════════════════

class WalkTurnRunner:
    """對單一 WalkSession 執行「一輪 = 一條 path 的一次決策」。

    Parameters
    ----------
    frontier_fetcher : Callable[[str, str], List[FrontierCandidate]]
        傳入 (current_eid, context_query)，回傳已完成 sbert prefilter
        top-N 篩選的候選清單。context_query 是 session 目前累積的語意
        脈絡（見 WalkSession.context_summary），供排序候選使用。
        注入而非直接依賴 AdjacencyCache，方便測試與之後替換資料源。
    llm_caller : Callable[[str, type[BaseModel]], Any]
        傳入 (system_prompt, advance_tool_schema)，回傳一個物件，
        該物件需暴露 `.tool_calls`（list，可為空）與 `.content`（str）
        兩個屬性 —— 與 langchain AIMessage 的介面一致，因此正式串接時
        可以直接把 `llm.bind_tools([tool]).invoke(...)` 包一層當作
        llm_caller 使用；測試時則可以塞一個回傳 SimpleNamespace 的假函式。
    """

    def __init__(
        self,
        frontier_fetcher: Callable[[str, str], List[FrontierCandidate]],
        llm_caller:       Callable[[str, type], Any],
    ) -> None:
        self._frontier_fetcher = frontier_fetcher
        self._llm_caller       = llm_caller

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_turn(
        self,
        session:   WalkSession,
        scheduler: WalkScheduler,
    ) -> WalkTurnResult:
        """跑一輪。呼叫端負責在 kind == "question" 時暫停、等使用者回答，
        下一次呼叫 run_turn() 時 scheduler 會再選到同一條 path（因為它
        仍是 ACTIVE、frontier 未清空），該輪會重新問一次 LLM。
        """
        if session.all_done():
            return WalkTurnResult(kind="walk_complete", ranking_hint=session.all_foundation_eids())

        path = scheduler.next_path()
        if path is None:
            return WalkTurnResult(kind="walk_complete", ranking_hint=session.all_foundation_eids())

        if path.status == PathStatus.PENDING:
            return self._fetch_frontier(session, path)

        if path.status == PathStatus.ACTIVE:
            return self._decide(session, path)

        # 理論上不會到這裡（DEAD_END 的 path 不會被 scheduler 選中）
        raise RuntimeError(f"unexpected path status: {path.status}")

    # ------------------------------------------------------------------
    # Step 1：取得 frontier（不涉及 LLM）
    # ------------------------------------------------------------------

    def _fetch_frontier(self, session: WalkSession, path: PathState) -> WalkTurnResult:
        candidates = self._frontier_fetcher(path.current_eid, session.context_summary)
        path.set_frontier(candidates)

        if path.status == PathStatus.DEAD_END:
            return WalkTurnResult(
                kind="path_terminated", path_id=path.path_id,
                dead_end_reason=(path.dead_end_reason.value if path.dead_end_reason else None),
            )
        return WalkTurnResult(
            kind="frontier_ready", path_id=path.path_id,
            frontier=[
                {"eid": c.eid, "label": c.label, "is_rc": c.is_rc}
                for c in path.frontier
            ],
        )

    # ------------------------------------------------------------------
    # Step 2：LLM 決策
    # ------------------------------------------------------------------

    def _decide(self, session: WalkSession, path: PathState) -> WalkTurnResult:
        tool_schema = _build_advance_tool(path)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            path_id=path.path_id,
            visited_path=" → ".join(path.visited_labels_only()),
            frontier_desc=_format_frontier(path.frontier),
            context_summary=session.context_summary or "（尚無累積脈絡）",
        )

        ai_msg = self._llm_caller(system_prompt, tool_schema)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            question = getattr(ai_msg, "content", "") or ""
            return WalkTurnResult(kind="question", path_id=path.path_id, question=question)

        call   = tool_calls[0]
        args   = call.get("args", {})
        target_eid = args.get("target_eid", "")
        reason     = args.get("reason", "")
        from_eid   = path.current_eid   # 推進前先記下，advance() 之後 current_eid 就變了

        # PathState.advance() 內部會再次驗證 target_eid 落在 frontier 內，
        # 即使上方 Literal schema 已限制過，這裡不因此省略防禦。
        path.advance(target_eid, reason=reason)
        session.record_decision(reason)

        return WalkTurnResult(
            kind="advanced", path_id=path.path_id,
            from_eid=from_eid, target_eid=target_eid,
            target_label=path.current_label, reason=reason,
        )
