# -*- coding: utf-8 -*-
"""graph_service/walk_state.py — Random Walk Mode 的路徑狀態與排程器。

【變更說明 — LangGraph 化】
WalkScheduler._pointer 原本是「跨多次 run_turn() 呼叫、活在同一個 Python
物件生命週期裡」的隱含狀態，這在舊架構下沒問題，因為 routes_walk.py 把
(WalkSession, WalkScheduler) 這對 tuple 整個存在 app.state.walk_sessions
這個 process 內記憶體 dict 裡，同一個 WalkScheduler 實例會被重複拿出來用。

改用 LangGraph 之後，walk 的狀態改成隨 OrchestratorState 存進
checkpointer，每個 node 呼叫都是「讀 state → 算 → 回傳 state 更新」，
不會有同一個 WalkScheduler 物件跨呼叫存活這件事。因此 pointer 必須能夠
明確地被存進 state、下次呼叫時明確地被還原——不能再依賴「物件還活著」
這個假設。

WalkScheduler 現在接受一個可選的 pointer 建構參數（預設 -1，等同原本
行為），並透過 .pointer 屬性把當前值暴露出來，供呼叫端（orchestrator/
walk_nodes.py）讀寫進 OrchestratorState["walk_pointer"]。

其餘邏輯（PathState / WalkSession / DeadEndReason 等）完全不變。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════
# 基礎資料結構
# ══════════════════════════════════════════════════════════════

class PathStatus(str, Enum):
    """單條 path 的生命週期狀態。"""
    PENDING  = "pending"    # 尚未開始（尚未取得起點的 frontier）
    ACTIVE   = "active"     # 進行中，仍有 frontier 可推進
    DEAD_END = "dead_end"   # 已終止（無下游 或 達到 max_depth）


class DeadEndReason(str, Enum):
    NO_DOWNSTREAM = "no_downstream"
    MAX_DEPTH     = "max_depth"


@dataclass
class WalkNodeRef:
    """節點的最小引用，用於已走路徑的裁剪表示。

    is_rc 隨節點一起保存（而非只存在於 FrontierCandidate 上）：
    節點一旦離開 frontier、成為 visited 的一員，仍需要在之後判斷
    「path 是否終止在 RC 上」——不論終止原因是 no_downstream 還是
    max_depth，這個判斷都要成立，因此 is_rc 必須跟著節點本身走，
    不能只在 advance() 當下的呼叫端手動記錄一次（容易漏，見
    test_walk_state.py 最初版本踩到的坑）。
    """
    eid:   str
    label: str
    is_rc: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"eid": self.eid, "label": self.label, "is_rc": self.is_rc}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WalkNodeRef":
        return cls(eid=d["eid"], label=d["label"], is_rc=bool(d.get("is_rc", False)))


@dataclass
class FrontierCandidate:
    """當前 frontier 上的一個候選節點，供 LLM 判斷用（保留完整描述）。"""
    eid:        str
    label:      str
    definition: Optional[str] = None
    mechanism:  Optional[str] = None
    node_type:  Optional[str] = None   # 是否為 Resulting_Concept，供終止判斷用

    @property
    def is_rc(self) -> bool:
        return "resulting" in (self.node_type or "").lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eid": self.eid, "label": self.label,
            "definition": self.definition, "mechanism": self.mechanism,
            "node_type": self.node_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrontierCandidate":
        return cls(
            eid=d["eid"], label=d["label"],
            definition=d.get("definition"), mechanism=d.get("mechanism"),
            node_type=d.get("node_type"),
        )


@dataclass
class DecisionLogEntry:
    """一次 advance 的決策理由，並附上（若有）觸發這次推進前的問答。"""
    from_eid: str
    to_eid:   str
    reason:   str
    qa:       Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"from_eid": self.from_eid, "to_eid": self.to_eid,
                "reason": self.reason, "qa": self.qa}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionLogEntry":
        return cls(from_eid=d["from_eid"], to_eid=d["to_eid"],
                    reason=d["reason"], qa=d.get("qa"))


# ══════════════════════════════════════════════════════════════
# PathState — 單條 path 的完整狀態
# ══════════════════════════════════════════════════════════════

@dataclass
class PathState:
    """單條 random walk path。

    visited 是全量保留、不裁剪的 server-side state（供最終排名使用）；
    frontier 是「當前這一輪」的候選節點，每次 advance 後會被整個替換
    （不是 append），對應設計討論中「非選中 sibling 直接丟棄」的裁剪策略。
    """
    path_id:    int
    start_eid:  str
    start_label: str

    current_eid:   str = ""
    current_label: str = ""

    # 全量保留：依走訪順序排列的節點引用（含起點，不含尚未走到的 frontier）
    visited: List[WalkNodeRef] = field(default_factory=list)

    # 當前這一輪的候選節點（LLM 判斷用，advance 後整批替換）
    frontier: List[FrontierCandidate] = field(default_factory=list)

    status:     PathStatus = PathStatus.PENDING
    dead_end_reason: Optional[DeadEndReason] = None

    depth:      int = 0
    max_depth:  int = 8

    decision_log: List[DecisionLogEntry] = field(default_factory=list)
    pending_qa:   Optional[Dict[str, str]] = None     

    def __post_init__(self) -> None:
        if not self.current_eid:
            self.current_eid   = self.start_eid
            self.current_label = self.start_label
        if not self.visited:
            self.visited = [WalkNodeRef(eid=self.start_eid, label=self.start_label)]

    # ------------------------------------------------------------------
    # Frontier 管理
    # ------------------------------------------------------------------

    def set_frontier(self, candidates: List[FrontierCandidate]) -> None:
        """設定當前這一輪的候選節點（呼叫端已完成 sbert prefilter top-N 篩選）。

        若候選為空，直接標記 DEAD_END（no_downstream），呼叫端不需要
        另外判斷「frontier 是否為空」再手動呼叫 mark_dead_end。
        """
        self.frontier = candidates
        if not candidates:
            self.mark_dead_end(DeadEndReason.NO_DOWNSTREAM)
            return
        self.status = PathStatus.ACTIVE

    # ------------------------------------------------------------------
    # 狀態轉移
    # ------------------------------------------------------------------

    def advance(self, target_eid: str, reason: str = "") -> None:
        if self.status == PathStatus.DEAD_END:
            raise ValueError(f"path {self.path_id} 已終止，無法再 advance")

        target = next((c for c in self.frontier if c.eid == target_eid), None)
        if target is None:
            raise ValueError(
                f"target_eid='{target_eid}' 不在 path {self.path_id} 的 frontier 內"
            )

        self.decision_log.append(DecisionLogEntry(
            from_eid=self.current_eid, to_eid=target_eid,
            reason=reason, qa=self.pending_qa,
        ))
        self.pending_qa = None

        self.current_eid   = target.eid
        self.current_label = target.label
        self.visited.append(WalkNodeRef(eid=target.eid, label=target.label, is_rc=target.is_rc))
        self.depth += 1
        self.frontier = []

        if self.depth >= self.max_depth:
            self.mark_dead_end(DeadEndReason.MAX_DEPTH)
        else:
            self.status = PathStatus.PENDING

    def mark_dead_end(self, reason: DeadEndReason) -> None:
        self.status = PathStatus.DEAD_END
        self.dead_end_reason = reason
        self.frontier = []

    # ------------------------------------------------------------------
    # 序列化（供 OrchestratorState 存進 LangGraph checkpointer 使用）
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "path_id":         self.path_id,
            "start_eid":       self.start_eid,
            "start_label":     self.start_label,
            "current_eid":     self.current_eid,
            "current_label":   self.current_label,
            "visited":         [n.to_dict() for n in self.visited],
            "frontier":        [c.to_dict() for c in self.frontier],
            "status":          self.status.value,
            "dead_end_reason": self.dead_end_reason.value if self.dead_end_reason else None,
            "depth":           self.depth,
            "max_depth":       self.max_depth,
            "decision_log":    [e.to_dict() for e in self.decision_log],
            "pending_qa":      self.pending_qa,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PathState":
        obj = cls(
            path_id=d["path_id"],
            start_eid=d["start_eid"],
            start_label=d["start_label"],
            current_eid=d.get("current_eid", ""),
            current_label=d.get("current_label", ""),
            status=PathStatus(d.get("status", PathStatus.PENDING.value)),
            dead_end_reason=(
                DeadEndReason(d["dead_end_reason"]) if d.get("dead_end_reason") else None
            ),
            depth=d.get("depth", 0),
            max_depth=d.get("max_depth", 8),
        )
        # __post_init__ 已經根據 start_eid/current_eid 補了預設的
        # visited=[起點]；這裡有序列化資料就整批覆寫回真正的歷史紀錄。
        obj.visited      = [WalkNodeRef.from_dict(n) for n in d.get("visited", [])] or obj.visited
        obj.frontier      = [FrontierCandidate.from_dict(c) for c in d.get("frontier", [])]
        obj.decision_log  = [DecisionLogEntry.from_dict(e) for e in d.get("decision_log", [])]
        obj.pending_qa    = d.get("pending_qa")
        return obj

    # ------------------------------------------------------------------
    # Prompt context 組裝（裁剪後）
    # ------------------------------------------------------------------

    def visited_labels_only(self) -> List[str]:
        """已走路徑只留 label，供 prompt 呈現「已經走過哪些節點」的脈絡。"""
        return [n.label for n in self.visited]

    def to_prompt_block(self) -> dict:
        """組成餵給 LLM 的裁剪後 context 區塊。

        - visited_path：只有 label 的序列（不帶 description）
        - frontier：當前候選節點的完整資訊（LLM 據此判斷）
        """
        return {
            "path_id":       self.path_id,
            "visited_path":  self.visited_labels_only(),
            "depth":         self.depth,
            "frontier": [
                {
                    "eid":        c.eid,
                    "label":      c.label,
                    "definition": c.definition,
                    "mechanism":  c.mechanism,
                    "is_rc":      c.is_rc,
                }
                for c in self.frontier
            ],
        }

    # ------------------------------------------------------------------
    # 排名用輸出
    # ------------------------------------------------------------------

    def foundation_eids(self) -> List[str]:
        """回傳此 path 上、不含最終節點本身的 node id 清單。

        供 computeRCRanking 風格的 interScore 計算使用——最終節點若是 RC，
        不該把自己算進「foundation 命中 Personal Brain」的分母/分子裡。
        """
        if len(self.visited) <= 1:
            return []
        return [n.eid for n in self.visited[:-1]]

    @property
    def final_eid(self) -> Optional[str]:
        return self.current_eid if self.status == PathStatus.DEAD_END else None

    @property
    def final_is_rc(self) -> bool:
        """最終節點是否為 RC（僅在 DEAD_END 後有意義）。

        直接讀 visited 最後一個節點的 is_rc——不論終止原因是
        no_downstream 或 max_depth，is_rc 都是隨節點寫入 visited 時
        就已經確定的事實，不需要呼叫端在終止當下另外手動設定。
        """
        if self.status != PathStatus.DEAD_END or not self.visited:
            return False
        return self.visited[-1].is_rc

    def thinking_chain(self, last_n: Optional[int] = None) -> str:
        """把 visited + decision_log 組成 node --reason--> node 的鏈狀敘述。

        每個節點只出現一次；若某一步是在使用者回答問題後才推進的，
        問答會併進通往下一節點的那一段箭頭裡（不讓節點重複出現）。
        last_n：只取最近 N 步（decision_log 條目數），None 代表全部。
        """
        entries = self.decision_log[-last_n:] if last_n else self.decision_log
        if not entries:
            return self.visited[0].label if self.visited else self.start_label

        start_idx = max(len(self.visited) - len(entries) - 1, 0)
        parts = [self.visited[start_idx].label]
        for i, entry in enumerate(entries):
            node = self.visited[start_idx + 1 + i]
            seg = entry.reason
            if entry.qa:
                seg = f"(問：{entry.qa['question']}／答：{entry.qa['answer']}) {seg}"
            parts.append(f"--{seg}--> {node.label}")
        return " ".join(parts)
# ══════════════════════════════════════════════════════════════
# WalkSession — K 條 path 的集合
# ══════════════════════════════════════════════════════════════

@dataclass
class WalkSession:
    """一次 random walk 對話的完整狀態，供 session_store 序列化保存。"""
    session_id: str
    paths:      List[PathState] = field(default_factory=list)
    max_depth:  int = 8



    @classmethod
    def start(
        cls,
        session_id:    str,
        start_concepts: List[WalkNodeRef],
        max_depth:     int = 8,
    ) -> "WalkSession":
        paths = [
            PathState(
                path_id=i,
                start_eid=c.eid,
                start_label=c.label,
                max_depth=max_depth,
            )
            for i, c in enumerate(start_concepts)
        ]
        return cls(session_id=session_id, paths=paths, max_depth=max_depth)

    def full_thinking_process(self) -> str:
        """彙整所有 path 的思考鏈，供 OrchestratorState.thinking_process /
        SemanticPanel 的 Thinking Process 區塊顯示。"""
        return "\n".join(
            f"[path {p.path_id}] {p.thinking_chain()}" for p in self.paths
        )
    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------

    def get_path(self, path_id: int) -> PathState:
        return self.paths[path_id]

    def all_done(self) -> bool:
        return all(p.status == PathStatus.DEAD_END for p in self.paths)

    def active_or_pending_paths(self) -> List[PathState]:
        return [p for p in self.paths if p.status != PathStatus.DEAD_END]

    # ------------------------------------------------------------------
    # 排名用輸出
    # ------------------------------------------------------------------

    def all_foundation_eids(self) -> Dict[int, List[str]]:
        """path_id → foundation eid 清單，供前端新開的 ranking map 使用。"""
        return {p.path_id: p.foundation_eids() for p in self.paths}

    def final_rc_candidates(self) -> Dict[str, List[int]]:
        """final_eid（僅限確認為 RC 的終點）→ 命中它的 path_id 清單。

        多條 path 收斂到同一個 RC 時，這裡會回傳長度 >1 的 path_id 清單，
        供上層計算「收斂 bonus」使用（是否要加分屬於排名公式層的決定，
        這裡只負責提供收斂事實）。
        """
        result: Dict[str, List[int]] = {}
        for p in self.paths:
            if p.status == PathStatus.DEAD_END and p.final_eid and p.final_is_rc:
                result.setdefault(p.final_eid, []).append(p.path_id)
        return result


    # ------------------------------------------------------------------
    # 序列化（供 OrchestratorState 存進 LangGraph checkpointer 使用）
    # ------------------------------------------------------------------


    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "paths":      [p.to_dict() for p in self.paths],
            "max_depth":  self.max_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WalkSession":
        obj = cls(session_id=d["session_id"], max_depth=d.get("max_depth", 8))
        obj.paths = [PathState.from_dict(p) for p in d.get("paths", [])]
        return obj


# ══════════════════════════════════════════════════════════════
# WalkScheduler — 循序制：完整走完一條 path 才換下一條
# ══════════════════════════════════════════════════════════════

class WalkScheduler:
    """決定「現在該處理哪條 path」，不涉及 LLM 判斷。

    新版策略：永遠挑「清單中第一條尚未終止（DEAD_END）的 path」，
    也就是固定依 path_id 由小到大處理，把某條 path 完整走到終止
    （no_downstream 或 max_depth）才輪到下一條。效果：

      - 任一時刻最多只有一條 path 處於 ACTIVE/PENDING，
        pending_question 的歸屬永遠明確、不需要額外交叉比對。
      - 使用者體驗上更像「依序看完三個假設」，而不是三條線同時
        跳動——對「慢想」這種可解釋性優先的功能更合適。
      - 犧牲的是「三條 path 平行推進」帶來的總時長優化，但單條
        path 最多 max_depth 步、K 通常是 3，總步數可接受。

    pointer 仍保留（雖然循序制下它幾乎恆等於目前處理中 path 的
    path_id），維持與呼叫端既有慣例相容：

        scheduler = WalkScheduler(session, state.get("walk_pointer", -1))
        path = scheduler.next_path()
        ...
        return {"walk_pointer": scheduler.pointer, ...}
    """

    def __init__(self, session: WalkSession, pointer: int = -1) -> None:
        self._session = session
        self._pointer = pointer

    @property
    def pointer(self) -> int:
        """目前指標位置，呼叫端應在 node 回傳時把這個值存回 state。"""
        return self._pointer

    def next_path(self) -> Optional[PathState]:
        """回傳清單中第一條尚未終止的 path（依 path_id 由小到大）。

        找不到任何非 DEAD_END 的 path 時回傳 None（代表 walk 已全部
        結束）。與舊版的差異只在於不再使用 self._pointer 做環狀偏移
        搜尋起點——pointer 純粹是「上次處理到誰」的紀錄，不影響這裡
        的選取邏輯。
        """
        for path in self._session.paths:
            if path.status != PathStatus.DEAD_END:
                self._pointer = path.path_id
                return path
        return None
