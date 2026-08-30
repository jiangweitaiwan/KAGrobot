# -*- coding: utf-8 -*-
"""graph_service/walk_state.py — Random Walk Mode 的路徑狀態與排程器。

職責：
  - PathState：單條 path 的完整狀態（visited 全量保留、frontier 當輪候選、
    決策理由 log），並提供「裁剪後 prompt context」的組裝方法。
  - WalkSession：K 條 path 的集合 + 全域設定（max_depth 等）。
  - WalkScheduler：round-robin 排程「這一輪換哪條 path」，跳過已終止的 path。

刻意不含任何 LLM / SSE / Neo4j 依賴——這一層只處理「path 狀態機」本身，
確保這份邏輯可以用假資料獨立測試，之後才接上 tool-loop 與串流層。

與現有程式碼的介接點（之後串接時）：
  - frontier 的取得：呼叫端應透過 BFSCoordinator.downstream_async() 或
    AdjacencyCache.get_downstream_neighbors() 取得下一輪候選，再經過
    sbert prefilter（比照 api/routes/concept.py 的 _sbert_prefilter）
    砍到 top-N 後，餵進 PathState.set_frontier()。
  - 最終排名：WalkSession.all_foundation_eids() 回傳「每條 path 上、
    不含最終節點本身」的 node id 清單，供前端 computeRCRanking 風格的
    公式使用（interScore 計算方式與 DetailPanel.tsx 一致）。
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


@dataclass
class DecisionLogEntry:
    """一次 advance 的決策理由，摺進 context_summary 供可解釋性使用。"""
    from_eid: str
    to_eid:   str
    reason:   str   # LLM 給的一句話理由（例如「使用者提到通膨，這個方向與通膨機制相關」）


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
        """推進到 frontier 中指定的節點，frontier 整批清空。

        Raises
        ------
        ValueError : target_eid 不在當前 frontier 內，或 path 已終止
        """
        if self.status == PathStatus.DEAD_END:
            raise ValueError(f"path {self.path_id} 已終止，無法再 advance")

        target = next((c for c in self.frontier if c.eid == target_eid), None)
        if target is None:
            raise ValueError(
                f"target_eid='{target_eid}' 不在 path {self.path_id} 的 frontier 內"
            )

        self.decision_log.append(DecisionLogEntry(
            from_eid=self.current_eid, to_eid=target_eid, reason=reason,
        ))

        self.current_eid   = target.eid
        self.current_label = target.label
        self.visited.append(WalkNodeRef(eid=target.eid, label=target.label, is_rc=target.is_rc))
        self.depth += 1
        self.frontier = []   # 裁剪：sibling 直接丟棄，不留痕跡

        if self.depth >= self.max_depth:
            self.mark_dead_end(DeadEndReason.MAX_DEPTH)
        else:
            self.status = PathStatus.PENDING   # 等待下一輪 set_frontier

    def mark_dead_end(self, reason: DeadEndReason) -> None:
        self.status = PathStatus.DEAD_END
        self.dead_end_reason = reason
        self.frontier = []

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


# ══════════════════════════════════════════════════════════════
# WalkSession — K 條 path 的集合
# ══════════════════════════════════════════════════════════════

@dataclass
class WalkSession:
    """一次 random walk 對話的完整狀態，供 session_store 序列化保存。"""
    session_id: str
    paths:      List[PathState] = field(default_factory=list)
    max_depth:  int = 8

    # 累加式 semantic context summary（比照 core/semantic_context.py 的角色，
    # 隨對話累加，每次 advance 的 reason 會摺進這裡，供最終排名 + 可解釋性使用）
    context_summary: str = ""

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

    def record_decision(self, reason: str) -> None:
        """把一次 advance 的理由摺進 session 級的 context_summary。"""
        if not reason:
            return
        self.context_summary = (self.context_summary + " " + reason).strip()


# ══════════════════════════════════════════════════════════════
# WalkScheduler — round-robin，跳過已終止 path
# ══════════════════════════════════════════════════════════════

class WalkScheduler:
    """決定「這一輪換哪條 path」，不涉及 LLM 判斷。

    Round-robin：從上一次指標的下一位開始找，跳過 DEAD_END 的 path；
    找不到任何非 DEAD_END 的 path 時回傳 None（代表 walk 已全部結束）。
    """

    def __init__(self, session: WalkSession) -> None:
        self._session = session
        self._pointer = -1   # 上一輪選中的 index（在 session.paths 中的位置）

    def next_path(self) -> Optional[PathState]:
        n = len(self._session.paths)
        if n == 0:
            return None

        for offset in range(1, n + 1):
            idx = (self._pointer + offset) % n
            candidate = self._session.paths[idx]
            if candidate.status != PathStatus.DEAD_END:
                self._pointer = idx
                return candidate

        return None   # 所有 path 皆已終止
