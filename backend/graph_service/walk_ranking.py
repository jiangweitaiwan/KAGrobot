# -*- coding: utf-8 -*-
"""graph_service/walk_ranking.py — Random Walk Mode 的最終 RC 排名公式。

沿用 frontend DetailPanel.tsx / page.tsx 的 computeRanking 精神：
    totalScore = alpha * semantic + (1 - alpha) * interScore
其中 interScore 是「這個 RC 的 foundation 有多少比例被命中」。dialogue
mode 命中來源是 Personal Brain；random walk mode 的命中來源改成
「這次 walk 所有 path 走過的節點」（WalkSession.all_foundation_eids()
的聯集），把整趟 walk 當作一次性的、自動生成的 foundation 選取。

額外的收斂 bonus（random walk mode 特有，dialogue mode 沒有對應概念）：
  多條獨立起點的 path 若最終都走到同一個 RC，這是比「foundation 命中比例」
  更強的訊號——代表從不同角度出發的推理都收斂到同一個結論。用
  WalkSession.final_rc_candidates() 取得收斂事實，疊加一個與收斂
  path 數成正比的 bonus，不取代 base score，只是額外加分。

一個容易漏掉但重要的邊界情況：path 走到的 RC 不一定在「語意 Top-N pool」
內（rc_foundation_map / rc_semantic_scores 通常只涵蓋 CandidateRetriever
語意排名前 N 名，見 settings.rc_ranking_pool_size）。如果某條 path 走到
一個語意排名較後面、但透過圖上實際路徑到達的 RC，這正是 random walk
模式相對純語意搜尋的價值所在——不能因為它「不在語意 pool 裡」就從排名
結果中消失。這裡用 _ensure_walked_rcs_included() 補一個近似 entry：
foundation set 用「走到它的那幾條 path 各自走過的節點」聯集近似替代
（沒有真正的 foundation tree 資料可用），semantic score 預設為 0
（除非呼叫端額外提供），並標記 approx=True，供前端呈現「這是透過路徑
探索、非語意排名找到的候選」的區別提示。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_service.walk_state import WalkSession

_DEFAULT_ALPHA = 0.5
_DEFAULT_CONVERGENCE_BONUS_WEIGHT = 0.2


@dataclass
class WalkRCRanked:
    """單一 RC 候選的排名明細，供前端呈現分數拆解（比照既有 RankBar 的
    embed/intent 雙色拆解精神，這裡拆成 semantic / interScore / 收斂 bonus）。
    """
    eid:              str
    label:            str
    semantic:         float
    inter_score:      float
    hits:             int
    total_foundations: int
    convergence_count: int
    convergence_bonus: float
    total_score:      float
    approx:           bool = False   # True 代表此 RC 不在語意 pool 內，foundation set 為近似值


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def compute_walk_ranking(
    session:            WalkSession,
    rc_foundation_map:  Dict[str, Dict[str, Any]],
    rc_semantic_scores: Dict[str, float],
    alpha:              float = _DEFAULT_ALPHA,
    convergence_bonus_weight: float = _DEFAULT_CONVERGENCE_BONUS_WEIGHT,
) -> List[WalkRCRanked]:
    """計算 random walk 結束後的 RC 推薦排名。

    Parameters
    ----------
    session : 已跑完（或部分跑完）的 WalkSession
    rc_foundation_map : eid -> {"label": str, "foundation_eids": List[str]}
        對應前端 RCFoundationMap 的後端版本，資料來源與既有
        /graph/expand/batch 相同（每個候選 RC 的 foundation 清單）。
    rc_semantic_scores : eid -> semantic score（CandidateRetriever.score_all()
        的輸出，通常只涵蓋語意排名前 N 名——即便如此，見檔頭說明，
        走到的 RC 若不在其中仍會被補進結果）。
    alpha : 與 frontend ALPHA 常數意義相同，semantic 與 interScore 的權重比例。
    convergence_bonus_weight : 收斂 bonus 的最大權重（見 _convergence_bonus）。

    Returns
    -------
    依 total_score 由高到低排序的清單。
    """
    walked_eids = _collect_walked_eids(session)
    convergence = session.final_rc_candidates()
    total_paths = len(session.paths)

    merged_map = _ensure_walked_rcs_included(session, rc_foundation_map, convergence)

    ranked: List[WalkRCRanked] = []
    for eid, info in merged_map.items():
        foundation_eids = info.get("foundation_eids") or []
        n    = len(foundation_eids)
        hits = sum(1 for f in foundation_eids if f in walked_eids)
        inter_score = hits / n if n > 0 else 0.0

        semantic = rc_semantic_scores.get(eid, 0.0)
        base_score = alpha * semantic + (1 - alpha) * inter_score

        conv_count = len(convergence.get(eid, []))
        bonus = _convergence_bonus(conv_count, total_paths, convergence_bonus_weight)

        ranked.append(WalkRCRanked(
            eid=eid, label=info.get("label", eid),
            semantic=semantic, inter_score=inter_score,
            hits=hits, total_foundations=n,
            convergence_count=conv_count, convergence_bonus=bonus,
            total_score=base_score + bonus,
            approx=bool(info.get("approx", False)),
        ))

    ranked.sort(key=lambda r: r.total_score, reverse=True)
    return ranked


# ══════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════

def _collect_walked_eids(session: WalkSession) -> set:
    """所有 path 走過的節點聯集（不含各自的最終節點本身，理由見
    graph_service/walk_state.py 的 PathState.foundation_eids() docstring：
    避免一個節點在計算「它自己所屬 RC」的命中率時把自己算進去）。
    """
    walked: set = set()
    for eids in session.all_foundation_eids().values():
        walked.update(eids)
    return walked


def _convergence_bonus(
    conv_count:   int,
    total_paths:  int,
    bonus_weight: float,
) -> float:
    """收斂 path 數 → bonus 分數。

    只有 conv_count >= 2（至少兩條獨立 path 收斂）才給 bonus；單一 path
    走到的 RC 不算「收斂」，那只是正常結果，不該額外加分。bonus 隨收斂
    path 佔全部 path 的比例線性增加，封頂在 bonus_weight（當「幾乎所有
    path 都收斂到同一個 RC」時達到最大加分）。
    """
    if conv_count < 2 or total_paths <= 1:
        return 0.0
    return bonus_weight * (conv_count - 1) / (total_paths - 1)


def _ensure_walked_rcs_included(
    session:           WalkSession,
    rc_foundation_map: Dict[str, Dict[str, Any]],
    convergence:       Dict[str, List[int]],
) -> Dict[str, Dict[str, Any]]:
    """把「path 實際走到、但不在 rc_foundation_map 裡」的 RC 補進來。

    近似 foundation set：用走到它的那幾條 path 各自的 foundation_eids
    聯集替代（沒有真正的 foundation tree 資料可用於這些 RC，因為它們
    當初就是因為不在語意 pool 前 N 名，才沒有被批次 expand 拿到真正的
    foundation tree）。
    """
    merged = dict(rc_foundation_map)

    for eid, path_ids in convergence.items():
        if eid in merged:
            continue

        approx_foundations: set = set()
        for pid in path_ids:
            approx_foundations.update(session.get_path(pid).foundation_eids())

        merged[eid] = {
            "label":           session.get_path(path_ids[0]).current_label,
            "foundation_eids": list(approx_foundations),
            "approx":          True,
        }

    return merged
