# -*- coding: utf-8 -*-
"""core/rc_briefing.py — RC 全局說明（briefing）。

使用者點選推薦 RC 標籤後、進入 main_agent 之前的「全局視野」說明：
把 context_summary（問題脈絡）與 RC 的 foundation_tree（上游匯流結構，
含 RC 自己）交給 LLM，說明 RC 方案如何從全局層面解決問題，並且
明確標出方案的邊界（fit_gaps / contested_points）。

設計重點
  - 獨立於 SocratesChatBot：soc 的輸出契約是嚴格 JSON 且在 PATH_CONFIRMED
    後凍結；briefing 是說明型任務，用獨立 prompt + pydantic structured output。
  - 預覽無副作用：不碰 selected_rc_eid、不建 sandbox、不改 personal brain。
  - depth 約定：depth 0 = RC 自己，數字越大越上游。tree 節點本身沒有 depth
    欄位，這裡用邊做無向 BFS 從 RC 起算（不依賴邊的方向約定）。
  - 所有 LLM 回傳的 eid 都會對照 tree 的 eid 集合驗證，不合法的丟棄。
  - walk_overlap 由程式計算（walk visited ∩ tree），不交給 LLM。
  - 快取：BriefingCache，key = (uid, session_id, eid, context_hash)；
    另存 latest 索引供 main_agent 取 handoff（main_agent 不知道 context_hash）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.google_llm import create_chat_google_generative_ai
from core.llm_retry import call_with_retry

MAX_NODES          = 28     # 送進 LLM 的節點上限（含 RC 本身）
MAX_LAYERS         = 4
MAX_CONTESTED      = 3
MAX_FIT_GAPS       = 3
DEF_TRUNC          = 160    # definition / mechanism 截斷字數


# ═══════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════

class Layer(BaseModel):
    name:      str       = Field(description="這一層的名稱（繁體中文，10 字內）。")
    node_eids: List[str] = Field(description="屬於這一層的節點 eid，必須來自提供的節點清單。")
    role:      str       = Field(description="這一層在整體結構中負責提供什麼（一句話）。")


class ContestedPoint(BaseModel):
    eid: str = Field(description="有爭議的節點 eid，必須來自節點清單，優先選 critique 節點。")
    why: str = Field(description="為什麼這個節點會讓方案的可靠度打折（一句話）。")


class Handoff(BaseModel):
    assumptions: List[str] = Field(default_factory=list, description="後續分析必須成立的前提假設。")
    params_hint: List[str] = Field(default_factory=list, description="後續分析需要向使用者確認的參數方向。")


class LLMBriefing(BaseModel):
    """LLM 負責產出的欄位（walk_overlap 由程式計算，不在此）。"""
    problem_frame:    str                  = Field(description="根據 context_summary 重述的問題脈絡（繁體中文，2-3 句）。")
    layers:           List[Layer]          = Field(description=f"依上游到下游排列的結構分層，最多 {MAX_LAYERS} 層。")
    convergence:      str                  = Field(description="各層如何逐步匯流到 RC，只能依據提供的邊說明，不得自行發明連結。")
    rc_role:          str                  = Field(description="RC 在整體結構中負責解決什麼。")
    contested_points: List[ContestedPoint] = Field(default_factory=list, description=f"最多 {MAX_CONTESTED} 條。")
    fit_gaps:         List[str]            = Field(
        default_factory=list,
        description=(
            f"最多 {MAX_FIT_GAPS} 條。逐條對照問題脈絡中的需求點，列出 RC 沒有覆蓋的部分；"
            "若確實都有覆蓋，回傳空陣列，不要為了湊數編造。"
        ),
    )
    handoff:          Handoff


class RCBriefing(LLMBriefing):
    walk_overlap: List[str] = Field(default_factory=list, description="walk 探索過的節點中，落在此 RC 結構內的 eid。")


# ═══════════════════════════════════════════════════════════════
# Tree 前處理：depth 計算 / 剪枝 / 序列化
# ═══════════════════════════════════════════════════════════════

def compute_depths(anchor_eid: str, edges: List[Dict[str, Any]]) -> Dict[str, int]:
    """以 RC 為 depth 0，沿邊（無向）BFS 計算每個節點的 depth。"""
    adj: Dict[str, set] = {}
    for e in edges:
        s, t = e["source"], e["target"]
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    depths = {anchor_eid: 0}
    q = deque([anchor_eid])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in depths:
                depths[nxt] = depths[cur] + 1
                q.append(nxt)
    return depths


def prune_tree(
    anchor_eid: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_nodes: int = MAX_NODES,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """保留 RC + 全部 critique 節點（其次）+ 由近到遠的其餘節點，補上 depth。"""
    depths = compute_depths(anchor_eid, edges)
    by_id  = {n["element_id"]: n for n in nodes}
    by_id.setdefault(anchor_eid, {"element_id": anchor_eid, "label": anchor_eid, "node_type": "support"})

    def sort_key(eid: str):
        n = by_id[eid]
        is_critique = n.get("node_type") == "critique"
        # RC 最先；critique 次之；其餘依 depth 由近到遠
        return (eid != anchor_eid, not is_critique, depths.get(eid, 99), eid)

    ordered = sorted(by_id.keys(), key=sort_key)
    kept    = set(ordered[:max_nodes])
    kept_nodes = [
        {**by_id[eid], "depth": depths.get(eid, 99)} for eid in ordered if eid in kept
    ]
    kept_edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
    return kept_nodes, kept_edges, depths


def _trunc(s: Optional[str], n: int = DEF_TRUNC) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def serialize_tree(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
    anchor_eid: str,
) -> str:
    """節點清單 + 明列的邊。邊明列是為了避免 LLM 自己發明節點間的連結。"""
    label = {n["element_id"]: n["label"] for n in nodes}
    lines = ["## 節點（depth 0 = RC 本身，數字越大越上游）"]
    for n in nodes:
        eid = n["element_id"]
        d   = details.get(eid) or {}
        tag = "【RC】" if eid == anchor_eid else f"[{n.get('node_type', 'support')}]"
        desc = "；".join(filter(None, [
            f"定義：{_trunc(d.get('definition'))}" if d.get("definition") else "",
            f"機制：{_trunc(d.get('mechanism'))}" if d.get("mechanism") else "",
            f"功能：{_trunc(d.get('function'))}" if d.get("function") else "",
        ])) or "（無描述）"
        lines.append(f"- eid={eid} | depth={n['depth']} | {n['label']} {tag} | {desc}")
    lines.append("\n## 邊（source → target）")
    for e in edges:
        lines.append(
            f"- {label.get(e['source'], e['source'])} → {label.get(e['target'], e['target'])}"
            f" ({e.get('edge_type', 'support')})"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = f"""\
你是一位專業經濟學顧問。使用者正在評估一個推薦的解決方案概念（RC）。
在進入分析之前，你要先給出「全局視野」：RC 的上游結構（foundation tree）
如何一層層匯流成這個方案，以及它如何回應使用者的問題。

## 規則
- 只能依據下方提供的節點與邊說明，不得引入清單以外的概念，也不得自行發明節點間的連結。
- layers：依 depth 由上游到 RC 分成最多 {MAX_LAYERS} 層；node_eids 必須來自節點清單。
- convergence：說明各層如何匯流，引用邊的方向，不要寫成泛泛的「整合各種理論」。
- contested_points：優先從 [critique] 節點挑，最多 {MAX_CONTESTED} 條。
- fit_gaps：逐條對照「使用者需求」中的每個需求點，列出 RC 沒有覆蓋的部分，最多 {MAX_FIT_GAPS} 條。
  這是全局視野最重要的價值——讓使用者看清楚方案的邊界。確實都覆蓋時回傳空陣列，不要湊數。
- 推薦是語意相似度排出來的，不保證方案真的適用；不要替方案編造通順的因果故事。
- 全部用繁體中文，簡潔具體。
"""


def _build_user_prompt(rc_label: str, context_text: str, tree_text: str) -> str:
    return (
        f"## 使用者需求（問題脈絡）\n{context_text}\n\n"
        f"## 推薦的 RC\n{rc_label}\n\n{tree_text}"
    )


# ═══════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════

class RCBriefingGenerator:
    def __init__(self, *, model: str = "gemini-2.5-flash", api_key: str = "") -> None:
        llm = create_chat_google_generative_ai(
            model=model, temperature=0.1, api_key=api_key, feature="rc_briefing",
        )
        self._llm = llm.with_structured_output(LLMBriefing)

    def generate(
        self,
        *,
        rc_eid:       str,
        rc_label:     str,
        context_text: str,
        nodes:        List[Dict[str, Any]],
        edges:        List[Dict[str, Any]],
        details:      Dict[str, Dict[str, Any]],
        walk_visited: Iterable[str] = (),
    ) -> Tuple[RCBriefing, Dict[str, str]]:
        """同步執行（呼叫端以 asyncio.to_thread 包裝）。

        回傳 (briefing, labels)；labels 是 briefing 引用到的 eid → label，
        供前端渲染卡片，不必另外查詢。
        """
        kept_nodes, kept_edges, _ = prune_tree(rc_eid, nodes, edges)
        tree_text = serialize_tree(kept_nodes, kept_edges, details, rc_eid)
        valid     = {n["element_id"] for n in kept_nodes}
        prompt    = _build_user_prompt(rc_label, context_text, tree_text)

        result: Optional[LLMBriefing] = None
        for _ in range(2):   # 驗證後 layers 為空就重試一次
            raw = call_with_retry(lambda: self._llm.invoke(
                [("system", _SYSTEM_PROMPT), ("human", prompt)]
            ))
            candidate = self.validate(raw, valid)
            if candidate.layers:
                result = candidate
                break
            result = candidate
        assert result is not None

        overlap = sorted(set(walk_visited) & valid - {rc_eid})
        briefing = RCBriefing(**result.model_dump(), walk_overlap=overlap)

        referenced = set(overlap)
        for layer in briefing.layers:
            referenced.update(layer.node_eids)
        referenced.update(c.eid for c in briefing.contested_points)
        label_map = {n["element_id"]: n["label"] for n in kept_nodes}
        return briefing, {e: label_map[e] for e in referenced if e in label_map}

    @staticmethod
    def validate(b: LLMBriefing, valid_eids: set) -> LLMBriefing:
        """丟棄不在 tree 內的 eid、超出上限的項目；空層整層移除。"""
        layers: List[Layer] = []
        for layer in b.layers[:MAX_LAYERS]:
            eids = [e for e in layer.node_eids if e in valid_eids]
            if eids:
                layers.append(Layer(name=layer.name, node_eids=eids, role=layer.role))
        contested = [c for c in b.contested_points if c.eid in valid_eids][:MAX_CONTESTED]
        return b.model_copy(update={
            "layers": layers,
            "contested_points": contested,
            "fit_gaps": [g for g in b.fit_gaps if g.strip()][:MAX_FIT_GAPS],
        })


# ═══════════════════════════════════════════════════════════════
# Cache（記憶體 LRU；重啟後重新生成，成本可接受）
# ═══════════════════════════════════════════════════════════════

def context_hash(embedding_text: str, walk_visited: Iterable[str] = ()) -> str:
    payload = json.dumps([embedding_text.strip(), sorted(set(walk_visited))], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class BriefingCache:
    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._data:   "OrderedDict[tuple, dict]" = OrderedDict()
        self._latest: Dict[tuple, dict] = {}
        self._locks:  Dict[tuple, asyncio.Lock] = {}

    def lock_for(self, key: tuple) -> asyncio.Lock:
        """同一個 key 的並發請求共用一把鎖，避免重複生成／重複計費。"""
        return self._locks.setdefault(key, asyncio.Lock())

    def get(self, key: tuple) -> Optional[dict]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: tuple, value: dict) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        uid, sid, eid, _ = key
        self._latest[(uid, sid, eid)] = value
        while len(self._data) > self._maxsize:
            old_key, _ = self._data.popitem(last=False)
            self._locks.pop(old_key, None)

    def latest(self, uid: str, session_id: str, eid: str) -> Optional[dict]:
        """main_agent 取 handoff 用：不需要知道 context_hash。"""
        return self._latest.get((uid, session_id, eid))


def normalize_briefing_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """快取／Firestore 讀回後統一 briefing 為 RCBriefing 實例。"""
    b = payload.get("briefing")
    if isinstance(b, dict):
        return {**payload, "briefing": RCBriefing(**b)}
    return payload


def handoff_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """從 briefing 快取 payload 抽出 main_agent system prompt 用的 handoff。"""
    if not payload:
        return None
    payload = normalize_briefing_payload(payload)
    b: RCBriefing = payload["briefing"]
    return {
        "assumptions": b.handoff.assumptions,
        "params_hint": b.handoff.params_hint,
        "fit_gaps":    b.fit_gaps,
    }
