# -*- coding: utf-8 -*-
"""core/brain/rc_briefing.py — RC 全局說明（briefing）。

使用者點選推薦 RC 標籤後、進入 main_agent 之前的「全局視野」說明：
把問題脈絡與 RC 的 foundation_tree（上游匯流結構，含 RC 自己）交給 LLM，
說明 RC 方案如何從全局層面解決問題，並明確標出方案的邊界
（fit_gaps / contested_points）。

【本版重點】
  - 拓樸序列化：以「目標節點 ← 來源節點」分組的上游清單取代逐條邊，
    prompt 內完全不出現 eid；LLM 一律以 label 引用節點，
    由 validate() 反查回 eid 並丟棄不合法引用。對外的 RCBriefing 仍是 eid 版，
    Firestore 舊資料、labels 字典、前端卡片不受影響。
  - depth = 沿「上游 → 下游」有向邊到 RC 的最長路徑（所有邊由高 depth 指向低 depth，
    圖才會乾淨）；偵測到環時退回無向 BFS。
    邊方向約定：edge.source = 上游支撐，edge.target = 下游（朝 RC 流動）。
  - 剪枝保證連通：保留任何節點時，連同它通往 RC 的中繼節點一起保留。
    MAX_NODES 為軟上限，critique 節點可擴張到 HARD_MAX_NODES。
  - 剪枝只做一次：路由呼叫 prune_tree()，結果（PrunedTree）同時用於
    retrieve_batch、prompt 與 graph，三者保證一致。
  - 需求與探索分開：requirement_text（使用者要什麼）與 exploration_text
    （慢想探索軌跡）是兩個獨立段落，fit_gaps 只對照需求。
  - 回傳 graph（剪枝後、LLM 實際看到的那張圖），供前端畫拓樸並隨 briefing 持久化。
  - 預覽無副作用：不碰 selected_rc_eid、不建 sandbox、不改 personal brain。
  - 快取：BriefingCache，key = (uid, session_id, eid, context_hash)。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.llm.google_llm import create_chat_google_generative_ai
from core.llm.llm_retry import call_with_retry

MAX_NODES       = 28    # 送進 LLM 的節點「軟」上限（含 RC 本身）
HARD_MAX_NODES  = 36    # 為了保留 critique 節點與其通往 RC 的路徑，最多可擴張到這裡
MAX_LAYERS      = 4
MAX_CONTESTED   = 3
MAX_FIT_GAPS    = 3
FULL_TRUNC      = 160   # RC / critique 節點：定義、機制、功能各自截斷字數
SHORT_TRUNC     = 80    # 其他節點：只給定義，截短


# ═══════════════════════════════════════════════════════════════
# Schema — 對外（eid 版，與既有 Firestore / 前端相容）
# ═══════════════════════════════════════════════════════════════

class Layer(BaseModel):
    name:      str       = Field(description="這一層的名稱（繁體中文，10 字內）。")
    node_eids: List[str] = Field(default_factory=list, description="屬於這一層的節點 eid。")
    role:      str       = Field(description="這一層在整體結構中負責提供什麼（一句話）。")


class ContestedPoint(BaseModel):
    eid: str = Field(description="有爭議的節點 eid。")
    why: str = Field(description="為什麼這個節點會讓方案的可靠度打折（一句話）。")


class Handoff(BaseModel):
    assumptions: List[str] = Field(default_factory=list, description="後續分析必須成立的前提假設。")
    params_hint: List[str] = Field(default_factory=list, description="後續分析需要向使用者確認的參數方向。")


class RCBriefing(BaseModel):
    problem_frame:    str
    layers:           List[Layer]
    convergence:      str
    rc_role:          str
    contested_points: List[ContestedPoint] = Field(default_factory=list)
    fit_gaps:         List[str]            = Field(default_factory=list)
    handoff:          Handoff
    walk_overlap:     List[str]            = Field(
        default_factory=list, description="walk 探索過的節點中，落在此 RC 結構內的 eid。"
    )


class GraphNode(BaseModel):
    eid:       str
    label:     str
    node_type: str = "support"   # support | critique | other
    depth:     int = 0           # 0 = RC，數字越大越上游


class GraphEdge(BaseModel):
    source:    str               # 上游 eid
    target:    str               # 下游 eid（朝 RC）
    edge_type: str = "support"


class BriefingGraph(BaseModel):
    """剪枝後、LLM 實際看到的那張圖。注意 Firestore 不允許陣列直接巢狀陣列，
    所以 nodes / edges 都是「dict 的 list」。"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ═══════════════════════════════════════════════════════════════
# Schema — LLM 端（label 版）
# ═══════════════════════════════════════════════════════════════

class LLMLayer(BaseModel):
    name:        str       = Field(description="這一層的名稱（繁體中文，10 字內）。")
    node_labels: List[str] = Field(
        description="屬於這一層的節點名稱，必須逐字複製自「節點說明」中的名稱（不含【RC】、〔批判〕等標記）。"
    )
    role:        str       = Field(description="這一層在整體結構中負責提供什麼（一句話）。")


class LLMContested(BaseModel):
    label: str = Field(description="有爭議的節點名稱，逐字複製自節點說明，優先選〔批判〕節點。")
    why:   str = Field(description="為什麼這個節點會讓方案的可靠度打折（一句話）。")


class LLMBriefing(BaseModel):
    """LLM 負責產出的欄位（walk_overlap 由程式計算，不在此）。"""
    problem_frame:    str                 = Field(description="根據使用者需求重述的問題脈絡（繁體中文，2-3 句）。")
    layers:           List[LLMLayer]      = Field(description=f"依上游到下游排列的結構分層，最多 {MAX_LAYERS} 層。")
    convergence:      str                 = Field(description="各層如何逐步匯流到 RC，只能依據提供的推導結構說明，不得自行發明連結。")
    rc_role:          str                 = Field(description="RC 在整體結構中負責解決什麼。")
    contested_points: List[LLMContested]  = Field(default_factory=list, description=f"最多 {MAX_CONTESTED} 條。")
    fit_gaps:         List[str]           = Field(
        default_factory=list,
        description=(
            f"最多 {MAX_FIT_GAPS} 條。逐條對照使用者需求中的需求點，列出 RC 沒有覆蓋的部分；"
            "若確實都有覆蓋，回傳空陣列，不要為了湊數編造。"
        ),
    )
    handoff:          Handoff


# ═══════════════════════════════════════════════════════════════
# Tree 前處理：depth / 剪枝（保證連通）
# ═══════════════════════════════════════════════════════════════

@dataclass
class PrunedTree:
    """剪枝後的樹。路由與 generator 共用同一份，確保 details / prompt / graph 一致。"""
    anchor_eid: str
    nodes:      List[Dict[str, Any]] = field(default_factory=list)   # element_id,label,node_type,depth
    edges:      List[Dict[str, Any]] = field(default_factory=list)   # source,target,edge_type
    directed:   bool = True       # False = 偵測到環，depth 退回無向 BFS


def _dedupe_edges(edges: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for e in edges:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        key = (s, t, e.get("edge_type", "support"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _undirected_bfs(anchor: str, edges: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, str]]:
    adj: Dict[str, set] = {}
    for e in edges:
        adj.setdefault(e["source"], set()).add(e["target"])
        adj.setdefault(e["target"], set()).add(e["source"])
    depth  = {anchor: 0}
    parent: Dict[str, str] = {}
    q = deque([anchor])
    while q:
        cur = q.popleft()
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in depth:
                depth[nxt]  = depth[cur] + 1
                parent[nxt] = cur
                q.append(nxt)
    return depth, parent


def compute_depths(anchor_eid: str, edges: List[Dict[str, Any]]) -> Tuple[Dict[str, int], bool]:
    """depth = 沿有向邊（source→target）到 RC 的最長路徑長度。

    回傳 (depths, directed)。directed=False 表示子圖有環，已退回無向 BFS。
    無法有向抵達 RC 的節點，depth 退而採無向 BFS 距離（少見，僅為不遺漏）。
    """
    edges = _dedupe_edges(edges)
    preds: Dict[str, set] = {}
    succs: Dict[str, set] = {}
    for e in edges:
        succs.setdefault(e["source"], set()).add(e["target"])
        preds.setdefault(e["target"], set()).add(e["source"])

    # R = 能有向抵達 RC 的節點
    reach = {anchor_eid}
    q = deque([anchor_eid])
    while q:
        cur = q.popleft()
        for p in preds.get(cur, ()):
            if p not in reach:
                reach.add(p)
                q.append(p)

    # RC 視為匯點：忽略從 RC 出發的邊。Kahn：由匯點往上游逐層確定最長路徑。
    out_deg = {
        n: (0 if n == anchor_eid else sum(1 for t in succs.get(n, ()) if t in reach))
        for n in reach
    }
    depth: Dict[str, int] = {}
    q = deque([anchor_eid])
    done = 0
    while q:
        n = q.popleft()
        done += 1
        if n == anchor_eid:
            depth[n] = 0
        else:
            depth[n] = 1 + max(depth[t] for t in succs[n] if t in reach)
        for p in preds.get(n, ()):
            if p in reach and p != anchor_eid:
                out_deg[p] -= 1
                if out_deg[p] == 0:
                    q.append(p)

    if done < len(reach):                       # 有環
        und, _ = _undirected_bfs(anchor_eid, edges)
        return und, False

    und, _ = _undirected_bfs(anchor_eid, edges)
    for n, d in und.items():
        depth.setdefault(n, d)
    return depth, True


def _next_hops(anchor_eid: str, edges: List[Dict[str, Any]]) -> Dict[str, str]:
    """每個節點通往 RC 的「下一步」。優先走有向邊（朝下游），其餘退回無向。"""
    preds: Dict[str, set] = {}
    for e in edges:
        preds.setdefault(e["target"], set()).add(e["source"])
    hop: Dict[str, str] = {}
    seen = {anchor_eid}
    q = deque([anchor_eid])
    while q:
        cur = q.popleft()
        for p in sorted(preds.get(cur, ())):
            if p not in seen:
                seen.add(p)
                hop[p] = cur
                q.append(p)
    _, parent = _undirected_bfs(anchor_eid, edges)
    for n, par in parent.items():
        hop.setdefault(n, par)
    return hop


def prune_tree(
    anchor_eid:   str,
    nodes:        List[Dict[str, Any]],
    edges:        List[Dict[str, Any]],
    max_nodes:    int = MAX_NODES,
    hard_max:     int = HARD_MAX_NODES,
    anchor_label: Optional[str] = None,
) -> PrunedTree:
    """RC 必留；critique 節點次之（可擴張到 hard_max）；其餘由近到遠補到 max_nodes。

    保留任何節點時，一併保留它通往 RC 的中繼節點，確保剪枝後的圖連通。
    """
    by_id = {n["element_id"]: n for n in nodes}
    by_id.setdefault(anchor_eid, {
        "element_id": anchor_eid, "label": anchor_label or anchor_eid, "node_type": "support",
    })
    edges = [e for e in _dedupe_edges(edges) if e["source"] in by_id and e["target"] in by_id]

    depths, directed = compute_depths(anchor_eid, edges)
    hop = _next_hops(anchor_eid, edges)

    def is_critique(eid: str) -> bool:
        return by_id[eid].get("node_type") == "critique"

    candidates = sorted(
        (e for e in by_id if e != anchor_eid and e in depths),   # 與 RC 不連通的節點直接捨棄
        key=lambda e: (not is_critique(e), depths[e], e),
    )

    kept = {anchor_eid}
    for cand in candidates:
        if cand in kept:
            continue
        chain, cur = [], cand
        while cur is not None and cur not in kept:
            chain.append(cur)
            cur = hop.get(cur)
        if cur is None:                          # 找不到通往 RC 的路，略過
            continue
        limit = hard_max if is_critique(cand) else max_nodes
        if len(kept) + len(chain) <= limit:
            kept.update(chain)

    ordered = sorted(kept, key=lambda e: (e != anchor_eid, depths.get(e, 99), e))
    kept_nodes = [
        {
            "element_id": e,
            "label":      by_id[e].get("label") or e,
            "node_type":  by_id[e].get("node_type", "support"),
            "depth":      depths.get(e, 99),
        }
        for e in ordered
    ]
    kept_edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
    return PrunedTree(anchor_eid=anchor_eid, nodes=kept_nodes, edges=kept_edges, directed=directed)


# ═══════════════════════════════════════════════════════════════
# Label 處理：消歧 / 正規化 / 反查
# ═══════════════════════════════════════════════════════════════

_TAG_RE = re.compile(r"【\s*RC\s*】|〔\s*批判\s*〕|\[\s*(?:RC|critique|support|other)\s*\]", re.I)
_QUOTES = "「」『』\"'“”‘’"


def _norm_label(s: str) -> str:
    """只做正規化，不做模糊比對：全半形統一、去掉 LLM 順手抄上的標記與引號、空白收斂。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = _TAG_RE.sub("", s)
    s = s.strip().strip(_QUOTES).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def build_label_index(tree: PrunedTree) -> Tuple[Dict[str, str], Dict[str, str]]:
    """回傳 (display, l2e)。

    display: eid → prompt 中使用的名稱（重名者加「（2）」「（3）」…後綴消歧）
    l2e:     正規化後的 display 名稱 → eid
    """
    display: Dict[str, str] = {}
    l2e:     Dict[str, str] = {}
    for n in tree.nodes:                          # tree.nodes 已是 RC 優先、depth 由近到遠的穩定順序
        eid  = n["element_id"]
        base = re.sub(r"\s+", " ", (n["label"] or eid)).strip()
        cand, k = base, 2
        while _norm_label(cand) in l2e:
            cand = f"{base}（{k}）"
            k += 1
        display[eid] = cand
        l2e[_norm_label(cand)] = eid
    return display, l2e


def _resolve(label: str, l2e: Dict[str, str]) -> Optional[str]:
    return l2e.get(_norm_label(label))


# ═══════════════════════════════════════════════════════════════
# 序列化（拓樸）
# ═══════════════════════════════════════════════════════════════

def _trunc(s: Optional[str], n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def serialize_tree(
    tree:    PrunedTree,
    details: Dict[str, Dict[str, Any]],
    display: Dict[str, str],
) -> str:
    """兩段：
      1. 推導結構：依目標節點分組的上游清單（匯流一眼可見），從 RC 往上游排。
      2. 節點說明：由最上游到 RC。RC / critique 給完整描述，其餘只給截短的定義。
    prompt 內不含 eid。
    """
    anchor = tree.anchor_eid
    by_id  = {n["element_id"]: n for n in tree.nodes}

    incoming: Dict[str, List[Dict[str, Any]]] = {}
    for e in tree.edges:
        incoming.setdefault(e["target"], []).append(e)

    lines = ["## 推導結構（「A ← B、C」表示 B、C 支撐 A；〔批判〕表示該來源對 A 提出質疑）"]
    targets = sorted(incoming, key=lambda t: (by_id[t]["depth"], display[t]))
    if not targets:
        lines.append("（此 RC 沒有上游節點）")
    for t in targets:
        srcs = sorted(incoming[t], key=lambda e: (-by_id[e["source"]]["depth"], display[e["source"]]))
        parts = [
            display[e["source"]] + ("〔批判〕" if e.get("edge_type") == "critique" else "")
            for e in srcs
        ]
        head = ("【RC】" if t == anchor else "") + display[t]
        lines.append(f"{head} ← {'、'.join(parts)}")

    lines.append("\n## 節點說明（第 0 層 = RC 本身，層數越大越上游）")
    for n in sorted(tree.nodes, key=lambda n: (-n["depth"], display[n["element_id"]])):
        eid         = n["element_id"]
        d           = details.get(eid) or {}
        is_rc       = eid == anchor
        is_critique = n.get("node_type") == "critique"
        tag         = "【RC】" if is_rc else ("〔批判〕" if is_critique else "")
        if is_rc or is_critique:
            desc = "；".join(filter(None, [
                f"定義：{_trunc(d.get('definition'), FULL_TRUNC)}" if d.get("definition") else "",
                f"機制：{_trunc(d.get('mechanism'), FULL_TRUNC)}" if d.get("mechanism") else "",
                f"功能：{_trunc(d.get('function'),  FULL_TRUNC)}" if d.get("function")  else "",
            ]))
        else:
            short = d.get("definition") or d.get("mechanism") or d.get("function")
            desc  = _trunc(short, SHORT_TRUNC)
        lines.append(f"第{n['depth']}層｜{display[eid]}{tag}：{desc or '（無描述）'}")
    return "\n".join(lines)


def build_graph(tree: PrunedTree) -> BriefingGraph:
    return BriefingGraph(
        nodes=[
            GraphNode(eid=n["element_id"], label=n["label"],
                      node_type=n.get("node_type", "support"), depth=n["depth"])
            for n in tree.nodes
        ],
        edges=[
            GraphEdge(source=e["source"], target=e["target"], edge_type=e.get("edge_type", "support"))
            for e in tree.edges
        ],
    )


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = f"""\
你是一位專業經濟學顧問。使用者正在評估一個推薦的解決方案概念（RC）。
在進入分析之前，你要先給出「全局視野」：RC 的上游結構如何一層層匯流成這個方案，
讓使用者看懂「這個建議是怎麼推導出來的」，以及它如何回應使用者的需求。

## 怎麼讀輸入
- 「推導結構」每行 `A ← B、C` 表示 B、C 支撐 A；箭頭一律朝 RC 匯流。
- 「節點說明」的層數：第 0 層是 RC，數字越大越上游。
- 〔批判〕表示質疑或限制，不是支撐。

## 寫作角度
- 敘事要像一條推導鏈：從最上游的基礎概念出發，說明每一層如何成為下一層的前提，最後說明 RC 導出什麼行動或決策。
- 例如（僅示意格式，不要套用內容）：「先用折現法決定價值，再用 DCF 估算內在價值，最後買進價格低於內在價值的公司。」
  使用者應能由此理解最終建議是怎麼來的。

## 規則
- 只能依據提供的節點與推導結構說明，不得引入清單以外的概念，也不得自行發明節點間的連結。
- 凡是需要填入節點名稱的欄位（node_labels、contested_points.label），必須逐字複製「節點說明」中的名稱，不要附帶【RC】、〔批判〕、層數等標記，也不要改寫或縮寫。
- layers：依層數由大（上游）到小（RC）分成最多 {MAX_LAYERS} 層。
- convergence：說明各層如何匯流，引用推導結構中的方向，不要寫成泛泛的「整合各種理論」。
- contested_points：優先從〔批判〕節點挑，最多 {MAX_CONTESTED} 條。
- fit_gaps：逐條對照「使用者需求」中的每個需求點，列出 RC 沒有覆蓋的部分，最多 {MAX_FIT_GAPS} 條。
  這是全局視野最重要的價值——讓使用者看清楚方案的邊界。確實都覆蓋時回傳空陣列，不要湊數。
- 若有「使用者的探索思路」：那只說明使用者先前關注過什麼，不是需求清單。
  fit_gaps 只對照「使用者需求」；可在敘事中點出使用者探索過的節點正好對應哪一步推導。
- 推薦是語意相似度排出來的，不保證方案真的適用；不要替方案編造通順的因果故事。
- 全部用繁體中文，簡潔具體。
"""


def _build_user_prompt(
    rc_label:         str,
    requirement_text: str,
    exploration_text: str,
    overlap_labels:   List[str],
    tree_text:        str,
) -> str:
    parts = [f"## 使用者需求（問題脈絡）\n{requirement_text.strip()}"]
    if exploration_text.strip():
        explore = (
            "## 使用者的探索思路（慢想模式）\n"
            "以下是使用者先前自行探索時的思路摘要，僅說明他關注過什麼，不是需求。\n"
            f"{exploration_text.strip()}"
        )
        if overlap_labels:
            explore += f"\n使用者探索時走過、且位於此結構內的節點：{'、'.join(overlap_labels)}"
        parts.append(explore)
    parts.append(f"## 推薦的 RC\n{rc_label}\n\n{tree_text}")
    return "\n\n".join(parts)


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
        rc_eid:           str,
        requirement_text: str,
        tree:             PrunedTree,
        details:          Dict[str, Dict[str, Any]],
        exploration_text: str = "",
        walk_visited:     Iterable[str] = (),
        rc_label:         str = "",          # 保留參數以相容舊呼叫；實際使用 tree 內的 RC label
    ) -> Tuple[RCBriefing, Dict[str, str], BriefingGraph]:
        """同步執行（呼叫端以 asyncio.to_thread 包裝）。

        tree 由路由呼叫 prune_tree() 取得（只剪一次）。
        回傳 (briefing, labels, graph)：
          labels = briefing 引用到的 eid → label（相容舊前端）
          graph  = 剪枝後的拓樸圖
        """
        display, l2e = build_label_index(tree)
        valid        = {n["element_id"] for n in tree.nodes}
        overlap      = sorted((set(walk_visited) & valid) - {rc_eid})

        tree_text = serialize_tree(tree, details, display)
        prompt    = _build_user_prompt(
            display[rc_eid], requirement_text, exploration_text,
            [display[e] for e in overlap], tree_text,
        )

        result: Optional[RCBriefing] = None
        for _ in range(2):                         # 驗證後 layers 為空就重試一次
            raw = call_with_retry(lambda: self._llm.invoke(
                [("system", _SYSTEM_PROMPT), ("human", prompt)]
            ))
            if raw is None:
                continue
            candidate = self.validate(raw, l2e)
            result = candidate
            if candidate.layers:
                break
        if result is None:
            raise RuntimeError("RC briefing: LLM 未回傳可解析的結果")

        briefing = result.model_copy(update={"walk_overlap": overlap})

        referenced = set(overlap)
        for layer in briefing.layers:
            referenced.update(layer.node_eids)
        referenced.update(c.eid for c in briefing.contested_points)
        label_map = {n["element_id"]: n["label"] for n in tree.nodes}
        labels    = {e: label_map[e] for e in referenced if e in label_map}
        return briefing, labels, build_graph(tree)

    @staticmethod
    def validate(b: LLMBriefing, l2e: Dict[str, str]) -> RCBriefing:
        """label → eid 反查；找不到的丟棄，超出上限的項目截掉，空層整層移除。"""
        layers: List[Layer] = []
        for layer in b.layers[:MAX_LAYERS]:
            eids = list(dict.fromkeys(
                e for e in (_resolve(lb, l2e) for lb in layer.node_labels) if e
            ))
            if eids:
                layers.append(Layer(name=layer.name, node_eids=eids, role=layer.role))

        contested: List[ContestedPoint] = []
        seen: set = set()
        for c in b.contested_points:
            eid = _resolve(c.label, l2e)
            if eid and eid not in seen:
                seen.add(eid)
                contested.append(ContestedPoint(eid=eid, why=c.why))
        return RCBriefing(
            problem_frame=b.problem_frame,
            layers=layers,
            convergence=b.convergence,
            rc_role=b.rc_role,
            contested_points=contested[:MAX_CONTESTED],
            fit_gaps=[g for g in b.fit_gaps if g.strip()][:MAX_FIT_GAPS],
            handoff=b.handoff,
            walk_overlap=[],
        )


# ═══════════════════════════════════════════════════════════════
# Cache（記憶體 LRU；重啟後重新生成，成本可接受）
# ═══════════════════════════════════════════════════════════════

def context_hash(
    requirement_text: str,
    exploration_text: str = "",
    walk_visited:     Iterable[str] = (),
) -> str:
    """需求、探索軌跡、走過的節點任一改變都會產生新 hash（再跑一次慢想不會命中舊快取）。"""
    payload = json.dumps(
        [requirement_text.strip(), exploration_text.strip(), sorted(set(walk_visited))],
        ensure_ascii=False,
    )
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
    """快取／Firestore 讀回後統一 briefing 為 RCBriefing 實例。graph 保持 dict 或 None。"""
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
