"use client";

/**
 * DetailPanel.tsx（改版）
 *
 * 變更重點：
 *   1. FoundationTreeViz 改為分層佈局（foundations 頂部，anchor 底部），
 *      edge 以曲線（貝茲）畫出，呈現「從 foundations 匯聚到 concept」感。
 *   2. definition / mechanism 提升至 foundation tree 正上方，讓概念內容
 *      更容易看見，不再埋在最底部。
 *   3. RC 排名 semantic scores 由外部（page.tsx 混合策略）傳入，
 *      foundation 勾選的 interScore 仍在前端即時計算。
 */

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as d3 from "d3";

// ─────────────────────────────────────────────────────────────
// Exported types
// ─────────────────────────────────────────────────────────────

export type PersonalBrainConcept = {
  eid:     string;
  label:   string;
  source:  "manual" | "llm_suggest" | "path_confirm";
  weight:  number;
  addedAt: number;
};

export type ConceptDetail = {
  found:            boolean;
  eid:              string;
  rc_name:          string;
  definition:       string | null;
  mechanism:        string | null;
  key_assumptions:  string | null;
  function:         string | null;
  analysis_types:   string[];
  concept_type:     string | null;
  papers: { title: string; premises: string | null; constraints: string | null }[];
};

export type FoundationTreeData = {
  anchor_eid:   string;
  anchor_label: string;
  foundations:  { element_id: string; label: string }[];
  nodes:        { element_id: string; label: string }[];
  edges:        { source: string; target: string }[];
};



export type RCFoundationMap = Map<string, {
  label:          string;
  foundationEids: string[];
  treeNodes?:     { element_id: string; label: string }[];
  treeEdges?:     { source: string; target: string }[];
}>;

type Props = {
  node:              { id: string; name: string; type: string; source?: string } | null;
  detail:            ConceptDetail | null;
  foundationTree:    FoundationTreeData | null;
  loadingTree:       boolean;
  personalBrain:     PersonalBrainConcept[];
  rcSemanticScores:  RCSemanticScore[];
  rcFoundationMap:   RCFoundationMap;
  alpha:             number;
  onAddFoundations:  (eids: { eid: string; label: string }[]) => void;
  onRemove:          (eid: string) => void;
  onUpdateWeight:    (eid: string, weight: number) => void;
};

export type RCSemanticScore = {
  eid:           string;
  label:         string;
  score:         number;
  embed_score?:  number;
  intent_score?: number;
};

export type RecommendedFoundation = {
  element_id: string;
  label:      string;
  score:      number;
};

type Props = {
  // ...既有 props 不變
  recommendedFoundations?: RecommendedFoundation[];
};

// ─────────────────────────────────────────────────────────────
// Score computation
// ─────────────────────────────────────────────────────────────

type RCRanked = {
  eid:           string;
  label:         string;
  semantic:      number;
  interScore:    number;
  totalScore:    number;
  hits:          number;
  totalFound:    number;
};

function computeRanking(
  selectedEids:     Set<string>,
  personalBrain:    PersonalBrainConcept[],
  rcSemanticScores: RCSemanticScore[],
  rcFoundationMap:  RCFoundationMap,
  alpha:            number,
): RCRanked[] {
  const weightMap = new Map<string, number>();
  for (const c of personalBrain) {
    if (selectedEids.has(c.eid)) weightMap.set(c.eid, c.weight);
  }
  for (const eid of selectedEids) {
    if (!weightMap.has(eid)) weightMap.set(eid, 0.5);
  }
  const semMap = new Map(rcSemanticScores.map((s) => [s.eid, s.score]));
  const results: RCRanked[] = [];
  rcFoundationMap.forEach(({ label, foundationEids }, rcEid) => {
    const semantic  = semMap.get(rcEid) ?? 0;
    const n         = foundationEids.length;
    let weightedSum = 0;
    let hits        = 0;
    for (const feid of foundationEids) {
      const w = weightMap.get(feid);
      if (w !== undefined) { weightedSum += w; hits++; }
    }
    const interScore = n > 0 ? weightedSum / n : 0;
    results.push({
      eid: rcEid, label,
      semantic, interScore,
      totalScore: alpha * semantic + (1 - alpha) * interScore,
      hits, totalFound: n,
    });
  });
  return results.sort((a, b) => b.totalScore - a.totalScore);
}

// ─────────────────────────────────────────────────────────────
// FoundationTreeViz — 分層佈局 + 曲線 edge
// ─────────────────────────────────────────────────────────────

type FNode = {
  id:       string;
  label:    string;
  isAnchor: boolean;
  isLeaf:   boolean;
  depth:    number;   // 0 = anchor, 增加 = 越靠近 foundation
  x?: number;
  y?: number;
};

type FEdge = {
  source: string;
  target: string;
};

const T = {
  bg:             "transparent",
  anchor:         "#00d4aa",
  anchor_glow:    "#00d4aa44",
  foundation:     "#7f77dd",
  foundation_dim: "#4a3a8a",
  intermediate:   "#2d4a7a",
  selected:       "#e8a030",
  selected_glow:  "#e8a03044",
  text:           "#6a7a9a",
  text_leaf:      "#9090cc",
  text_anchor:    "#00d4aa",
  text_selected:  "#e8c070",
  // edge colours — gradient from foundation (top) to anchor (bottom)
  edge_base:      "#1e3050",
  edge_active:    "#e8a030",
  edge_foundation:"#534AB7",
};

/**
 * 計算分層 y 座標：BFS 從 anchor 往上追，depth 越大越靠近 foundation。
 * anchor → y 最大（底部），foundations → y 最小（頂部）。
 */
function assignDepths(
  nodes: { id: string; isAnchor: boolean; isLeaf: boolean }[],
  edges: FEdge[],
): Map<string, number> {
  // edges 方向：foundation → anchor（source 更上游，target 更下游）
  // 我們想讓 anchor depth=0，沿 edge 往 source 方向 depth+1
  const childrenOf = new Map<string, string[]>(); // target → sources
  for (const e of edges) {
    if (!childrenOf.has(e.target)) childrenOf.set(e.target, []);
    childrenOf.get(e.target)!.push(e.source);
  }

  const depthMap = new Map<string, number>();
  const anchor = nodes.find((n) => n.isAnchor);
  if (!anchor) {
    nodes.forEach((n, i) => depthMap.set(n.id, i));
    return depthMap;
  }

  // BFS from anchor upward
  const queue: string[] = [anchor.id];
  depthMap.set(anchor.id, 0);
  while (queue.length) {
    const cur = queue.shift()!;
    const curDepth = depthMap.get(cur)!;
    for (const parent of (childrenOf.get(cur) ?? [])) {
      if (!depthMap.has(parent)) {
        depthMap.set(parent, curDepth + 1);
        queue.push(parent);
      }
    }
  }
  // 未被 BFS 到的節點放在最大 depth
  const maxDepth = Math.max(0, ...depthMap.values());
  nodes.forEach((n) => {
    if (!depthMap.has(n.id)) depthMap.set(n.id, maxDepth + 1);
  });
  return depthMap;
}

function FoundationTreeViz({
  data,
  selectedEids,
  onToggle,
  width,
  height,
}: {
  data:         FoundationTreeData;
  selectedEids: Set<string>;
  onToggle:     (eid: string, label: string) => void;
  width:        number;
  height:       number;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const leafEids = new Set(data.foundations.map((f) => f.element_id));

    // ── 1. 建立節點與邊 ───────────────────────────────────────
    const rawNodes: FNode[] = data.nodes.map((n) => ({
      id:       n.element_id,
      label:    n.label,
      isAnchor: n.element_id === data.anchor_eid,
      isLeaf:   leafEids.has(n.element_id),
      depth:    0,
      x:        0,  // 初始化为 0，之后会计算
      y:        0,  // 初始化为 0，之后会计算
    }));

    const rawEdges: FEdge[] = data.edges.map((e) => ({
      source: e.source,
      target: e.target,
    }));

    // ── 2. 計算分層 depth ──────────────────────────────────────
    const depthMap = assignDepths(rawNodes, rawEdges);
    rawNodes.forEach((n) => { n.depth = depthMap.get(n.id) ?? 0; });

    const maxDepth = Math.max(...rawNodes.map((n) => n.depth));
    const numLayers = maxDepth + 1;

    // ── 3. 計算每個節點的 x, y（分層佈局）──────────────────────
    // y: depth=0（anchor）在底部, depth=maxDepth（foundations）在頂部
    const PAD_TOP    = 28;
    const PAD_BOTTOM = 36;
    const PAD_SIDE   = 24;
    const usableH    = height - PAD_TOP - PAD_BOTTOM;
    const usableW    = width  - PAD_SIDE * 2;

    const layerNodes: Map<number, typeof rawNodes> = new Map();
    rawNodes.forEach((n) => {
      if (!layerNodes.has(n.depth)) layerNodes.set(n.depth, []);
      layerNodes.get(n.depth)!.push(n);
    });

    rawNodes.forEach((n) => {
      // y: anchor (depth=0) → bottom; maxDepth → top
      n.y = PAD_TOP + usableH * (1 - n.depth / Math.max(maxDepth, 1));

      // x: 在同一層均勻分佈
      const siblings = layerNodes.get(n.depth)!;
      const idx      = siblings.indexOf(n);
      const total    = siblings.length;
      n.x = PAD_SIDE + usableW * (total === 1 ? 0.5 : idx / (total - 1));
    });

    const nodeById = new Map(rawNodes.map((n) => [n.id, n]));

    // ── 4. SVG 根元素 + zoom ───────────────────────────────────
    const root = svg
      .attr("width",  width)
      .attr("height", height)
      .style("background", T.bg);

    const zG = root.append("g");
    root.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.4, 2.5])
        .on("zoom", (ev) => zG.attr("transform", ev.transform))
    );

    // ── 5. Defs：gradient 箭頭 marker + glow filter ────────────
    const defs = root.append("defs");

    // glow filter
    const filter = defs.append("filter").attr("id", "ft-glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    filter.append("feGaussianBlur").attr("in", "SourceGraphic").attr("stdDeviation", "3").attr("result", "blur");
    filter.append("feComposite").attr("in", "SourceGraphic").attr("in2", "blur").attr("operator", "over");

    // edge gradient (foundation-purple at top → teal at bottom)
    const grad = defs.append("linearGradient")
      .attr("id", "ft-edge-grad")
      .attr("x1", "0%").attr("y1", "0%")
      .attr("x2", "0%").attr("y2", "100%");
    grad.append("stop").attr("offset", "0%").attr("stop-color",   "#534AB7").attr("stop-opacity", 0.9);
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#00d4aa").attr("stop-opacity", 0.7);

    // selected edge gradient
    const gradSel = defs.append("linearGradient")
      .attr("id", "ft-edge-grad-sel")
      .attr("x1", "0%").attr("y1", "0%")
      .attr("x2", "0%").attr("y2", "100%");
    gradSel.append("stop").attr("offset", "0%").attr("stop-color",   "#e8a030").attr("stop-opacity", 1);
    gradSel.append("stop").attr("offset", "100%").attr("stop-color", "#e8c070").attr("stop-opacity", 0.8);

    // arrowhead marker
    const mkMarker = (id: string, color: string) => {
      defs.append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -4 8 8")
        .attr("refX", 7).attr("refY", 0)
        .attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L8,0L0,4")
        .attr("fill", color);
    };
    mkMarker("ft-arr-base", "#534AB7");
    mkMarker("ft-arr-sel",  "#e8a030");
    mkMarker("ft-arr-anc",  "#00d4aa");

    // ── 6. 邊（cubic bezier 曲線，從 source 匯聚向 target）────
    const edgeG = zG.append("g").attr("class", "ft-edges");

    rawEdges.forEach((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      if (!s || !t) return;

      const sx = s.x!, sy = s.y!;
      const tx = t.x!, ty = t.y!;

      // 控制點：水平在中點，垂直偏向 source（讓曲線更自然）
      const cy1 = sy + (ty - sy) * 0.4;
      const cy2 = sy + (ty - sy) * 0.7;

      const isSelected = selectedEids.has(e.source) || selectedEids.has(e.target);
      const isToAnchor = t.isAnchor;

      // 底層 glow（粗線，低透明）
      edgeG.append("path")
        .attr("d", `M${sx},${sy} C${sx},${cy1} ${tx},${cy2} ${tx},${ty}`)
        .attr("fill",         "none")
        .attr("stroke",       isSelected ? "#e8a030" : (isToAnchor ? "#00d4aa" : "#534AB7"))
        .attr("stroke-width", isSelected ? 6 : 4)
        .attr("stroke-opacity", isSelected ? 0.18 : 0.10)
        .attr("stroke-linecap", "round");

      // 主線
      edgeG.append("path")
        .attr("d", `M${sx},${sy} C${sx},${cy1} ${tx},${cy2} ${tx},${ty}`)
        .attr("fill",         "none")
        .attr("stroke",       isSelected
          ? "url(#ft-edge-grad-sel)"
          : (isToAnchor ? "url(#ft-edge-grad)" : "url(#ft-edge-grad)"))
        .attr("stroke-width", isSelected ? 2.2 : 1.4)
        .attr("stroke-opacity", isSelected ? 1 : 0.55)
        .attr("stroke-linecap", "round")
        .attr("marker-end", isSelected
          ? "url(#ft-arr-sel)"
          : (isToAnchor ? "url(#ft-arr-anc)" : "url(#ft-arr-base)"));
    });

    // ── 7. 節點群組 ───────────────────────────────────────────
    const nodeG = zG.append("g").attr("class", "ft-nodes");

    const nodeGroup = nodeG
      .selectAll<SVGGElement, typeof rawNodes[0]>("g.ft-node")
      .data(rawNodes)
      .join("g")
      .attr("class", "ft-node")
      .attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`)
      .style("cursor", (d) => (d.isLeaf && !d.isAnchor) ? "pointer" : "default");

    // 光暈圓（背景 glow）
    nodeGroup
      .filter((d) => d.isAnchor || d.isLeaf)
      .append("circle")
      .attr("r", (d) => d.isAnchor ? 26 : 18)
      .attr("fill", (d) => {
        if (d.isAnchor)             return T.anchor_glow;
        if (selectedEids.has(d.id)) return T.selected_glow;
        return T.foundation_dim;
      })
      .attr("fill-opacity", 0.25)
      .attr("filter", "url(#ft-glow)");

    // 主圓
    nodeGroup
      .append("circle")
      .attr("r", (d) => d.isAnchor ? 18 : d.isLeaf ? 12 : 7)
      .attr("fill", (d) => {
        if (d.isAnchor)             return T.anchor;
        if (selectedEids.has(d.id)) return T.selected;
        if (d.isLeaf)               return T.foundation;
        return T.intermediate;
      })
      .attr("stroke", (d) => {
        if (d.isAnchor)             return "#00ffcc";
        if (selectedEids.has(d.id)) return "#e8c070";
        if (d.isLeaf)               return "#7f77dd";
        return "transparent";
      })
      .attr("stroke-width", (d) => d.isAnchor ? 2 : 1.5)
      .attr("stroke-opacity", 0.8);

    // leaf 節點加一個小菱形裝飾（強調 clickable）
    nodeGroup
      .filter((d) => d.isLeaf && !d.isAnchor)
      .append("polygon")
      .attr("points", "0,-5 4,0 0,5 -4,0")
      .attr("fill", (d) => selectedEids.has(d.id) ? "#e8a030" : "#534AB7")
      .attr("opacity", 0.9);

    // 標籤
    nodeGroup
      .append("text")
      .attr("dy", (d) => d.isAnchor ? 30 : d.isLeaf ? 22 : 16)
      .attr("text-anchor", "middle")
      .attr("font-size",   (d) => d.isAnchor ? "9px" : "8px")
      .attr("fill", (d) => {
        if (d.isAnchor)             return T.text_anchor;
        if (selectedEids.has(d.id)) return T.text_selected;
        if (d.isLeaf)               return T.text_leaf;
        return T.text;
      })
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("pointer-events", "none")
      .text((d) => d.label.length > 18 ? d.label.slice(0, 17) + "…" : d.label);

    // ── 8. 互動 ─────────────────────────────────────────────
    nodeGroup
      .filter((d) => d.isLeaf && !d.isAnchor)
      .on("click", (_ev, d) => onToggle(d.id, d.label))
      .on("mouseenter", function (_ev, d) {
        d3.select(this).select("circle:nth-child(2)")
          .transition().duration(120)
          .attr("r", 15)
          .attr("stroke-opacity", 1);
      })
      .on("mouseleave", function (_ev, d) {
        d3.select(this).select("circle:nth-child(2)")
          .transition().duration(120)
          .attr("r", 12)
          .attr("stroke-opacity", 0.8);
      });

    // 層標籤（頂部：Foundation，底部：Concept）
    const labelStyle = {
      fontSize:      "8px",
      fill:          "rgba(127,119,221,0.35)",
      fontFamily:    "IBM Plex Mono, monospace",
      letterSpacing: "0.15em",
      pointerEvents: "none" as const,
    };
    root.append("text")
      .attr("x", PAD_SIDE).attr("y", 12)
      .attr("text-anchor", "start")
      .attr("font-size",   labelStyle.fontSize)
      .attr("fill",        labelStyle.fill)
      .attr("font-family", labelStyle.fontFamily)
      .attr("letter-spacing", labelStyle.letterSpacing)
      .text("FOUNDATIONS ▲");
    root.append("text")
      .attr("x", PAD_SIDE).attr("y", height - 8)
      .attr("text-anchor", "start")
      .attr("font-size",   labelStyle.fontSize)
      .attr("fill",        "#00d4aa44")
      .attr("font-family", labelStyle.fontFamily)
      .attr("letter-spacing", labelStyle.letterSpacing)
      .text("▼ CONCEPT");

  }, [data, width, height, selectedEids]);

  return (
    <svg
      ref={svgRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}

// ─────────────────────────────────────────────────────────────
// RankingBar
// ─────────────────────────────────────────────────────────────

function RankBar({
  label, score, max, personal, rank,
}: {
  label: string; score: number; max: number; personal: boolean; rank: number;
}) {
  const pct = max > 0 ? (score / max) * 100 : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
      <span style={{ color: "var(--muted)", fontSize: "0.6rem", width: 12, flexShrink: 0 }}>{rank}.</span>
      <span style={{
        color: personal ? "var(--teal)" : "#9090b0",
        fontSize: "0.68rem", width: 90,
        overflow: "hidden", textOverflow: "ellipsis",
        whiteSpace: "nowrap", flexShrink: 0,
      }} title={label}>{label}</span>
      <div style={{ flex: 1, height: 6, background: "rgba(83,74,183,0.15)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 3,
          background: personal
            ? "linear-gradient(90deg,#534AB7,#00d4aa)"
            : "#534AB7",
          transition: "width 0.35s ease",
        }} />
      </div>
      <span style={{ color: "#afa9ec", fontSize: "0.62rem", width: 32, textAlign: "right", flexShrink: 0 }}>
        {score.toFixed(2)}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// ConceptInfoCard — definition + mechanism 卡片
// ─────────────────────────────────────────────────────────────

function ConceptInfoCard({ detail }: { detail: ConceptDetail | null }) {
  const [expanded, setExpanded] = useState(false);

  if (!detail?.found) return null;
  const hasContent = detail.definition || detail.mechanism || detail.key_assumptions || detail.function;
  if (!hasContent) return null;

  const PREVIEW = 120;

  return (
    <div style={{
      background:    "rgba(6,8,15,0.6)",
      border:        "1px solid rgba(83,74,183,0.25)",
      borderRadius:  10,
      padding:       "10px 12px",
      flexShrink:    0,
    }}>
      {/* Definition */}
      {detail.definition && (
        <div style={{ marginBottom: detail.mechanism ? 8 : 0 }}>
          <span style={{
            fontSize: "0.58rem", letterSpacing: "0.15em",
            textTransform: "uppercase", color: "var(--purple)", opacity: 0.55,
            display: "block", marginBottom: 3,
          }}>
            Definition
          </span>
          <p style={{ margin: 0, fontSize: "0.7rem", color: "#9090c0", lineHeight: 1.6 }}>
            {expanded
              ? detail.definition
              : detail.definition.slice(0, PREVIEW) + (detail.definition.length > PREVIEW ? "…" : "")}
          </p>
        </div>
      )}

      {/* Mechanism */}
      {detail.mechanism && (
        <div style={{ marginTop: detail.definition ? 8 : 0 }}>
          <span style={{
            fontSize: "0.58rem", letterSpacing: "0.15em",
            textTransform: "uppercase", color: "var(--purple)", opacity: 0.55,
            display: "block", marginBottom: 3,
          }}>
            Mechanism
          </span>
          <p style={{ margin: 0, fontSize: "0.7rem", color: "#9090c0", lineHeight: 1.6 }}>
            {expanded
              ? detail.mechanism
              : detail.mechanism.slice(0, PREVIEW) + (detail.mechanism.length > PREVIEW ? "…" : "")}
          </p>
        </div>
      )}

      {/* RC-only: function */}
      {detail.function && (
        <div style={{ marginTop: 8 }}>
          <span style={{
            fontSize: "0.58rem", letterSpacing: "0.15em",
            textTransform: "uppercase", color: "var(--teal)", opacity: 0.6,
            display: "block", marginBottom: 3,
          }}>
            Function
            <span style={{ marginLeft: 6, fontSize: "0.52rem", opacity: 0.7 }}>· RC</span>
          </span>
          <p style={{ margin: 0, fontSize: "0.7rem", color: "#80b0a0", lineHeight: 1.6 }}>
            {expanded
              ? detail.function
              : detail.function.slice(0, PREVIEW) + (detail.function.length > PREVIEW ? "…" : "")}
          </p>
        </div>
      )}

      {/* expand toggle */}
      {(
        (detail.definition?.length ?? 0) > PREVIEW ||
        (detail.mechanism?.length  ?? 0) > PREVIEW ||
        (detail.function?.length   ?? 0) > PREVIEW
      ) && (
        <button
          onClick={() => setExpanded((v) => !v)}
          style={{
            marginTop: 6, padding: "2px 0",
            background: "transparent", border: "none",
            color: "rgba(127,119,221,0.5)", fontSize: "0.62rem",
            cursor: "pointer", fontFamily: "IBM Plex Mono, monospace",
            letterSpacing: "0.08em",
          }}
        >
          {expanded ? "▲ 收合" : "▼ 展開全文"}
        </button>
      )}
    </div>
  );
}

function RecommendedFoundationsCard({
  items, onAdd,
}: {
  items: RecommendedFoundation[];
  onAdd: (eid: string, label: string) => void;
}) {
  if (!items.length) return null;
  return (
    <div style={{ flexShrink: 0 }}>
      <div className="detail-section-label" style={{ marginBottom: 4 }}>
        語意推薦 Foundation
        <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
          · 依對話語意匹配
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {items.map((f) => (
          <span
            key={f.element_id}
            style={{
              fontSize: "0.62rem", padding: "2px 8px", borderRadius: 12,
              border:     "1px solid rgba(0,212,170,0.35)",
              background: "rgba(0,212,170,0.06)",
              color:      "var(--teal)",
              cursor:     "pointer",
            }}
            title={`語意分數 ${f.score.toFixed(2)}`}
            onClick={() => onAdd(f.element_id, f.label)}
          >
            + {f.label}
          </span>
        ))}
      </div>
    </div>
  );
}
// ─────────────────────────────────────────────────────────────
// DetailPanel
// ─────────────────────────────────────────────────────────────

export default function DetailPanel({
  node,
  detail,
  foundationTree,
  loadingTree,
  personalBrain,
  rcSemanticScores,
  rcFoundationMap,
  alpha,
  onAddFoundations,
  onRemove,
  onUpdateWeight,
}: Props) {
  const [selectedEids,  setSelectedEids]  = useState<Set<string>>(new Set());
  const [showWeightFor, setShowWeightFor] = useState<string | null>(null);

  useEffect(() => {
    setSelectedEids(new Set());
    setShowWeightFor(null);
  }, [node?.id]);

  const handleToggle = useCallback((eid: string, _label: string) => {
    setSelectedEids((prev) => {
      const next = new Set(prev);
      if (next.has(eid)) next.delete(eid); else next.add(eid);
      return next;
    });
  }, []);

  const previewBrain = useMemo<PersonalBrainConcept[]>(() => {
    if (!foundationTree) return personalBrain;
    const base     = [...personalBrain];
    const brainMap = new Map(base.map((c) => [c.eid, c]));
    for (const eid of selectedEids) {
      if (!brainMap.has(eid)) {
        const fNode = foundationTree.nodes.find((n) => n.element_id === eid);
        base.push({ eid, label: fNode?.label ?? eid, source: "manual", weight: 0.5, addedAt: Date.now() });
      }
    }
    return base;
  }, [selectedEids, personalBrain, foundationTree]);

  const rankingBefore = useMemo(
    () => computeRanking(
      new Set(personalBrain.map((c) => c.eid)),
      personalBrain, rcSemanticScores, rcFoundationMap, alpha,
    ).slice(0, 3),
    [personalBrain, rcSemanticScores, rcFoundationMap, alpha],
  );
  const rankingAfter = useMemo(
    () => computeRanking(
      new Set(previewBrain.map((c) => c.eid)),
      previewBrain, rcSemanticScores, rcFoundationMap, alpha,
    ).slice(0, 3),
    [previewBrain, rcSemanticScores, rcFoundationMap, alpha],
  );

  const hasSelection = selectedEids.size > 0;
  const maxBefore    = rankingBefore[0]?.totalScore ?? 1;
  const maxAfter     = rankingAfter[0]?.totalScore  ?? 1;
  const beforePositions = useMemo(
    () => new Map(rankingBefore.map((r, i) => [r.eid, i + 1])),
    [rankingBefore],
  );

  const handleConfirmAdd = useCallback(() => {
    if (!foundationTree) return;
    const toAdd = [...selectedEids].map((eid) => {
      const fn = foundationTree.nodes.find((n) => n.element_id === eid);
      return { eid, label: fn?.label ?? eid };
    });
    onAddFoundations(toAdd);
    setSelectedEids(new Set());
  }, [selectedEids, foundationTree, onAddFoundations]);

  const treeWidth  = 260;
  const treeHeight = 200;

  if (!node) {
    return (
      <div className="detail-panel">
        <div className="detail-label" style={{ flexShrink: 0 }}>Detail Panel</div>
        {recommendedFoundations && recommendedFoundations.length > 0 ? (
          <RecommendedFoundationsCard
            items={recommendedFoundations}
            onAdd={(eid, label) => onAddFoundations([{ eid, label }])}
          />
        ) : (
          <div className="detail-empty">
            <span>點擊 Global Brain</span>
            <span>探索 concept 的 foundations</span>
          </div>
        )}
      </div>
    );
  }

  const brainEntry = personalBrain.find((c) => c.eid === node.id) ?? null;
  const isInBrain  = !!brainEntry;
  const isRC       = (node.type || "").toLowerCase().includes("resulting");

  return (
    <div className="detail-panel" style={{ gap: 8 }}>

      {/* ── Header ── */}
      <div className="detail-header">
        <div className="detail-title-row">
          <span className="detail-label">
            {isRC ? "Resulting Concept" : "Concept"}
          </span>
          <span className="detail-concept-name">
            {detail?.rc_name ?? node.name}
          </span>
        </div>
        {isInBrain && (
          <span style={{
            fontSize: "0.6rem", color: "var(--teal)",
            flexShrink: 0, alignSelf: "flex-start", marginTop: 14,
          }}>◉</span>
        )}
      </div>

      {/* ── Concept Info（definition + mechanism）── 提升至頂部 ── */}
      <ConceptInfoCard detail={detail} />

      {recommendedFoundations && recommendedFoundations.length > 0 && (
        <RecommendedFoundationsCard
          items={recommendedFoundations}
          onAdd={(eid, label) => onAddFoundations([{ eid, label }])}
        />
      )}
      {/* ── Foundation Tree ── */}
      <div style={{ flexShrink: 0 }}>
        <div className="detail-section-label" style={{ marginBottom: 4 }}>
          Foundation Tree
          <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
            · 點選 ◆ foundation 加入
          </span>
        </div>

        {loadingTree ? (
          <div className="detail-loading-bar" style={{ marginBottom: 4 }} />
        ) : foundationTree && foundationTree.nodes.length > 0 ? (
          <div style={{
            border:     "1px solid rgba(83,74,183,0.2)",
            borderRadius: 10,
            overflow:   "hidden",
            background: "rgba(6,8,15,0.55)",
            position:   "relative",
          }}>
            <FoundationTreeViz
              data={foundationTree}
              selectedEids={selectedEids}
              onToggle={handleToggle}
              width={treeWidth}
              height={treeHeight}
            />
          </div>
        ) : (
          <div style={{ color: "var(--muted)", fontSize: "0.68rem", padding: "8px 0", opacity: 0.5 }}>
            無 foundation 資料
          </div>
        )}

        {/* Selected chips */}
        {hasSelection && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 6 }}>
            {[...selectedEids].map((eid) => {
              const fn        = foundationTree?.nodes.find((n) => n.element_id === eid);
              const alreadyIn = personalBrain.some((c) => c.eid === eid);
              return (
                <span
                  key={eid}
                  style={{
                    fontSize: "0.62rem", padding: "2px 7px", borderRadius: 12,
                    border:     `1px solid ${alreadyIn ? "var(--teal)" : "var(--orange)"}`,
                    background: alreadyIn ? "rgba(0,212,170,0.08)" : "rgba(232,160,48,0.1)",
                    color:      alreadyIn ? "var(--teal)" : "var(--orange)",
                    cursor:     "pointer",
                  }}
                  onClick={() => handleToggle(eid, fn?.label ?? eid)}
                >
                  {alreadyIn ? "✓ " : "◆ "}{fn?.label ?? eid}
                </span>
              );
            })}
          </div>
        )}
      </div>

      {/* ── RC Ranking preview ── */}
      {rcSemanticScores.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 6 }}>
            {hasSelection ? "RC 排名預覽（加入後）" : "RC 排名"}
          </div>

          {hasSelection ? (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div>
                <div style={{ fontSize: "0.58rem", color: "var(--muted)", marginBottom: 4, opacity: 0.6, letterSpacing: "0.1em" }}>BEFORE</div>
                {rankingBefore.map((r, i) => (
                  <RankBar key={r.eid} label={r.label} score={r.totalScore} max={maxBefore} personal={false} rank={i + 1} />
                ))}
              </div>
              <div>
                <div style={{ fontSize: "0.58rem", color: "var(--teal)", marginBottom: 4, opacity: 0.8, letterSpacing: "0.1em" }}>AFTER ▶</div>
                {rankingAfter.map((r, i) => {
                  const prevRank  = beforePositions.get(r.eid);
                  const rankDelta = prevRank !== undefined ? prevRank - (i + 1) : null;
                  return (
                    <div key={r.eid} style={{ display: "flex", alignItems: "center", gap: 3, marginBottom: 4 }}>
                      <div style={{ flex: 1 }}>
                        <RankBar label={r.label} score={r.totalScore} max={maxAfter} personal rank={i + 1} />
                      </div>
                      {rankDelta !== null && rankDelta !== 0 && (
                        <span style={{ fontSize: "0.58rem", color: rankDelta > 0 ? "var(--teal)" : "#e85d60", flexShrink: 0, width: 18, textAlign: "right" }}>
                          {rankDelta > 0 ? `▲${rankDelta}` : `▼${Math.abs(rankDelta)}`}
                        </span>
                      )}
                      {prevRank === undefined && (
                        <span style={{ fontSize: "0.58rem", color: "var(--orange)", flexShrink: 0, width: 18, textAlign: "right" }}>NEW</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div>
              {rankingBefore.map((r, i) => (
                <RankBar key={r.eid} label={r.label} score={r.totalScore} max={maxBefore} personal={false} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Intersection breakdown ── */}
      {hasSelection && rankingAfter.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>交集說明</div>
          {rankingAfter.slice(0, 2).map((r) => (
            <div key={r.eid} style={{ display: "flex", gap: 6, alignItems: "baseline", marginBottom: 3, fontSize: "0.66rem" }}>
              <span style={{ color: "var(--purple)", opacity: 0.7, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 90 }} title={r.label}>{r.label}</span>
              <span style={{ color: "var(--muted)" }}>·</span>
              <span style={{ color: "#afa9ec" }}>{r.hits}/{r.totalFound} matched</span>
              <span style={{ color: "var(--muted)", marginLeft: "auto" }}>inter={r.interScore.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── 知識重心（按 weight 排序）── */}
      {personalBrain.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>知識重心</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {[...personalBrain]
              .sort((a, b) => b.weight - a.weight)
              .slice(0, 6)
              .map((c) => (
                <span
                  key={c.eid}
                  style={{
                    fontSize: "0.6rem", padding: "2px 6px", borderRadius: 10,
                    border:     "1px solid rgba(127,119,221,0.3)",
                    background: `rgba(83,74,183,${(c.weight * 0.25).toFixed(2)})`,
                    color:      `rgba(175,169,236,${(0.5 + c.weight * 0.5).toFixed(2)})`,
                    cursor:     "pointer",
                  }}
                  title={`weight: ${c.weight.toFixed(2)}`}
                  onClick={() => setShowWeightFor(showWeightFor === c.eid ? null : c.eid)}
                >
                  {c.label}
                </span>
              ))}
          </div>
          {showWeightFor && (() => {
            const c = personalBrain.find((x) => x.eid === showWeightFor);
            if (!c) return null;
            return (
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, fontSize: "0.66rem", color: "var(--purple)" }}>
                <span style={{ opacity: 0.7, minWidth: 60, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.label}</span>
                <input
                  type="range" min="0.1" max="1.0" step="0.05"
                  value={c.weight}
                  onChange={(e) => onUpdateWeight(c.eid, parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: "var(--purple)", height: 3 }}
                />
                <span style={{ color: "#afa9ec", width: 28, textAlign: "right" }}>{c.weight.toFixed(2)}</span>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── Key Assumptions（維持在底部）── */}
      {detail?.key_assumptions && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>Key Assumptions</div>
          <p style={{ margin: 0, fontSize: "0.68rem", color: "#8880c0", lineHeight: 1.55 }}>
            {detail.key_assumptions.slice(0, 160)}
            {detail.key_assumptions.length > 160 ? "…" : ""}
          </p>
        </div>
      )}

      {/* ── Actions ── */}
      <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 6, flexShrink: 0, paddingTop: 4 }}>
        {hasSelection && (
          <button className="detail-btn-add" onClick={handleConfirmAdd}>
            + 加入 {selectedEids.size} 個 foundation 到 Personal Brain
          </button>
        )}
        {isInBrain ? (
          <button className="detail-btn-secondary detail-btn-remove" onClick={() => onRemove(node.id)}>
            從 Personal Brain 移除此 concept
          </button>
        ) : (
          <button
            className="detail-btn-add"
            style={{ background: "rgba(13,18,32,0.5)", borderColor: "rgba(0,212,170,0.2)", color: "rgba(0,212,170,0.4)" }}
            onClick={() => onAddFoundations([{ eid: node.id, label: node.name }])}
          >
            + 加入此 concept（無 foundation 選取）
          </button>
        )}
      </div>
    </div>
  );
}