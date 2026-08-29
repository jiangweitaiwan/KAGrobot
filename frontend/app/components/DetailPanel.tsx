"use client";

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

export type RCSemanticScore = {
  eid:           string;
  label:         string;
  score:         number;
  embed_score?:  number;
  intent_score?: number;
};

type Props = {
  node:              { id: string; name: string; type: string; source?: string } | null;
  detail:            ConceptDetail | null;
  foundationTree:    FoundationTreeData | null;
  loadingTree:       boolean;
  personalBrain:     PersonalBrainConcept[];
  rcSemanticScores:  RCSemanticScore[];
  rcFoundationMap:   RCFoundationMap;
  rcCategoryMap?:    Map<string, { category: string; color?: string | null }>;
  alpha:             number;
  fetchConceptDetail: (eid: string) => Promise<ConceptDetail | null>;
  onAddFoundations:  (eids: { eid: string; label: string }[]) => void;
  onRemove:          (eid: string) => void;
  onUpdateWeight:    (eid: string, weight: number) => void;
  onAskAboutNode:    (eid: string, label: string) => void;
};

// ─────────────────────────────────────────────────────────────
// Score computation（不變）
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
// leaf 與中繼節點皆可勾選加入 / 觸發 concept info 預覽
// ─────────────────────────────────────────────────────────────

type FNode = {
  id:       string;
  label:    string;
  isAnchor: boolean;
  isLeaf:   boolean;
  depth:    number;
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
  edge_base:      "#1e3050",
  edge_active:    "#e8a030",
  edge_foundation:"#534AB7",
};

function assignDepths(
  nodes: { id: string; isAnchor: boolean; isLeaf: boolean }[],
  edges: FEdge[],
): Map<string, number> {
  const childrenOf = new Map<string, string[]>();
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
  onNodeInfo,
  width,
  height,
}: {
  data:         FoundationTreeData;
  selectedEids: Set<string>;
  onToggle:     (eid: string, label: string) => void;
  onNodeInfo:   (eid: string, label: string) => void;
  width:        number;
  height:       number;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const leafEids = new Set(data.foundations.map((f) => f.element_id));

    const rawNodes: FNode[] = data.nodes.map((n) => ({
      id:       n.element_id,
      label:    n.label,
      isAnchor: n.element_id === data.anchor_eid,
      isLeaf:   leafEids.has(n.element_id),
      depth:    0,
      x:        0,
      y:        0,
    }));

    const rawEdges: FEdge[] = data.edges.map((e) => ({
      source: e.source,
      target: e.target,
    }));

    const depthMap = assignDepths(rawNodes, rawEdges);
    rawNodes.forEach((n) => { n.depth = depthMap.get(n.id) ?? 0; });

    const maxDepth = Math.max(...rawNodes.map((n) => n.depth));

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
      n.y = PAD_TOP + usableH * (1 - n.depth / Math.max(maxDepth, 1));
      const siblings = layerNodes.get(n.depth)!;
      const idx      = siblings.indexOf(n);
      const total    = siblings.length;
      n.x = PAD_SIDE + usableW * (total === 1 ? 0.5 : idx / (total - 1));
    });

    const nodeById = new Map(rawNodes.map((n) => [n.id, n]));

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

    const defs = root.append("defs");

    const filter = defs.append("filter").attr("id", "ft-glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    filter.append("feGaussianBlur").attr("in", "SourceGraphic").attr("stdDeviation", "3").attr("result", "blur");
    filter.append("feComposite").attr("in", "SourceGraphic").attr("in2", "blur").attr("operator", "over");

    const grad = defs.append("linearGradient")
      .attr("id", "ft-edge-grad")
      .attr("x1", "0%").attr("y1", "0%")
      .attr("x2", "0%").attr("y2", "100%");
    grad.append("stop").attr("offset", "0%").attr("stop-color",   "#534AB7").attr("stop-opacity", 0.9);
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#00d4aa").attr("stop-opacity", 0.7);

    const gradSel = defs.append("linearGradient")
      .attr("id", "ft-edge-grad-sel")
      .attr("x1", "0%").attr("y1", "0%")
      .attr("x2", "0%").attr("y2", "100%");
    gradSel.append("stop").attr("offset", "0%").attr("stop-color",   "#e8a030").attr("stop-opacity", 1);
    gradSel.append("stop").attr("offset", "100%").attr("stop-color", "#e8c070").attr("stop-opacity", 0.8);

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

    // ── 邊 ──
    const edgeG = zG.append("g").attr("class", "ft-edges");

    rawEdges.forEach((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      if (!s || !t) return;

      const sx = s.x!, sy = s.y!;
      const tx = t.x!, ty = t.y!;
      const cy1 = sy + (ty - sy) * 0.4;
      const cy2 = sy + (ty - sy) * 0.7;

      const isSelected = selectedEids.has(e.source) || selectedEids.has(e.target);
      const isToAnchor = t.isAnchor;

      edgeG.append("path")
        .attr("d", `M${sx},${sy} C${sx},${cy1} ${tx},${cy2} ${tx},${ty}`)
        .attr("fill",         "none")
        .attr("stroke",       isSelected ? "#e8a030" : (isToAnchor ? "#00d4aa" : "#534AB7"))
        .attr("stroke-width", isSelected ? 6 : 4)
        .attr("stroke-opacity", isSelected ? 0.18 : 0.10)
        .attr("stroke-linecap", "round");

      edgeG.append("path")
        .attr("d", `M${sx},${sy} C${sx},${cy1} ${tx},${cy2} ${tx},${ty}`)
        .attr("fill",         "none")
        .attr("stroke",       isSelected ? "url(#ft-edge-grad-sel)" : "url(#ft-edge-grad)")
        .attr("stroke-width", isSelected ? 2.2 : 1.4)
        .attr("stroke-opacity", isSelected ? 1 : 0.55)
        .attr("stroke-linecap", "round")
        .attr("marker-end", isSelected
          ? "url(#ft-arr-sel)"
          : (isToAnchor ? "url(#ft-arr-anc)" : "url(#ft-arr-base)"));
    });

    // ── 節點群組 ──
    const nodeG = zG.append("g").attr("class", "ft-nodes");

    const nodeGroup = nodeG
      .selectAll<SVGGElement, typeof rawNodes[0]>("g.ft-node")
      .data(rawNodes)
      .join("g")
      .attr("class", "ft-node")
      .attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`)
      .style("cursor", (d) => (!d.isAnchor) ? "pointer" : "default");

    // 光暈圓（所有非 anchor 節點都掛，確保 nth-child(2) 選擇器一致）
    nodeGroup
      .append("circle")
      .attr("r", (d) => d.isAnchor ? 26 : d.isLeaf ? 18 : 12)
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

    // 可勾選節點（leaf + 中繼）加小菱形裝飾，強調 clickable
    nodeGroup
      .filter((d) => !d.isAnchor)
      .append("polygon")
      .attr("points", (d) => d.isLeaf ? "0,-5 4,0 0,5 -4,0" : "0,-3.5 3,0 0,3.5 -3,0")
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

    // ── 互動：leaf + 中繼皆可點擊，同時勾選 + 觸發 info 預覽 ──
    nodeGroup
      .filter((d) => !d.isAnchor)
      .on("click", (_ev, d) => {
        onToggle(d.id, d.label);
        onNodeInfo(d.id, d.label);
      })
      .on("mouseenter", function (_ev, d) {
        d3.select(this).select("circle:nth-child(2)")
          .transition().duration(120)
          .attr("r", d.isLeaf ? 15 : 10)
          .attr("stroke-opacity", 1);
      })
      .on("mouseleave", function (_ev, d) {
        d3.select(this).select("circle:nth-child(2)")
          .transition().duration(120)
          .attr("r", d.isLeaf ? 12 : 7)
          .attr("stroke-opacity", 0.8);
      });

    root.append("text")
      .attr("x", PAD_SIDE).attr("y", 12)
      .attr("text-anchor", "start")
      .attr("font-size",   "8px")
      .attr("fill",        "rgba(127,119,221,0.35)")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("letter-spacing", "0.15em")
      .text("FOUNDATIONS ▲");
    root.append("text")
      .attr("x", PAD_SIDE).attr("y", height - 8)
      .attr("text-anchor", "start")
      .attr("font-size",   "8px")
      .attr("fill",        "#00d4aa44")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("letter-spacing", "0.15em")
      .text("▼ CONCEPT");

  }, [data, width, height, selectedEids, onToggle, onNodeInfo]);

  return (
    <svg
      ref={svgRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}

// ─────────────────────────────────────────────────────────────
// RankBar（不變）
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
// ConceptInfoCard（不變，接受 preview 或 anchor detail）
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

// ─────────────────────────────────────────────────────────────
// TabButton
// ─────────────────────────────────────────────────────────────

function TabButton({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: "0.6rem", padding: "2px 8px", borderRadius: 10,
        border: `1px solid ${active ? "var(--purple)" : "rgba(127,119,221,0.2)"}`,
        background: active ? "rgba(127,119,221,0.15)" : "transparent",
        color: active ? "var(--purple)" : "rgba(127,119,221,0.4)",
        fontFamily: "IBM Plex Mono, monospace", cursor: "pointer", letterSpacing: "0.06em",
      }}
    >
      {children}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────
// BrainTab — 知識重心管理（第二頁）
// ─────────────────────────────────────────────────────────────

function BrainTab({
  personalBrain, onUpdateWeight, onRemove,
}: {
  personalBrain:  PersonalBrainConcept[];
  onUpdateWeight: (eid: string, weight: number) => void;
  onRemove:       (eid: string) => void;
}) {
  const [expandedEid, setExpandedEid] = useState<string | null>(null);
  const sorted = useMemo(
    () => [...personalBrain].sort((a, b) => b.weight - a.weight),
    [personalBrain],
  );

  useEffect(() => {
    if (expandedEid && !sorted.some((c) => c.eid === expandedEid)) {
      setExpandedEid(null);
    }
  }, [sorted, expandedEid]);

  if (sorted.length === 0) {
    return (
      <div className="detail-empty">
        <span>尚未加入任何 concept</span>
        <span>回到「瀏覽」分頁挑選 foundation 加入</span>
      </div>
    );
  }

  const expanded = sorted.find((c) => c.eid === expandedEid) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, minHeight: 0, overflowY: "auto" }}>
      <div className="detail-section-label">
        知識重心
        <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
          · 共 {sorted.length} 個 · 點擊調整權重
        </span>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {sorted.map((c) => (
          <span
            key={c.eid}
            style={{
              fontSize: "0.6rem", padding: "2px 6px", borderRadius: 10,
              border:     `1px solid ${expandedEid === c.eid ? "var(--purple)" : "rgba(127,119,221,0.3)"}`,
              background: `rgba(83,74,183,${(c.weight * 0.25).toFixed(2)})`,
              color:      `rgba(175,169,236,${(0.5 + c.weight * 0.5).toFixed(2)})`,
              cursor:     "pointer",
            }}
            title={`weight: ${c.weight.toFixed(2)}`}
            onClick={() => setExpandedEid(expandedEid === c.eid ? null : c.eid)}
          >
            {c.label}
          </span>
        ))}
      </div>

      {expanded && (
        <div style={{
          display: "flex", flexDirection: "column", gap: 6,
          fontSize: "0.66rem", color: "var(--purple)",
          background: "rgba(6,8,15,0.5)", border: "1px solid rgba(83,74,183,0.2)",
          borderRadius: 8, padding: "8px 10px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ opacity: 0.8, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {expanded.label}
            </span>
            <span style={{ color: "#afa9ec", opacity: 0.5, fontSize: "0.58rem" }}>{expanded.source}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="range" min="0.1" max="1.0" step="0.05"
              value={expanded.weight}
              onChange={(e) => onUpdateWeight(expanded.eid, parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: "var(--purple)", height: 3 }}
            />
            <span style={{ color: "#afa9ec", width: 28, textAlign: "right" }}>{expanded.weight.toFixed(2)}</span>
          </div>
          <button
            onClick={() => { onRemove(expanded.eid); setExpandedEid(null); }}
            style={{
              alignSelf: "flex-start", fontSize: "0.6rem", padding: "3px 9px",
              border: "1px solid rgba(162,45,45,0.4)", borderRadius: 8,
              background: "transparent", color: "#e24b4a", cursor: "pointer",
              fontFamily: "IBM Plex Mono, monospace",
            }}
          >
            從 Personal Brain 移除
          </button>
        </div>
      )}
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
  rcCategoryMap,
  alpha,
  fetchConceptDetail,
  onAddFoundations,
  onRemove,
  onUpdateWeight,
  onAskAboutNode,
}: Props) {
  const [activeTab, setActiveTab] = useState<"browse" | "brain">("browse");

  const [selectedEids,  setSelectedEids]  = useState<Set<string>>(new Set());

  // ── concept info 預覽狀態（點擊 foundation tree 上任一非 anchor 節點觸發）──
  const [previewEid,     setPreviewEid]     = useState<string | null>(null);
  const [previewLabel,   setPreviewLabel]   = useState<string>("");
  const [previewDetail,  setPreviewDetail]  = useState<ConceptDetail | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  useEffect(() => {
    setSelectedEids(new Set());
    setPreviewEid(null);
    setPreviewDetail(null);
  }, [node?.id]);

  const handleToggle = useCallback((eid: string, _label: string) => {
    setSelectedEids((prev) => {
      const next = new Set(prev);
      if (next.has(eid)) next.delete(eid); else next.add(eid);
      return next;
    });
  }, []);

  const handleNodeInfo = useCallback(async (eid: string, label: string) => {
    setPreviewEid(eid);
    setPreviewLabel(label);
    setPreviewLoading(true);
    const d = await fetchConceptDetail(eid);
    setPreviewDetail(d);
    setPreviewLoading(false);
  }, [fetchConceptDetail]);

  const clearPreview = useCallback(() => {
    setPreviewEid(null);
    setPreviewDetail(null);
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

  const brainEntry = node ? (personalBrain.find((c) => c.eid === node.id) ?? null) : null;
  const isInBrain  = !!brainEntry;
  const isRC       = (node?.type || "").toLowerCase().includes("resulting");
  // ── RC 分類標籤（來自 GCS rc_categories.json，經 /graph/global-layout
  //    合併後由 page.tsx 建表傳入；查無分類時 undefined，不顯示）──────
  const categoryInfo = node ? rcCategoryMap?.get(node.id) : undefined;
  
  const activeDetail = previewLoading ? null : (previewDetail ?? detail);

  return (
    <div className="detail-panel" style={{ gap: 8 }}>

      {/* ── Tab header（永遠顯示）── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
        <span className="detail-label">Detail Panel</span>
        <div style={{ display: "flex", gap: 4 }}>
          <TabButton active={activeTab === "browse"} onClick={() => setActiveTab("browse")}>瀏覽</TabButton>
          <TabButton active={activeTab === "brain"}  onClick={() => setActiveTab("brain")}>重心</TabButton>
        </div>
      </div>

      {activeTab === "brain" ? (
        <BrainTab
          personalBrain={personalBrain}
          onUpdateWeight={onUpdateWeight}
          onRemove={onRemove}
        />
      ) : !node ? (
        <div className="detail-empty">
          <span>點擊 Global Brain</span>
          <span>探索 concept 的 foundations</span>
        </div>
      ) : (
        <>
          {/* ── Header ── */}
          <div className="detail-header">
            <div className="detail-title-row">
              <span className="detail-label">
                {isRC ? "Resulting Concept" : "Concept"}
              </span>
              <span className="detail-concept-name">
                {detail?.rc_name ?? node.name}
              </span>
              {categoryInfo && (
                <span style={{
                  fontSize: "0.6rem",
                  color:    categoryInfo.color || "var(--purple)",
                  opacity:  0.85,
                }}>
                  #{categoryInfo.category}
                </span>
              )}
            </div>
            <button
              className="detail-btn-secondary"
              style={{
                flexShrink: 0,
                alignSelf:  "flex-start",
                border:     "1px solid var(--teal)",
                background: "transparent",
                color:      "var(--teal)",
                whiteSpace: "nowrap",
              }}
              onClick={() => onAskAboutNode(node.id, detail?.rc_name ?? node.name)}
              title="把此概念的 eid 帶入對話框，讓 socrates 精確查詢"
            >
              💬 詢問這個
            </button>

            {isInBrain && (
              <span style={{
                fontSize: "0.6rem", color: "var(--teal)",
                flexShrink: 0, alignSelf: "flex-start", marginTop: 14,
              }}>◉</span>
            )}
          </div>

          {/* ── 預覽提示（點擊 foundation tree 節點時顯示）── */}
          {previewEid && (
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              fontSize: "0.62rem", color: "var(--orange)", flexShrink: 0,
            }}>
              <span>🔍 檢視中：{previewLabel}</span>
              <button
                onClick={clearPreview}
                style={{
                  fontSize: "0.58rem", padding: "1px 7px", borderRadius: 8,
                  border: "1px solid rgba(232,112,48,0.35)", background: "transparent",
                  color: "var(--orange)", cursor: "pointer",
                }}
              >
                返回 {node.name}
              </button>
            </div>
          )}

          {/* ── Concept Info ── */}
          {previewLoading ? (
            <div className="detail-loading-bar" style={{ marginBottom: 4 }} />
          ) : (
            <ConceptInfoCard detail={activeDetail} />
          )}

          {/* ── Foundation Tree ── */}
          <div style={{ flexShrink: 0 }}>
            <div className="detail-section-label" style={{ marginBottom: 4 }}>
              Foundation Tree
              <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
                · 點選節點加入 / 檢視
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
                  onNodeInfo={handleNodeInfo}
                  width={treeWidth}
                  height={treeHeight}
                />
              </div>
            ) : (
              <div style={{ color: "var(--muted)", fontSize: "0.68rem", padding: "8px 0", opacity: 0.5 }}>
                無 foundation 資料
              </div>
            )}

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

          {/* ── Key Assumptions ── */}
          {detail?.key_assumptions && !previewEid && (
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
        </>
      )}
    </div>
  );
}