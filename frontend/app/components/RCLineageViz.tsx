"use client";

/**
 * RCLineageViz.tsx
 *
 * 垂直剖面圖(取代 PersonalBrain 原本的 3D fMRI 場景)。
 *
 * 相對於 BertopicMap(水平剖面圖,看得到所有 concept 的群聚分佈),
 * 這個元件是「垂直剖面圖」:針對『目前聚焦的單一 RC』,往上長出它的
 * foundation_tree(理論基礎溯源),往下長出它的 downstream 鏈
 * (這個 RC 再往後可以連到哪些其他 RC / skill),讓使用者在套用某個
 * skill 之前,看得到它的完整來歷。
 *
 * 資料來源(由 PersonalBrain.tsx 組裝好傳入,這裡只負責畫圖):
 *   - foundationTree : /graph/expand/{focusedEid} 的結果(與 DetailPanel
 *                      共用同一份 FoundationTreeData 型別)
 *   - downstreamChains: 從 /graph/rc-paths/{focusedEid} 的 paths[] 裡,
 *                      篩出「以 focusedEid 為起點、屬於 Resulting_Concept
 *                      類型節點」的子序列所組成的一組鏈(見
 *                      PersonalBrain.tsx 的 buildDownstreamChains())
 *
 * 互動:
 *   - foundation 節點唯讀,滑鼠停留顯示名稱;若該 foundation 已存在於
 *     Personal Brain,以較深/飽和的顏色呈現。
 *   - downstream 鏈上的每一顆節點皆可點擊 → 呼叫 onFocusNode(eid,label)
 *     把焦點換到那個節點,圖會以新的焦點重新渲染(由外部觸發重新 fetch)。
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { FoundationTreeData } from "./DetailPanel";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type ChainStep = { eid: string; label: string };

type Props = {
  focusedEid:        string;
  focusedLabel:       string;
  foundationTree:     FoundationTreeData | null;
  downstreamChains:   ChainStep[][];
  personalBrainEids:  Set<string>;
  loading?:           boolean;
  onFocusNode:        (eid: string, label: string) => void;
  width:              number;
  height:             number;
};

type VKind = "foundation-leaf" | "foundation-mid" | "focus" | "downstream";

type VNode = {
  id:    string;
  label: string;
  kind:  VKind;
  depth: number;   // 負 = 上(foundation)、0 = focus、正 = 下(downstream)
  x?:    number;
  y?:    number;
};

type VEdge = { source: string; target: string; kind: "foundation" | "downstream" };

// ─────────────────────────────────────────────────────────────
// Design tokens(延續既有暗色科技感調性)
// ─────────────────────────────────────────────────────────────

const T = {
  bg:               "transparent",
  focus:            "#00d4aa",
  focus_glow:       "#00d4aa33",
  foundation_leaf:  "#7f77dd",
  foundation_leaf_in_brain: "#c9aaff",
  foundation_mid:   "#3a3a5a",
  downstream:       "#4f8ef7",
  downstream_link:  "#3a5a8a",
  edge_up:          "#534AB7",
  edge_down:        "#2f5a8a",
  text_dim:         "#6a7a9a",
  text_focus:       "#00ffcc",
  text_downstream:  "#a8c8ff",
};

const PAD = { top: 30, bottom: 30, side: 26 };

// ─────────────────────────────────────────────────────────────
// Layout: 合併 foundation(往上)與 downstream(往下)深度
// ─────────────────────────────────────────────────────────────

function buildVGraph(
  focusedEid:       string,
  focusedLabel:     string,
  foundationTree:   FoundationTreeData | null,
  downstreamChains: ChainStep[][],
  personalBrainEids: Set<string>,
): { nodes: VNode[]; edges: VEdge[] } {
  const nodeMap = new Map<string, VNode>();
  const edges: VEdge[] = [];

  nodeMap.set(focusedEid, {
    id: focusedEid, label: focusedLabel, kind: "focus", depth: 0,
  });

  // ── foundation 部分:從 anchor(focusedEid)往上做 BFS 深度指派 ──
  if (foundationTree) {
    const leafEids = new Set(foundationTree.foundations.map((f) => f.element_id));
    const childrenOf = new Map<string, string[]>(); // target(下游) → sources(上游)
    for (const e of foundationTree.edges) {
      if (!childrenOf.has(e.target)) childrenOf.set(e.target, []);
      childrenOf.get(e.target)!.push(e.source);
      edges.push({ source: e.source, target: e.target, kind: "foundation" });
    }

    const depthOf = new Map<string, number>([[focusedEid, 0]]);
    const queue: string[] = [focusedEid];
    while (queue.length) {
      const cur = queue.shift()!;
      const d   = depthOf.get(cur)!;
      for (const parent of childrenOf.get(cur) ?? []) {
        if (!depthOf.has(parent)) {
          depthOf.set(parent, d - 1); // 往上(foundation)= 負深度
          queue.push(parent);
        }
      }
    }

    for (const n of foundationTree.nodes) {
      if (n.element_id === focusedEid) continue;
      const isLeaf = leafEids.has(n.element_id);
      const inBrain = personalBrainEids.has(n.element_id);
      nodeMap.set(n.element_id, {
        id:    n.element_id,
        label: n.label,
        kind:  isLeaf ? "foundation-leaf" : "foundation-mid",
        depth: depthOf.get(n.element_id) ?? -1,
      });
      // 用 inBrain 直接覆寫顏色不方便走 kind,改在 render 階段查表判斷
      void inBrain;
    }
  }

  // ── downstream 部分:每條鏈的第 i 個節點 depth = i ──
  for (const chain of downstreamChains) {
    for (let i = 1; i < chain.length; i++) {
      const step = chain[i];
      if (!nodeMap.has(step.eid)) {
        nodeMap.set(step.eid, {
          id: step.eid, label: step.label, kind: "downstream", depth: i,
        });
      }
      const prev = chain[i - 1];
      edges.push({ source: prev.eid, target: step.eid, kind: "downstream" });
    }
  }

  return { nodes: [...nodeMap.values()], edges };
}

// ─────────────────────────────────────────────────────────────
// RCLineageViz
// ─────────────────────────────────────────────────────────────

export default function RCLineageViz({
  focusedEid,
  focusedLabel,
  foundationTree,
  downstreamChains,
  personalBrainEids,
  loading = false,
  onFocusNode,
  width,
  height,
}: Props) {
  const svgRef     = useRef<SVGSVGElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !focusedEid) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { nodes, edges } = buildVGraph(
      focusedEid, focusedLabel, foundationTree, downstreamChains, personalBrainEids,
    );

    const minDepth = Math.min(0, ...nodes.map((n) => n.depth));
    const maxDepth = Math.max(0, ...nodes.map((n) => n.depth));
    const span     = Math.max(1, maxDepth - minDepth);

    const usableH = height - PAD.top - PAD.bottom;
    const usableW = width  - PAD.side * 2;

    const byDepth = new Map<number, VNode[]>();
    nodes.forEach((n) => {
      if (!byDepth.has(n.depth)) byDepth.set(n.depth, []);
      byDepth.get(n.depth)!.push(n);
    });

    nodes.forEach((n) => {
      n.y = PAD.top + ((n.depth - minDepth) / span) * usableH;
      const siblings = byDepth.get(n.depth)!;
      const idx      = siblings.indexOf(n);
      const total    = siblings.length;
      n.x = PAD.side + usableW * (total === 1 ? 0.5 : idx / (total - 1));
    });

    const nodeById = new Map(nodes.map((n) => [n.id, n]));

    const root = svg
      .attr("width", width)
      .attr("height", height)
      .style("background", T.bg);

    const g = root.append("g");
    root.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 2.5])
        .on("zoom", (ev) => g.attr("transform", ev.transform)),
    );

    const defs = root.append("defs");
    const mkMarker = (id: string, color: string) => {
      defs.append("marker")
        .attr("id", id).attr("viewBox", "0 -4 8 8")
        .attr("refX", 16).attr("refY", 0)
        .attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", color);
    };
    mkMarker("lin-arr-up",   T.edge_up);
    mkMarker("lin-arr-down", T.edge_down);

    // ── edges(貝茲曲線)──
    const edgeG = g.append("g");
    edges.forEach((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      if (!s || !t) return;
      const cy1 = s.y! + (t.y! - s.y!) * 0.4;
      const cy2 = s.y! + (t.y! - s.y!) * 0.7;
      edgeG.append("path")
        .attr("d", `M${s.x},${s.y} C${s.x},${cy1} ${t.x},${cy2} ${t.x},${t.y}`)
        .attr("fill", "none")
        .attr("stroke", e.kind === "foundation" ? T.edge_up : T.edge_down)
        .attr("stroke-width", e.kind === "downstream" ? 2 : 1.2)
        .attr("stroke-dasharray", e.kind === "foundation" ? "4,3" : "none")
        .attr("stroke-opacity", 0.7)
        .attr("marker-end", `url(#${e.kind === "foundation" ? "lin-arr-up" : "lin-arr-down"})`);
    });

    // ── nodes ──
    const nodeG = g.append("g")
      .selectAll<SVGGElement, VNode>("g")
      .data(nodes)
      .join("g")
      .attr("transform", (d) => `translate(${d.x},${d.y})`)
      .style("cursor", (d) => (d.kind === "downstream" ? "pointer" : "default"));

    nodeG.each(function (d) {
      const sel = d3.select(this);
      if (d.kind === "focus") {
        sel.append("circle").attr("r", 8).attr("fill", "none")
          .attr("stroke", T.focus_glow).attr("stroke-width", 10);
        sel.append("circle").attr("r", 8)
          .attr("fill", T.focus).attr("stroke", "#00ffcc").attr("stroke-width", 2);
      } else if (d.kind === "downstream") {
        sel.append("circle").attr("r", 6.5)
          .attr("fill", T.downstream).attr("stroke", "#a8c8ff").attr("stroke-width", 1.5);
      } else if (d.kind === "foundation-leaf") {
        const inBrain = personalBrainEids.has(d.id);
        sel.append("polygon")
          .attr("points", "0,-6 6,0 0,6 -6,0")
          .attr("fill", inBrain ? T.foundation_leaf_in_brain : T.foundation_leaf)
          .attr("stroke", inBrain ? "#e8ddff" : "#9b89e8")
          .attr("stroke-width", inBrain ? 1.8 : 1);
      } else {
        sel.append("circle").attr("r", 3.5)
          .attr("fill", T.foundation_mid).attr("stroke", "#5a5a8a").attr("stroke-width", 1);
      }
    });

    nodeG.append("text")
      .attr("dy", (d) => (d.kind === "focus" ? 22 : d.kind === "foundation-mid" ? 14 : 18))
      .attr("text-anchor", "middle")
      .attr("font-size", (d) => (d.kind === "focus" ? "9px" : "7.5px"))
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("fill", (d) => {
        if (d.kind === "focus") return T.text_focus;
        if (d.kind === "downstream") return T.text_downstream;
        if (d.kind === "foundation-leaf" && personalBrainEids.has(d.id)) return T.foundation_leaf_in_brain;
        return T.text_dim;
      })
      .attr("pointer-events", "none")
      .text((d) => (d.label.length > 16 ? d.label.slice(0, 15) + "…" : d.label));

    nodeG
      .on("click", (_ev, d) => {
        if (d.kind === "downstream") onFocusNode(d.id, d.label);
      })
      .on("mouseenter", (event, d) => {
        if (!tooltipRef.current) return;
        const kindLabel =
          d.kind === "focus" ? "◉ 目前聚焦" :
          d.kind === "downstream" ? "↓ downstream RC(點擊可聚焦)" :
          d.kind === "foundation-leaf" ? (personalBrainEids.has(d.id) ? "◈ foundation(已在 Personal Brain)" : "◈ foundation") :
          "· 中繼概念";
        tooltipRef.current.innerHTML = `<strong>${d.label}</strong><br/><span style="opacity:0.6">${kindLabel}</span>`;
        tooltipRef.current.style.display = "block";
        tooltipRef.current.style.left = `${event.offsetX + 12}px`;
        tooltipRef.current.style.top  = `${event.offsetY - 8}px`;
      })
      .on("mousemove", (event) => {
        if (!tooltipRef.current) return;
        tooltipRef.current.style.left = `${event.offsetX + 12}px`;
        tooltipRef.current.style.top  = `${event.offsetY - 8}px`;
      })
      .on("mouseleave", () => {
        if (tooltipRef.current) tooltipRef.current.style.display = "none";
      });

    // 層標籤
    root.append("text")
      .attr("x", PAD.side).attr("y", 14)
      .attr("font-size", "8px").attr("fill", "rgba(127,119,221,0.4)")
      .attr("font-family", "IBM Plex Mono, monospace").attr("letter-spacing", "0.15em")
      .text("FOUNDATIONS ▲");
    root.append("text")
      .attr("x", PAD.side).attr("y", height - 10)
      .attr("font-size", "8px").attr("fill", "rgba(79,142,247,0.4)")
      .attr("font-family", "IBM Plex Mono, monospace").attr("letter-spacing", "0.15em")
      .text("▼ DOWNSTREAM");

  }, [focusedEid, focusedLabel, foundationTree, downstreamChains, personalBrainEids, width, height, onFocusNode]);

  return (
    <div style={{ position: "relative", width, height }}>
      <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />
      <div
        ref={tooltipRef}
        style={{
          display: "none", position: "absolute", pointerEvents: "none",
          background: "rgba(13,18,32,0.92)", border: "1px solid rgba(127,119,221,0.3)",
          borderRadius: 8, padding: "6px 10px", fontSize: 11, color: "#c8daf0",
          fontFamily: "IBM Plex Mono, monospace", maxWidth: 220, lineHeight: 1.5, zIndex: 10,
        }}
      />
      {loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(3,1,8,0.4)", color: "rgba(127,119,221,0.6)", fontSize: 11,
          fontFamily: "monospace", letterSpacing: "0.1em", pointerEvents: "none",
        }}>
          載入剖面資料…
        </div>
      )}
      {!focusedEid && !loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 6,
          color: "rgba(127,119,221,0.2)", fontSize: 11, letterSpacing: "0.14em",
          textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none",
        }}>
          <span>選擇上方推薦 RC</span>
          <span style={{ fontSize: 10 }}>檢視它的來歷與 downstream</span>
        </div>
      )}
    </div>
  );
}
