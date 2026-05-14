"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type GraphNode = {
  id: string;
  label: string;
  node_type: "concept" | "paper";
  concept_type: string | null;
  matched: boolean;
  meta?: {
    conceptual_premises?: string | null;
    contextual_model_constraints?: string | null;
  } | null;
  // D3 simulation fields (mutable)
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

export type GraphEdge = {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  paper_title: string;
};

export type GraphPath = {
  path_id: number;
  node_ids: string[];
  edge_ids: string[];
  final_concept: string;
  final_label: string;
  direction?: "upstream" | "downstream";
};

export type Foundation = {
  element_id: string;
  label: string;
};

export type GraphData = {
  query_concepts: { id: string; name: string; type: string; score: number }[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: GraphPath[];
  selected_path_id?: number | null;
  foundations?: Foundation[];
};

type Props = {
  data: GraphData;
  selectedPathId: number | null;
  onPathSelect: (pathId: number, finalLabel: string) => void;
  width?: number;
  height?: number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────

const TOKEN = {
  bg:              "transparent",
  concept_rc:      "#4f8ef7",
  concept_other:   "#2d4a7a",
  concept_matched: "#00d4aa",
  paper:           "#1e2a3a",
  paper_border:    "#3a5070",
  edge_dim:        "#1e2d45",
  edge_active:     "#4f8ef7",
  edge_selected:   "#00d4aa",
  text_dim:        "#4a6080",
  text_node:       "#c8daf0",
  text_selected:   "#ffffff",
  highlight_glow:  "#00d4aa",
  // Foundation-specific
  foundation_fill:   "#1a1a2e",
  foundation_border: "#6a5acd",
  foundation_text:   "#9b89e8",
  foundation_dash:   "#4a3a8a",
  path_colors: [
    "#4f8ef7", "#00d4aa", "#e8a030", "#e85d60",
    "#a060f0", "#3dd090", "#f07050", "#60b0f0",
  ],
};

// Foundation node layout constants
const F_R  = 14;   // diamond half-size
const F_GAP = 60;  // horizontal spacing between foundation nodes
// Vertical offset above candidate RC (in simulation coords, added after layout)
const F_Y_OFFSET = 90;

// ─────────────────────────────────────────────────────────────
// KnowledgeGraph
// ─────────────────────────────────────────────────────────────

export default function KnowledgeGraph({
  data,
  selectedPathId,
  onPathSelect,
  width  = 600,
  height = 480,
}: Props) {
  const svgRef     = useRef<SVGSVGElement | null>(null);
  const simRef     = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  // ── build / rebuild D3 scene whenever data changes ──────────
  useEffect(() => {
    if (!svgRef.current || !data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    if (!data.nodes.length) return;

    const nodes: GraphNode[] = data.nodes.map((n) => ({ ...n }));
    const nodeById = new Map(nodes.map((n) => [n.id, n]));

    const edges: GraphEdge[] = data.edges.map((e) => ({
      ...e,
      source: nodeById.get(e.source as string) ?? e.source,
      target: nodeById.get(e.target as string) ?? e.target,
    }));

    const foundations: Foundation[] = data.foundations ?? [];

    // ── candidate RC node（matched=true）──────────────────────
    const candidateNode = nodes.find((n) => n.matched) ?? null;

    // ── zoom layer ────────────────────────────────────────────
    const root = svg
      .attr("width",  width)
      .attr("height", height)
      .style("background", TOKEN.bg);

    const zoomG = root.append("g").attr("class", "zoom-root");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => {
        zoomG.attr("transform", event.transform);
      });

    root.call(zoom);

    // ── defs: markers ─────────────────────────────────────────
    const defs = root.append("defs");

    const mkMarker = (id: string, color: string) => {
      defs
        .append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -4 8 8")
        .attr("refX", 18)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L8,0L0,4")
        .attr("fill", color);
    };
    mkMarker("arrow-dim",      TOKEN.edge_dim);
    mkMarker("arrow-active",   TOKEN.edge_active);
    mkMarker("arrow-selected", TOKEN.edge_selected);
    TOKEN.path_colors.forEach((c, i) => mkMarker(`arrow-path-${i}`, c));

    // foundation dashed arrow marker（小一點，指向 candidate）
    defs
      .append("marker")
      .attr("id", "arrow-foundation")
      .attr("viewBox", "0 -3 6 6")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-3L6,0L0,3")
      .attr("fill", TOKEN.foundation_dash);

    // ── simulation ────────────────────────────────────────────
    const sim = d3
      .forceSimulation<GraphNode>(nodes)
      .force("link",
        d3.forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((e) => {
            const s = e.source as GraphNode;
            const t = e.target as GraphNode;
            if (s.node_type === "paper" || t.node_type === "paper") return 70;
            return 120;
          })
          .strength(0.6)
      )
      .force("charge",
        d3.forceManyBody<GraphNode>()
          .strength((d) => (d.node_type === "paper" ? -80 : -200))
      )
      // candidate RC 的起始位置偏下，為 foundation 留空間
      .force("center", d3.forceCenter(width / 2, height / 2 + F_Y_OFFSET / 2))
      .force("collision",
        d3.forceCollide<GraphNode>()
          .radius((d) => (d.node_type === "paper" ? 18 : 26))
      );

    simRef.current = sim;

    // ── foundation layer（固定位置，不加入 simulation）─────────
    const foundationG = zoomG.append("g").attr("class", "foundations");

    // Foundation 節點位置：以 candidate RC 初始中心為基準，
    // 排在上方；simulation tick 後再更新 dashed lines（見下方 tick）。
    // 初始時先用 SVG 中心估算。
    const fCount = foundations.length;

    // 計算每個 foundation 節點的靜態 x（以 width/2 為中心扇開）
    const fPositions: { x: number; y: number }[] = foundations.map((_, i) => ({
      x: width / 2 + (i - (fCount - 1) / 2) * F_GAP,
      y: height / 2 - F_Y_OFFSET,           // 初始估算，tick 後動態更新
    }));

    // dashed lines（foundation → candidate RC）
    const dashLines = foundationG
      .selectAll<SVGLineElement, Foundation>("line.dash")
      .data(foundations)
      .join("line")
      .attr("class", "dash")
      .attr("stroke",           TOKEN.foundation_dash)
      .attr("stroke-width",     1.2)
      .attr("stroke-dasharray", "5,4")
      .attr("opacity",          0.7)
      .attr("marker-end",       "url(#arrow-foundation)")
      .attr("x1", (_, i) => fPositions[i].x)
      .attr("y1", (_, i) => fPositions[i].y + F_R)
      .attr("x2", width / 2)
      .attr("y2", height / 2);

    // foundation 節點群組（diamond shape）
    const fNodes = foundationG
      .selectAll<SVGGElement, Foundation>("g.fnode")
      .data(foundations)
      .join("g")
      .attr("class", "fnode")
      .attr("transform", (_, i) =>
        `translate(${fPositions[i].x},${fPositions[i].y})`
      );

    // diamond polygon: 上、右、下、左
    fNodes
      .append("polygon")
      .attr("points", `0,${-F_R} ${F_R},0 0,${F_R} ${-F_R},0`)
      .attr("fill",         TOKEN.foundation_fill)
      .attr("stroke",       TOKEN.foundation_border)
      .attr("stroke-width", 1.5);

    fNodes
      .append("text")
      .attr("dy", F_R + 12)
      .attr("text-anchor", "middle")
      .attr("font-size",   "8px")
      .attr("fill",        TOKEN.foundation_text)
      .attr("pointer-events", "none")
      .text((d) => truncate(d.label, 18));

    // tooltip on foundation hover
    fNodes
      .style("cursor", "default")
      .on("mouseenter", (event, d) => {
        if (!tooltipRef.current) return;
        tooltipRef.current.innerHTML =
          `<strong>${d.label}</strong><br/><span style="color:${TOKEN.foundation_text}">◈ 基礎概念</span>`;
        tooltipRef.current.style.display = "block";
        tooltipRef.current.style.left    = `${event.offsetX + 12}px`;
        tooltipRef.current.style.top     = `${event.offsetY - 8}px`;
      })
      .on("mousemove", (event) => {
        if (!tooltipRef.current) return;
        tooltipRef.current.style.left = `${event.offsetX + 12}px`;
        tooltipRef.current.style.top  = `${event.offsetY - 8}px`;
      })
      .on("mouseleave", () => {
        if (tooltipRef.current) tooltipRef.current.style.display = "none";
      });

    // ── edge layer ────────────────────────────────────────────
    const linkG = zoomG.append("g").attr("class", "links");
    const link  = linkG
      .selectAll<SVGLineElement, GraphEdge>("line")
      .data(edges)
      .join("line")
      .attr("class", "edge")
      .attr("stroke",       TOKEN.edge_dim)
      .attr("stroke-width", 1)
      .attr("opacity",      0.5)
      .attr("marker-end",   "url(#arrow-dim)");

    // ── node layer ────────────────────────────────────────────
    const nodeG = zoomG.append("g").attr("class", "nodes");
    const node  = nodeG
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes)
      .join("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x; d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    // concept nodes — circles
    node
      .filter((d) => d.node_type === "concept")
      .append("circle")
      .attr("r", (d) => (d.matched ? 22 : 16))
      .attr("fill", (d) => {
        if (d.matched) return TOKEN.concept_matched;
        if ((d.concept_type || "").toLowerCase().includes("resulting"))
          return TOKEN.concept_rc;
        return TOKEN.concept_other;
      })
      .attr("stroke", (d) => (d.matched ? TOKEN.highlight_glow : "#1a2840"))
      .attr("stroke-width", (d) => (d.matched ? 2.5 : 1));

    // paper nodes — rounded rects
    node
      .filter((d) => d.node_type === "paper")
      .append("rect")
      .attr("x", -24).attr("y", -10)
      .attr("width", 48).attr("height", 20)
      .attr("rx", 5)
      .attr("fill",   TOKEN.paper)
      .attr("stroke", TOKEN.paper_border)
      .attr("stroke-width", 1);

    // labels below node
    node
      .append("text")
      .attr("dy", (d) => (d.node_type === "concept" ? 32 : 22))
      .attr("text-anchor", "middle")
      .attr("font-size",   (d) => (d.node_type === "paper" ? "8px" : "9px"))
      .attr("fill", TOKEN.text_node)
      .attr("pointer-events", "none")
      .text((d) => truncate(d.label, d.node_type === "paper" ? 20 : 22));

    // short label inside paper node
    node
      .append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", "7px")
      .attr("fill", TOKEN.text_dim)
      .attr("pointer-events", "none")
      .text((d) => (d.node_type === "paper" ? truncate(d.label, 10) : ""));

    // ── path interaction ──────────────────────────────────────
    const pathByNodeId = new Map<string, number[]>();
    data.paths.forEach((p) => {
      p.node_ids.forEach((nid) => {
        if (!pathByNodeId.has(nid)) pathByNodeId.set(nid, []);
        pathByNodeId.get(nid)!.push(p.path_id);
      });
    });

    function applyHighlight(activePathId: number | null) {
      const activePath    = activePathId !== null
        ? data.paths.find((p) => p.path_id === activePathId)
        : null;
      const activeNodeSet = new Set(activePath?.node_ids ?? []);
      const activeEdgeSet = new Set(activePath?.edge_ids ?? []);
      const colorIdx      = activePathId !== null
        ? activePathId % TOKEN.path_colors.length
        : 0;
      const pathColor = TOKEN.path_colors[colorIdx];

      link
        .attr("stroke", (e) => {
          if (activeEdgeSet.has((e as GraphEdge).id)) return pathColor;
          return TOKEN.edge_dim;
        })
        .attr("stroke-width", (e) => (activeEdgeSet.has((e as GraphEdge).id) ? 2.5 : 1))
        .attr("opacity", (e) => {
          if (activePathId === null) return 0.5;
          return activeEdgeSet.has((e as GraphEdge).id) ? 1 : 0.15;
        })
        .attr("marker-end", (e) => {
          if (activeEdgeSet.has((e as GraphEdge).id))
            return `url(#arrow-path-${colorIdx})`;
          return "url(#arrow-dim)";
        });

      node.select("circle")
        .attr("stroke", (d) => {
          if (activeNodeSet.has(d.id)) return pathColor;
          if (d.matched) return TOKEN.concept_matched;
          return "#1a2840";
        })
        .attr("stroke-width", (d) => (activeNodeSet.has(d.id) ? 3 : d.matched ? 2.5 : 1))
        .attr("opacity", (d) => {
          if (activePathId === null) return 1;
          return activeNodeSet.has(d.id) ? 1 : 0.25;
        });

      node.select("rect")
        .attr("stroke", (d) => (activeNodeSet.has(d.id) ? pathColor : TOKEN.paper_border))
        .attr("opacity", (d) => {
          if (activePathId === null) return 1;
          return activeNodeSet.has(d.id) ? 1 : 0.2;
        });

      node.selectAll("text")
        .attr("fill", (d) => {
          const gn = d as GraphNode;
          if (activePathId === null) return TOKEN.text_node;
          return activeNodeSet.has(gn.id) ? TOKEN.text_selected : TOKEN.text_dim;
        });
    }

    applyHighlight(data.selected_path_id ?? null);

    node.on("click", (_event, d) => {
      const paths = pathByNodeId.get(d.id) ?? [];
      if (paths.length === 0) return;
      const nextId = paths[(paths.indexOf(selectedPathId ?? -1) + 1) % paths.length];
      const p      = data.paths.find((x) => x.path_id === nextId);
      if (p) {
        applyHighlight(nextId);
        onPathSelect(nextId, p.final_label);
      }
    });

    node.on("mouseenter", (event, d) => {
      if (!tooltipRef.current) return;
      const tip = tooltipRef.current;
      let html   = `<strong>${d.label}</strong>`;
      if (d.node_type === "paper" && d.meta) {
        if (d.meta.conceptual_premises)
          html += `<br/><em>${d.meta.conceptual_premises.slice(0, 120)}…</em>`;
      }
      if (d.matched) html += `<br/><span style="color:#00d4aa">◉ 查詢命中</span>`;
      tip.innerHTML     = html;
      tip.style.display = "block";
      tip.style.left    = `${event.offsetX + 12}px`;
      tip.style.top     = `${event.offsetY - 8}px`;
    });
    node.on("mousemove", (event) => {
      if (!tooltipRef.current) return;
      tooltipRef.current.style.left = `${event.offsetX + 12}px`;
      tooltipRef.current.style.top  = `${event.offsetY - 8}px`;
    });
    node.on("mouseleave", () => {
      if (tooltipRef.current) tooltipRef.current.style.display = "none";
    });

    // ── tick ──────────────────────────────────────────────────
    sim.on("tick", () => {
      link
        .attr("x1", (e) => (e.source as GraphNode).x ?? 0)
        .attr("y1", (e) => (e.source as GraphNode).y ?? 0)
        .attr("x2", (e) => (e.target as GraphNode).x ?? 0)
        .attr("y2", (e) => (e.target as GraphNode).y ?? 0);

      node.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);

      // update foundation dashed lines to track candidate RC position
      if (candidateNode && fCount > 0) {
        const cx = candidateNode.x ?? width / 2;
        const cy = candidateNode.y ?? height / 2;

        // update foundation node y-position to stay above candidate RC
        fNodes.attr("transform", (_, i) => {
          const fx = cx + (i - (fCount - 1) / 2) * F_GAP;
          const fy = cy - F_Y_OFFSET;
          fPositions[i].x = fx;
          fPositions[i].y = fy;
          return `translate(${fx},${fy})`;
        });

        dashLines
          .attr("x1", (_, i) => fPositions[i].x)
          .attr("y1", (_, i) => fPositions[i].y + F_R)
          .attr("x2", cx)
          .attr("y2", cy - 22);  // stop at edge of candidate circle (r=22)
      }
    });

    return () => {
      sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, width, height]);

  // ── re-apply highlight when selectedPathId changes ──────────
  useEffect(() => {
    if (!svgRef.current || !data) return;
    const svg = d3.select(svgRef.current);

    const activePathId  = selectedPathId;
    const activePath    = activePathId !== null
      ? data.paths.find((p) => p.path_id === activePathId)
      : null;
    const activeNodeSet = new Set(activePath?.node_ids ?? []);
    const activeEdgeSet = new Set(activePath?.edge_ids ?? []);
    const colorIdx      = activePathId !== null
      ? activePathId % TOKEN.path_colors.length
      : 0;
    const pathColor = TOKEN.path_colors[colorIdx];

    svg.selectAll<SVGLineElement, GraphEdge>(".edge")
      .attr("stroke", (e) => activeEdgeSet.has(e.id) ? pathColor : TOKEN.edge_dim)
      .attr("stroke-width", (e) => activeEdgeSet.has(e.id) ? 2.5 : 1)
      .attr("opacity", (e) => {
        if (activePathId === null) return 0.5;
        return activeEdgeSet.has(e.id) ? 1 : 0.15;
      })
      .attr("marker-end", (e) =>
        activeEdgeSet.has(e.id)
          ? `url(#arrow-path-${colorIdx})`
          : "url(#arrow-dim)"
      );

    svg.selectAll<SVGGElement, GraphNode>(".node")
      .select("circle")
      .attr("stroke",       (d) => activeNodeSet.has(d.id) ? pathColor : (d.matched ? TOKEN.concept_matched : "#1a2840"))
      .attr("stroke-width", (d) => activeNodeSet.has(d.id) ? 3 : d.matched ? 2.5 : 1)
      .attr("opacity",      (d) => {
        if (activePathId === null) return 1;
        return activeNodeSet.has(d.id) ? 1 : 0.25;
      });

    svg.selectAll<SVGGElement, GraphNode>(".node")
      .select("rect")
      .attr("stroke",  (d) => activeNodeSet.has(d.id) ? pathColor : TOKEN.paper_border)
      .attr("opacity", (d) => {
        if (activePathId === null) return 1;
        return activeNodeSet.has(d.id) ? 1 : 0.2;
      });

    svg.selectAll<SVGGElement, GraphNode>(".node")
      .selectAll("text")
      .attr("fill", (d) => {
        const gn = d as GraphNode;
        if (activePathId === null) return TOKEN.text_node;
        return activeNodeSet.has(gn.id) ? TOKEN.text_selected : TOKEN.text_dim;
      });
  }, [selectedPathId, data]);

  return (
    <div style={{ position: "relative", width, height }}>
      <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />
      <div
        ref={tooltipRef}
        style={{
          display:       "none",
          position:      "absolute",
          pointerEvents: "none",
          background:    "#0d1626ee",
          border:        "1px solid #2a3f60",
          borderRadius:  "8px",
          padding:       "8px 12px",
          fontSize:      "11px",
          color:         "#c8daf0",
          maxWidth:      "220px",
          lineHeight:    "1.5",
          zIndex:        10,
        }}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Util
// ─────────────────────────────────────────────────────────────

function truncate(s: string, maxLen: number): string {
  if (!s) return "";
  return s.length > maxLen ? s.slice(0, maxLen - 1) + "…" : s;
}
