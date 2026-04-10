"use client";

import { useEffect, useRef, useCallback } from "react";
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
};

export type GraphData = {
  query_concepts: { id: string; name: string; type: string; score: number }[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: GraphPath[];
  selected_path_id?: number | null;
};

type Props = {
  data: GraphData;
  selectedPathId: number | null;
  onPathSelect: (pathId: number, finalLabel: string) => void;
  width?: number;
  height?: number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens — matches globals.css dark theme
// ─────────────────────────────────────────────────────────────

const TOKEN = {
  bg:              "transparent",
  concept_rc:      "#4f8ef7",   // resulting concept — bright blue
  concept_other:   "#2d4a7a",   // other concept — muted blue
  concept_matched: "#00d4aa",   // query-matched concept — teal
  paper:           "#1e2a3a",   // paper node
  paper_border:    "#3a5070",
  edge_dim:        "#1e2d45",
  edge_active:     "#4f8ef7",
  edge_selected:   "#00d4aa",
  text_dim:        "#4a6080",
  text_node:       "#c8daf0",
  text_selected:   "#ffffff",
  highlight_glow:  "#00d4aa",
  path_colors: [
    "#4f8ef7", "#00d4aa", "#e8a030", "#e85d60",
    "#a060f0", "#3dd090", "#f07050", "#60b0f0",
  ],
};

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
  const svgRef       = useRef<SVGSVGElement | null>(null);
  const simRef       = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);
  const tooltipRef   = useRef<HTMLDivElement | null>(null);

  // ── build / rebuild D3 scene whenever data changes ──────────
  useEffect(() => {
    if (!svgRef.current || !data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    if (!data.nodes.length) return;

    // deep-clone so D3 can mutate x/y without contaminating React state
    const nodes: GraphNode[] = data.nodes.map((n) => ({ ...n }));
    const nodeById = new Map(nodes.map((n) => [n.id, n]));

    const edges: GraphEdge[] = data.edges.map((e) => ({
      ...e,
      source: nodeById.get(e.source as string) ?? e.source,
      target: nodeById.get(e.target as string) ?? e.target,
    }));

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

    // ── arrow marker ──────────────────────────────────────────
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

    // ── simulation ────────────────────────────────────────────
    const sim = d3
      .forceSimulation<GraphNode>(nodes)
      .force("link",
        d3.forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((e) => {
            const s = e.source as GraphNode;
            const t = e.target as GraphNode;
            // paper→concept edges shorter; concept→concept longer
            if (s.node_type === "paper" || t.node_type === "paper") return 70;
            return 120;
          })
          .strength(0.6)
      )
      .force("charge",
        d3.forceManyBody()
          .strength((d) => (d.node_type === "paper" ? -80 : -200))
      )
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision",
        d3.forceCollide<GraphNode>()
          .radius((d) => (d.node_type === "paper" ? 18 : 26))
      );

    simRef.current = sim;

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

    // labels
    node
      .append("text")
      .attr("dy", (d) => (d.node_type === "concept" ? 32 : 22))
      .attr("text-anchor", "middle")
      .attr("font-size",   (d) => (d.node_type === "paper" ? "8px" : "9px"))
      .attr("fill", TOKEN.text_node)
      .attr("pointer-events", "none")
      .text((d) => truncate(d.label, d.node_type === "paper" ? 20 : 22));

    // short label inside node
    node
      .append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", "7px")
      .attr("fill", TOKEN.text_dim)
      .attr("pointer-events", "none")
      .text((d) =>
        d.node_type === "paper" ? truncate(d.label, 10) : ""
      );

    // ── path interaction ──────────────────────────────────────
    // build lookup maps
    const pathByNodeId  = new Map<string, number[]>();
    const pathByEdgeId  = new Map<string, number[]>();
    data.paths.forEach((p) => {
      p.node_ids.forEach((nid) => {
        if (!pathByNodeId.has(nid)) pathByNodeId.set(nid, []);
        pathByNodeId.get(nid)!.push(p.path_id);
      });
      p.edge_ids.forEach((eid) => {
        if (!pathByEdgeId.has(eid)) pathByEdgeId.set(eid, []);
        pathByEdgeId.get(eid)!.push(p.path_id);
      });
    });

    function applyHighlight(activePathId: number | null) {
      const activePath = activePathId !== null
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
          const eid = (e as GraphEdge).id;
          if (activeEdgeSet.has(eid)) return pathColor;
          return TOKEN.edge_dim;
        })
        .attr("stroke-width", (e) => (activeEdgeSet.has((e as GraphEdge).id) ? 2.5 : 1))
        .attr("opacity",      (e) => {
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

      node.select("text")
        .attr("fill", (d) => {
          if (activePathId === null) return TOKEN.text_node;
          return activeNodeSet.has(d.id) ? TOKEN.text_selected : TOKEN.text_dim;
        });
    }

    // initial highlight if server already resolved a path
    applyHighlight(data.selected_path_id ?? null);

    // node click → pick a path
    node.on("click", (_event, d) => {
      const paths = pathByNodeId.get(d.id) ?? [];
      if (paths.length === 0) return;
      // cycle through paths that share this node
      const nextId = paths[(paths.indexOf(selectedPathId ?? -1) + 1) % paths.length];
      const p      = data.paths.find((x) => x.path_id === nextId);
      if (p) {
        applyHighlight(nextId);
        onPathSelect(nextId, p.final_label);
      }
    });

    // hover tooltip
    node.on("mouseenter", (event, d) => {
      if (!tooltipRef.current) return;
      const tip = tooltipRef.current;
      let html   = `<strong>${d.label}</strong>`;
      if (d.node_type === "paper" && d.meta) {
        if (d.meta.conceptual_premises)
          html += `<br/><em>${d.meta.conceptual_premises.slice(0, 120)}…</em>`;
      }
      if (d.matched) html += `<br/><span style="color:#00d4aa">◉ 查詢命中</span>`;
      tip.innerHTML   = html;
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
    });

    return () => {
      sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, width, height]);

  // re-apply highlight when selectedPathId changes without full rebuild
  useEffect(() => {
    if (!svgRef.current || !data) return;
    const svg = d3.select(svgRef.current);

    const activePathId = selectedPathId;
    const activePath   = activePathId !== null
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
      .select("text")
      .attr("fill", (d) => {
        if (activePathId === null) return TOKEN.text_node;
        return activeNodeSet.has(d.id) ? TOKEN.text_selected : TOKEN.text_dim;
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
