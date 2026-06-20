"use client";

/**
 * DetailPanel.tsx（改版）
 *
 * 職責：
 *   1. 顯示被點選 concept 的 foundation tree 拓墣（D3 force graph）
 *   2. 使用者在拓墣上勾選 foundation → 即時三層評估指數預覽
 *   3. 確認加入 personal brain
 *
 * Neo4j Concept 欄位（同步後端）：
 *   definition / key_assumptions / mechanism / name：一般 concept 與 RC 皆有
 *   function：僅 Resulting Concept 才有值（一般 concept 為 null）
 *             mechanism 與 function 是並存欄位，不是替代關係
 *   舊版 description / analysis_types 欄位已移除
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
  function:         string | null;   // 僅 Resulting Concept 才有值
  analysis_types: string[];
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

export type RCSemanticScore = {
  eid:   string;
  label: string;
  score: number;
};

// RC foundation intersection data (pre-computed from all_graphs)

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
  alpha:             number;   // weight for semantic_sim (0–1), default 0.5
  onAddFoundations:  (eids: { eid: string; label: string }[]) => void;
  onRemove:          (eid: string) => void;
  onUpdateWeight:    (eid: string, weight: number) => void;
};

// ─────────────────────────────────────────────────────────────
// Score computation (pure, runs in render)
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
  // Build weight map from personal brain (selected foundations only)
  const weightMap = new Map<string, number>();
  for (const c of personalBrain) {
    if (selectedEids.has(c.eid)) weightMap.set(c.eid, c.weight);
  }
  // Newly toggled (not yet in brain) get weight=0.5
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
// Foundation Tree (D3)
// ─────────────────────────────────────────────────────────────

type FNode = {
  id:       string;
  label:    string;
  isAnchor: boolean;
  isLeaf:   boolean;
  x?: number; y?: number;
  vx?: number; vy?: number;
  fx?: number | null; fy?: number | null;
};
type FEdge = { source: string | FNode; target: string | FNode };

const T = {
  bg:             "transparent",
  anchor:         "#00d4aa",
  foundation:     "#6a5acd",
  intermediate:   "#2d4a7a",
  selected:       "#e8a030",
  text:           "#8888bb",
  text_selected:  "#ffffff",
  edge:           "#1e2d45",
  edge_selected:  "#e8a030",
};

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

    const nodes: FNode[] = data.nodes.map((n) => ({
      id:       n.element_id,
      label:    n.label,
      isAnchor: n.element_id === data.anchor_eid,
      isLeaf:   leafEids.has(n.element_id),
    }));
    const nodeById = new Map(nodes.map((n) => [n.id, n]));

    const edges: FEdge[] = data.edges.map((e) => ({
      source: nodeById.get(e.source) ?? e.source,
      target: nodeById.get(e.target) ?? e.target,
    }));

    // Pin anchor to bottom-centre
    const anchorNode = nodes.find((n) => n.isAnchor);
    if (anchorNode) { anchorNode.fx = width / 2; anchorNode.fy = height - 30; }

    const root = svg.attr("width", width).attr("height", height).style("background", T.bg);
    const zG   = root.append("g");

    root.call(d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (ev) => zG.attr("transform", ev.transform)));

    const defs = root.append("defs");
    ["dim", "selected"].forEach((id) => {
      defs.append("marker").attr("id", `ft-arrow-${id}`)
        .attr("viewBox", "0 -4 8 8").attr("refX", 16).attr("refY", 0)
        .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
        .append("path").attr("d", "M0,-4L8,0L0,4")
        .attr("fill", id === "selected" ? T.edge_selected : T.edge);
    });

    const sim = d3.forceSimulation<FNode>(nodes)
      .force("link", d3.forceLink<FNode, FEdge>(edges).id((d) => d.id).distance(70).strength(0.6))
      .force("charge", d3.forceManyBody<FNode>().strength(-180))
      .force("x", d3.forceX(width / 2).strength(0.06))
      .force("y", d3.forceY(height / 2).strength(0.04))
      .force("collision", d3.forceCollide<FNode>().radius(22));

    const link = zG.append("g").selectAll<SVGLineElement, FEdge>("line")
      .data(edges).join("line")
      .attr("stroke", T.edge).attr("stroke-width", 1.2)
      .attr("opacity", 0.5).attr("marker-end", "url(#ft-arrow-dim)");

    const node = zG.append("g").selectAll<SVGGElement, FNode>("g")
      .data(nodes).join("g")
      .style("cursor", (d) => (d.isLeaf && !d.isAnchor) ? "pointer" : "default")
      .call(d3.drag<SVGGElement, FNode>()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); if (!d.isAnchor) { d.fx = null; d.fy = null; } }));

    node.append("circle")
      .attr("r", (d) => d.isAnchor ? 20 : d.isLeaf ? 14 : 9)
      .attr("fill", (d) => {
        if (d.isAnchor)               return T.anchor;
        if (selectedEids.has(d.id))   return T.selected;
        if (d.isLeaf)                 return T.foundation;
        return T.intermediate;
      })
      .attr("stroke", (d) => selectedEids.has(d.id) ? T.selected : "transparent")
      .attr("stroke-width", 2);

    node.append("text")
      .attr("dy", (d) => (d.isAnchor ? 32 : d.isLeaf ? 24 : 18))
      .attr("text-anchor", "middle")
      .attr("font-size", (d) => d.isAnchor ? "9px" : "8px")
      .attr("fill", (d) => selectedEids.has(d.id) ? T.text_selected : T.text)
      .attr("pointer-events", "none")
      .text((d) => d.label.length > 20 ? d.label.slice(0, 19) + "…" : d.label);

    node.filter((d) => d.isLeaf && !d.isAnchor)
      .on("click", (_ev, d) => onToggle(d.id, d.label));

    sim.on("tick", () => {
      link
        .attr("x1", (e) => (e.source as FNode).x ?? 0)
        .attr("y1", (e) => (e.source as FNode).y ?? 0)
        .attr("x2", (e) => (e.target as FNode).x ?? 0)
        .attr("y2", (e) => (e.target as FNode).y ?? 0)
        .attr("stroke", (e) => {
          const sid = ((e.source as FNode).id);
          const tid = ((e.target as FNode).id);
          return (selectedEids.has(sid) || selectedEids.has(tid)) ? T.edge_selected : T.edge;
        })
        .attr("marker-end", (e) => {
          const sid = ((e.source as FNode).id);
          const tid = ((e.target as FNode).id);
          return (selectedEids.has(sid) || selectedEids.has(tid)) ? "url(#ft-arrow-selected)" : "url(#ft-arrow-dim)";
        });
      node.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { sim.stop(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, width, height, selectedEids]);

  return <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ─────────────────────────────────────────────────────────────
// RankingBar helpers
// ─────────────────────────────────────────────────────────────

function RankBar({ label, score, max, personal, rank }: { label: string; score: number; max: number; personal: boolean; rank: number }) {
  const pct = max > 0 ? (score / max) * 100 : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
      <span style={{ color: "var(--muted)", fontSize: "0.6rem", width: 12, flexShrink: 0 }}>{rank}.</span>
      <span style={{ color: personal ? "var(--teal)" : "#9090b0", fontSize: "0.68rem", width: 90, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }} title={label}>{label}</span>
      <div style={{ flex: 1, height: 6, background: "rgba(83,74,183,0.15)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: 3, background: personal ? "linear-gradient(90deg,#534AB7,#00d4aa)" : "#534AB7", transition: "width 0.35s ease" }} />
      </div>
      <span style={{ color: "#afa9ec", fontSize: "0.62rem", width: 32, textAlign: "right", flexShrink: 0 }}>{score.toFixed(2)}</span>
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
  // Which foundation nodes the user has toggled ON in this session
  const [selectedEids, setSelectedEids] = useState<Set<string>>(new Set());
  const [showWeightFor, setShowWeightFor] = useState<string | null>(null);

  // Reset selection when node changes
  useEffect(() => { setSelectedEids(new Set()); setShowWeightFor(null); }, [node?.id]);

  // ── Foundation toggle ─────────────────────────────────────
  const handleToggle = useCallback((eid: string, _label: string) => {
    setSelectedEids((prev) => {
      const next = new Set(prev);
      if (next.has(eid)) next.delete(eid); else next.add(eid);
      return next;
    });
  }, []);

  // ── Merge selected with existing personal brain for preview ─
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

  // ── Ranking: before (personalBrain only) vs after (previewBrain) ──
  const rankingBefore = useMemo(
    () => computeRanking(new Set(personalBrain.map((c) => c.eid)), personalBrain, rcSemanticScores, rcFoundationMap, alpha).slice(0, 3),
    [personalBrain, rcSemanticScores, rcFoundationMap, alpha],
  );
  const rankingAfter = useMemo(
    () => computeRanking(new Set(previewBrain.map((c) => c.eid)), previewBrain, rcSemanticScores, rcFoundationMap, alpha).slice(0, 3),
    [previewBrain, rcSemanticScores, rcFoundationMap, alpha],
  );

  const hasSelection = selectedEids.size > 0;
  const maxBefore    = rankingBefore[0]?.totalScore ?? 1;
  const maxAfter     = rankingAfter[0]?.totalScore  ?? 1;

  // Positions before → after
  const beforePositions = useMemo(
    () => new Map(rankingBefore.map((r, i) => [r.eid, i + 1])),
    [rankingBefore],
  );

  // ── Add to brain ──────────────────────────────────────────
  const handleConfirmAdd = useCallback(() => {
    if (!foundationTree) return;
    const toAdd = [...selectedEids].map((eid) => {
      const fn = foundationTree.nodes.find((n) => n.element_id === eid);
      return { eid, label: fn?.label ?? eid };
    });
    onAddFoundations(toAdd);
    setSelectedEids(new Set());
  }, [selectedEids, foundationTree, onAddFoundations]);

  // ── Viz size ──────────────────────────────────────────────
  const treeWidth  = 260;
  const treeHeight = 180;

  // ── Empty state ───────────────────────────────────────────
  if (!node) {
    return (
      <div className="detail-panel">
        <div className="detail-label" style={{ flexShrink: 0 }}>Detail Panel</div>
        <div className="detail-empty">
          <span>點擊 Global Brain</span>
          <span>探索 concept 的 foundations</span>
        </div>
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
          <span style={{ fontSize: "0.6rem", color: "var(--teal)", flexShrink: 0, alignSelf: "flex-start", marginTop: 14 }}>◉</span>
        )}
      </div>

      {/* ── Foundation Tree ── */}
      <div style={{ flexShrink: 0 }}>
        <div className="detail-section-label" style={{ marginBottom: 4 }}>
          Foundation Tree
          <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
            · 點選 foundation 節點（◆）加入
          </span>
        </div>

        {loadingTree ? (
          <div className="detail-loading-bar" style={{ marginBottom: 4 }} />
        ) : foundationTree && foundationTree.nodes.length > 0 ? (
          <div style={{ border: "1px solid rgba(30,46,72,0.6)", borderRadius: 10, overflow: "hidden", background: "rgba(6,8,15,0.5)" }}>
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
              const fn = foundationTree?.nodes.find((n) => n.element_id === eid);
              const alreadyIn = personalBrain.some((c) => c.eid === eid);
              return (
                <span key={eid} style={{
                  fontSize: "0.62rem", padding: "2px 7px", borderRadius: 12,
                  border: `1px solid ${alreadyIn ? "var(--teal)" : "var(--orange)"}`,
                  background: alreadyIn ? "rgba(0,212,170,0.08)" : "rgba(232,160,48,0.1)",
                  color: alreadyIn ? "var(--teal)" : "var(--orange)",
                  cursor: "pointer",
                }}
                  onClick={() => handleToggle(eid, fn?.label ?? eid)}
                >
                  {alreadyIn ? "✓ " : "◉ "}{fn?.label ?? eid}
                </span>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Layer 1: RC Ranking preview ── */}
      {rcSemanticScores.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 6 }}>
            {hasSelection ? "RC 排名預覽（加入後）" : "RC 排名"}
          </div>

          {hasSelection ? (
            /* Before / After comparison */
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {/* Before */}
              <div>
                <div style={{ fontSize: "0.58rem", color: "var(--muted)", marginBottom: 4, opacity: 0.6, letterSpacing: "0.1em" }}>BEFORE</div>
                {rankingBefore.map((r, i) => (
                  <RankBar key={r.eid} label={r.label} score={r.totalScore} max={maxBefore} personal={false} rank={i + 1} />
                ))}
              </div>
              {/* After */}
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
            /* No selection: just show current top-3 */
            <div>
              {rankingBefore.map((r, i) => (
                <RankBar key={r.eid} label={r.label} score={r.totalScore} max={maxBefore} personal={false} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Layer 2: Intersection breakdown ── */}
      {hasSelection && rankingAfter.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>交集說明</div>
          {rankingAfter.slice(0, 2).map((r) => (
            <div key={r.eid} style={{ display: "flex", gap: 6, alignItems: "baseline", marginBottom: 3, fontSize: "0.66rem" }}>
              <span style={{ color: "var(--purple)", opacity: 0.7, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 90 }} title={r.label}>{r.label}</span>
              <span style={{ color: "var(--muted)" }}>·</span>
              <span style={{ color: "#afa9ec" }}>
                {r.hits}/{r.totalFound} foundations matched
              </span>
              <span style={{ color: "var(--muted)", marginLeft: "auto" }}>
                inter={r.interScore.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ── Layer 3: Knowledge vector hint ── */}
      {personalBrain.length > 0 && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>知識重心</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {personalBrain.slice(0, 6).map((c) => (
              <span key={c.eid} style={{
                fontSize: "0.6rem", padding: "2px 6px", borderRadius: 10,
                border: "1px solid rgba(127,119,221,0.3)",
                background: `rgba(83,74,183,${(c.weight * 0.25).toFixed(2)})`,
                color: `rgba(175,169,236,${(0.5 + c.weight * 0.5).toFixed(2)})`,
                cursor: "pointer",
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
                <input type="range" min="0.1" max="1.0" step="0.05" value={c.weight}
                  onChange={(e) => onUpdateWeight(c.eid, parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: "var(--purple)", height: 3 }}
                />
                <span style={{ color: "#afa9ec", width: 28, textAlign: "right" }}>{c.weight.toFixed(2)}</span>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── Theory summary (definition + mechanism + key_assumptions) ── */}
      {detail?.definition && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>Definition</div>
          <p style={{ margin: 0, fontSize: "0.68rem", color: "#8880c0", lineHeight: 1.55 }}>
            {detail.definition.slice(0, 160)}{detail.definition.length > 160 ? "…" : ""}
          </p>
        </div>
      )}

      {detail?.mechanism && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>Mechanism</div>
          <p style={{ margin: 0, fontSize: "0.68rem", color: "#8880c0", lineHeight: 1.55 }}>
            {detail.mechanism.slice(0, 160)}{detail.mechanism.length > 160 ? "…" : ""}
          </p>
        </div>
      )}

      {detail?.key_assumptions && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>Key Assumptions</div>
          <p style={{ margin: 0, fontSize: "0.68rem", color: "#8880c0", lineHeight: 1.55 }}>
            {detail.key_assumptions.slice(0, 160)}{detail.key_assumptions.length > 160 ? "…" : ""}
          </p>
        </div>
      )}

      {detail?.function && (
        <div style={{ flexShrink: 0 }}>
          <div className="detail-section-label" style={{ marginBottom: 4 }}>
            Function
            <span style={{ color: "var(--teal)", marginLeft: 6, fontWeight: 400, fontSize: "0.56rem" }}>
              · Resulting Concept
            </span>
          </div>
          <p style={{ margin: 0, fontSize: "0.68rem", color: "#8880c0", lineHeight: 1.55 }}>
            {detail.function.slice(0, 160)}{detail.function.length > 160 ? "…" : ""}
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
          <button
            className="detail-btn-secondary detail-btn-remove"
            onClick={() => onRemove(node.id)}
          >
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