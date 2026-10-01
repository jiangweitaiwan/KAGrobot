"use client";

/**
 * BriefingGraph — RC 推導拓樸圖（純 SVG，無外部套件）
 *
 * 輸入是後端 briefing 回傳的 graph（剪枝後、LLM 實際看到的那張圖）。
 * 排版：
 *   - 依 depth 分欄，最上游在最左、RC（depth 0）在最右，箭頭整體朝右。
 *   - 欄內用 barycenter 排序（取相連節點在鄰欄的平均位置）減少連線交叉。
 *   - 跨欄的邊畫三次貝茲曲線；同欄/反向的邊（只會在後端偵測到環時出現）繞到右側。
 * 互動：
 *   - hover / focus 節點 → 只亮該節點與它的整條上游鏈。
 *   - 點上方的「層」標籤 → 亮該層的節點。
 * 舊資料沒有 graph 時，由呼叫端不渲染本元件即可。
 */

import { useId, useMemo, useState } from "react";

export type BriefingGraphNode = {
  eid: string;
  label: string;
  node_type: string;   // support | critique | other
  depth: number;       // 0 = RC，數字越大越上游
};
export type BriefingGraphEdge = {
  source: string;      // 上游
  target: string;      // 下游（朝 RC）
  edge_type: string;
};
export type BriefingGraphData = {
  nodes: BriefingGraphNode[];
  edges: BriefingGraphEdge[];
};
export type BriefingGraphLayer = { name: string; node_eids: string[] };

type Props = {
  graph: BriefingGraphData;
  rcEid: string;
  layers?: BriefingGraphLayer[];
  walkOverlap?: string[];
};

const NODE_W = 128;
const NODE_H = 28;
const COL_GAP = 52;
const ROW_GAP = 10;
const PAD = 12;
const MAX_LABEL_UNITS = 10;   // CJK 字 = 1，其他字元 = 0.55
const MAX_COLS = 13;

const C_RC = "var(--orange, #e8a030)";
const C_SUPPORT = "var(--purple, #7f77dd)";
const C_CRITIQUE = "var(--red, #e5645a)";
const C_WALK = "var(--teal, #00d4aa)";
const C_TEXT = "var(--text, #e8e6f0)";
const C_MUTED = "var(--muted, #8a86a0)";

function shortLabel(s: string): string {
  let units = 0;
  let out = "";
  for (const ch of s) {
    units += ch.charCodeAt(0) > 0x2e80 ? 1 : 0.55;
    if (units > MAX_LABEL_UNITS) return out + "…";
    out += ch;
  }
  return out;
}

type Placed = BriefingGraphNode & { x: number; y: number; col: number };

function computeLayout(graph: BriefingGraphData) {
  const nodes = graph.nodes;
  const depthOf = (n: BriefingGraphNode) => Math.min(Math.max(n.depth, 0), MAX_COLS - 1);
  const maxD = Math.max(0, ...nodes.map(depthOf));
  const nCols = maxD + 1;
  const colOf = (n: BriefingGraphNode) => maxD - depthOf(n);

  const cols: string[][] = Array.from({ length: nCols }, () => []);
  const colIdx = new Map<string, number>();
  [...nodes]
    .sort((a, b) => a.label.localeCompare(b.label))
    .forEach((n) => {
      cols[colOf(n)].push(n.eid);
      colIdx.set(n.eid, colOf(n));
    });

  const nbr = new Map<string, string[]>();
  const link = (a: string, b: string) => {
    if (!colIdx.has(a) || !colIdx.has(b)) return;
    nbr.set(a, [...(nbr.get(a) ?? []), b]);
    nbr.set(b, [...(nbr.get(b) ?? []), a]);
  };
  graph.edges.forEach((e) => link(e.source, e.target));

  const maxLen = Math.max(1, ...cols.map((c) => c.length));
  const pos = new Map<string, number>();
  const refresh = () =>
    cols.forEach((c) => c.forEach((id, i) => pos.set(id, i + (maxLen - c.length) / 2)));
  refresh();

  const sweep = (c: number, side: "left" | "right") => {
    const bary = new Map<string, number>();
    for (const id of cols[c]) {
      const ns = (nbr.get(id) ?? []).filter((x) =>
        side === "left" ? colIdx.get(x)! < c : colIdx.get(x)! > c,
      );
      bary.set(
        id,
        ns.length ? ns.reduce((s, x) => s + (pos.get(x) ?? 0), 0) / ns.length : pos.get(id) ?? 0,
      );
    }
    cols[c].sort((a, b) => (bary.get(a) ?? 0) - (bary.get(b) ?? 0));
    refresh();
  };
  for (let round = 0; round < 4; round++) {
    for (let c = 1; c < nCols; c++) sweep(c, "left");
    for (let c = nCols - 2; c >= 0; c--) sweep(c, "right");
  }

  const placed = new Map<string, Placed>();
  nodes.forEach((n) => {
    const col = colOf(n);
    placed.set(n.eid, {
      ...n,
      col,
      x: PAD + col * (NODE_W + COL_GAP),
      y: PAD + (pos.get(n.eid) ?? 0) * (NODE_H + ROW_GAP),
    });
  });

  return {
    placed,
    width: PAD * 2 + nCols * NODE_W + (nCols - 1) * COL_GAP + 24, // 右側留給繞行的邊
    height: PAD * 2 + maxLen * (NODE_H + ROW_GAP) - ROW_GAP,
  };
}

function edgePath(s: Placed, t: Placed): string {
  const y1 = s.y + NODE_H / 2;
  const y2 = t.y + NODE_H / 2;
  if (t.col > s.col) {
    const x1 = s.x + NODE_W;
    const x2 = t.x;
    const dx = Math.max(24, (x2 - x1) / 2);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }
  // 同欄或反向：右側繞出去再回來
  const x1 = s.x + NODE_W;
  const x2 = t.x + NODE_W;
  const out = Math.max(x1, x2) + 22;
  return `M ${x1} ${y1} C ${out} ${y1}, ${out} ${y2}, ${x2} ${y2}`;
}

export default function BriefingGraph({ graph, rcEid, layers = [], walkOverlap = [] }: Props) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [hover, setHover] = useState<string | null>(null);
  const [layerIdx, setLayerIdx] = useState<number | null>(null);

  const { placed, width, height } = useMemo(() => computeLayout(graph), [graph]);
  const overlap = useMemo(() => new Set(walkOverlap), [walkOverlap]);

  const inEdges = useMemo(() => {
    const m = new Map<string, string[]>();
    graph.edges.forEach((e) => m.set(e.target, [...(m.get(e.target) ?? []), e.source]));
    return m;
  }, [graph.edges]);

  /** 目前被點亮的節點集合；null = 全部正常顯示 */
  const active = useMemo<Set<string> | null>(() => {
    if (hover) {
      const seen = new Set<string>([hover]);
      const stack = [hover];
      while (stack.length) {
        for (const p of inEdges.get(stack.pop()!) ?? []) {
          if (!seen.has(p)) {
            seen.add(p);
            stack.push(p);
          }
        }
      }
      return seen;
    }
    if (layerIdx !== null && layers[layerIdx]) return new Set(layers[layerIdx].node_eids);
    return null;
  }, [hover, layerIdx, layers, inEdges]);

  if (graph.nodes.length === 0) return null;

  const colorOf = (n: BriefingGraphNode) =>
    n.eid === rcEid ? C_RC : n.node_type === "critique" ? C_CRITIQUE : C_SUPPORT;
  const nodeOn = (id: string) => active === null || active.has(id);

  return (
    <div style={{ marginTop: 10 }}>
      {layers.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
          {layers.map((l, i) => (
            <button
              key={`${l.name}-${i}`}
              type="button"
              aria-pressed={layerIdx === i}
              onClick={() => setLayerIdx((cur) => (cur === i ? null : i))}
              style={{
                fontSize: "0.68rem",
                padding: "2px 9px",
                borderRadius: 999,
                cursor: "pointer",
                color: layerIdx === i ? "#14101c" : C_TEXT,
                background: layerIdx === i ? C_SUPPORT : "transparent",
                border: `1px solid ${C_SUPPORT}`,
              }}
            >
              {l.name}
            </button>
          ))}
        </div>
      )}

      <div style={{ overflowX: "auto", paddingBottom: 4 }}>
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="RC 推導拓樸圖：由最上游基礎概念依序匯流到 RC"
          style={{ display: "block" }}
        >
          <defs>
            <marker id={`${uid}-a`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0 0 L8 4 L0 8 z" fill={C_MUTED} />
            </marker>
            <marker id={`${uid}-c`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0 0 L8 4 L0 8 z" fill={C_CRITIQUE} />
            </marker>
          </defs>

          <g fill="none">
            {graph.edges.map((e, i) => {
              const s = placed.get(e.source);
              const t = placed.get(e.target);
              if (!s || !t) return null;
              const critique = e.edge_type === "critique";
              const on = active === null || (active.has(e.source) && active.has(e.target));
              return (
                <path
                  key={`${e.source}-${e.target}-${i}`}
                  d={edgePath(s, t)}
                  stroke={critique ? C_CRITIQUE : C_MUTED}
                  strokeWidth={on ? 1.4 : 1}
                  strokeDasharray={critique ? "4 3" : undefined}
                  opacity={on ? 0.9 : 0.12}
                  markerEnd={`url(#${uid}-${critique ? "c" : "a"})`}
                />
              );
            })}
          </g>

          <g>
            {[...placed.values()].map((n) => {
              const color = colorOf(n);
              const isRc = n.eid === rcEid;
              return (
                <g
                  key={n.eid}
                  tabIndex={0}
                  opacity={nodeOn(n.eid) ? 1 : 0.22}
                  onMouseEnter={() => setHover(n.eid)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(n.eid)}
                  onBlur={() => setHover(null)}
                  style={{ cursor: "default", outline: "none" }}
                >
                  <title>
                    {n.label}
                    {n.node_type === "critique" ? "（批判）" : ""}
                    {overlap.has(n.eid) ? "（你探索過）" : ""}
                  </title>
                  {overlap.has(n.eid) && (
                    <rect
                      x={n.x - 3}
                      y={n.y - 3}
                      width={NODE_W + 6}
                      height={NODE_H + 6}
                      rx={NODE_H / 2 + 3}
                      fill="none"
                      stroke={C_WALK}
                      strokeWidth={1.2}
                      strokeDasharray="3 2"
                    />
                  )}
                  <rect
                    x={n.x}
                    y={n.y}
                    width={NODE_W}
                    height={NODE_H}
                    rx={NODE_H / 2}
                    fill={color}
                    fillOpacity={isRc ? 0.3 : 0.16}
                    stroke={color}
                    strokeWidth={isRc ? 1.8 : 1}
                  />
                  <text
                    x={n.x + NODE_W / 2}
                    y={n.y + NODE_H / 2}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={11}
                    fontWeight={isRc ? 600 : 400}
                    fill={C_TEXT}
                  >
                    {shortLabel(n.label)}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 6, fontSize: "0.62rem", color: C_MUTED }}>
        <span><span style={{ color: C_RC }}>●</span> RC</span>
        <span><span style={{ color: C_SUPPORT }}>●</span> 支撐</span>
        <span><span style={{ color: C_CRITIQUE }}>●</span> 批判（虛線）</span>
        {overlap.size > 0 && <span><span style={{ color: C_WALK }}>◌</span> 你探索過</span>}
        <span>箭頭朝 RC 匯流；移到節點上可看上游鏈</span>
      </div>
    </div>
  );
}
