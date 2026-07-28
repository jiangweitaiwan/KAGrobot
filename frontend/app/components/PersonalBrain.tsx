"use client";

/**
 * PersonalBrain.tsx（沙漏剖面圖版）
 *
 * 與上一版（RCLineageViz 垂直剖面圖）的差異：
 *   - 「剖面」視角改用 HourglassViz：waist = 聚焦 RC，上半錐 = downstream
 *     發散的多條 path，下半錐 = foundation_tree，rim 呼應 BERTopic 2D 平面。
 *   - 新增 bertopicPosByEid / bertopicCentroid 兩份衍生資料，供 HourglassViz
 *     計算節點的方位角（角度取自 BERTopic 語意方向，半徑取自樹狀深度）。
 *   - 「分佈」視角（BertopicMap）維持不變，讓使用者可以切換去看
 *     selected foundations 在整個語意地形裡的精確群聚效果——
 *     沙漏本身只保留方向性的呼應，不追求座標精確對齊。
 *   - 其餘邏輯（聚焦狀態機、鎖鏈萃取、confirm 流程）完全不變。
 */

import { useCallback, useMemo, useState } from "react";
import type { FoundationTreeData, PersonalBrainConcept } from "./DetailPanel";
import type { GlobalLayout } from "./GlobalBrainExplorer";
import BertopicMap, { type BertopicLayoutData } from "./BertopicMap";
import HourglassViz, { type ChainStep } from "./HourglassViz";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type RCRankedItem = {
  eid:        string;
  label:      string;
  totalScore: number;
};

export type RCPathNode = {
  id:           string;
  label:        string;
  node_type:    string;
  concept_type?: string | null;
  matched:      boolean;
  has_skill:    boolean;
};

export type RCPathData = {
  eid:   string;
  label: string;
  nodes: RCPathNode[];
  edges: { id: string; source: string; target: string }[];
  paths: {
    path_id:      number;
    node_ids:     string[];
    edge_ids:     string[];
    final_label:  string;
    intent_score?: number;
  }[];
};

type Props = {
  layout:        GlobalLayout | null;
  bertopicLayout: BertopicLayoutData | null;
  personalBrain: PersonalBrainConcept[];
  rcRanking:     RCRankedItem[];           // Top-3 from page.tsx
  onLaunchSkill: (rcEid: string, rcLabel: string) => void;
  fetchRCPaths:  (eid: string) => Promise<RCPathData | null>;
  fetchFoundationTree: (eid: string) => Promise<FoundationTreeData | null>;
  width?:        number;
  height?:       number;
};

type Crumb = { eid: string; label: string };

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function isRCNode(n: RCPathNode | undefined): boolean {
  return !!n && (n.concept_type || "").toLowerCase().includes("resulting");
}

/**
 * 從 rc-paths 的 paths[] 裡,萃取「以 focusedEid 為起點」的 downstream 鏈。
 * （邏輯與前一版完全相同，未變動。）
 */
function buildDownstreamChains(
  data:       RCPathData | null,
  focusedEid: string,
): ChainStep[][] {
  if (!data) return [];
  const nodeById = new Map(data.nodes.map((n) => [n.id, n]));

  const byFinal = new Map<string, ChainStep[]>();
  for (const p of data.paths) {
    const rcSeq = p.node_ids.filter((id) => isRCNode(nodeById.get(id)));
    const idx   = rcSeq.indexOf(focusedEid);
    if (idx === -1) continue;

    const tailIds = rcSeq.slice(idx);
    if (tailIds.length <= 1) continue;

    const finalEid = tailIds[tailIds.length - 1];
    const existing = byFinal.get(finalEid);
    if (!existing || tailIds.length < existing.length) {
      byFinal.set(finalEid, tailIds.map((id) => ({
        eid:   id,
        label: nodeById.get(id)?.label ?? id,
      })));
    }
  }
  return [...byFinal.values()];
}



// ─────────────────────────────────────────────────────────────
// PersonalBrain
// ─────────────────────────────────────────────────────────────

export default function PersonalBrain({
  layout,
  bertopicLayout,
  personalBrain,
  rcRanking,
  onLaunchSkill,
  fetchRCPaths,
  fetchFoundationTree,
  width  = 480,
  height = 480,
}: Props) {
  const [view, setView] = useState<"hourglass" | "2d">("hourglass");

  const [breadcrumb,      setBreadcrumb]      = useState<Crumb[]>([]);
  const [focusRCPathData, setFocusRCPathData] = useState<RCPathData | null>(null);
  const [focusFoundation, setFocusFoundation] = useState<FoundationTreeData | null>(null);
  const [loadingFocus,    setLoadingFocus]    = useState(false);

  const focused = breadcrumb.length > 0 ? breadcrumb[breadcrumb.length - 1] : null;

  const personalBrainEids = useMemo(
    () => new Set(personalBrain.map((c) => c.eid)),
    [personalBrain],
  );

  // ── BERTopic 座標查表 + 全域重心（供 HourglassViz 算方位角）──
  const bertopicPosByEid = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>();
    if (!bertopicLayout) return map;
    for (const n of bertopicLayout.nodes) {
      if (n.eid) map.set(n.eid, { x: n.x, y: n.y });
    }
    return map;
  }, [bertopicLayout]);

  const bertopicCentroid = useMemo(() => {
    if (!bertopicLayout || bertopicLayout.nodes.length === 0) return { x: 0, y: 0 };
    let sx = 0, sy = 0;
    for (const n of bertopicLayout.nodes) { sx += n.x; sy += n.y; }
    return { x: sx / bertopicLayout.nodes.length, y: sy / bertopicLayout.nodes.length };
  }, [bertopicLayout]);

  // ── 統一的聚焦函式:不論是 Top-3 標籤或鎖鏈上任一節點,都走這裡 ──
  const focusOn = useCallback(async (
    eid: string,
    label: string,
    opts?: { resetBreadcrumb?: boolean },
  ) => {
    setLoadingFocus(true);
    setBreadcrumb((prev) => (opts?.resetBreadcrumb ? [{ eid, label }] : [...prev, { eid, label }]));

    const [pathData, treeData] = await Promise.all([
      fetchRCPaths(eid),
      fetchFoundationTree(eid),
    ]);

    setFocusRCPathData(pathData);
    setFocusFoundation(treeData);
    setLoadingFocus(false);
  }, [fetchRCPaths, fetchFoundationTree]);

  // ── Top-3 推薦標籤點擊 ──
  const handleTagClick = useCallback((rc: RCRankedItem) => {
    // 再次點擊目前作為「根」的同一個標籤 → 取消聚焦
    if (breadcrumb.length === 1 && breadcrumb[0].eid === rc.eid) {
      setBreadcrumb([]);
      setFocusRCPathData(null);
      setFocusFoundation(null);
      return;
    }
    focusOn(rc.eid, rc.label, { resetBreadcrumb: true });
  }, [breadcrumb, focusOn]);

  // ── 沙漏上 downstream 節點點擊 → 一律重新聚焦 ──
  const handleFocusNode = useCallback((eid: string, label: string) => {
    focusOn(eid, label);
  }, [focusOn]);

  // ── Breadcrumb 上點擊較早的節點 → 跳回那一層(不必重新 fetch)──
  const handleBreadcrumbClick = useCallback((idx: number) => {
    if (idx === breadcrumb.length - 1) return;
    const target = breadcrumb[idx];
    setBreadcrumb(breadcrumb.slice(0, idx + 1));
    setLoadingFocus(true);
    Promise.all([fetchRCPaths(target.eid), fetchFoundationTree(target.eid)])
      .then(([pathData, treeData]) => {
        setFocusRCPathData(pathData);
        setFocusFoundation(treeData);
      })
      .finally(() => setLoadingFocus(false));
  }, [breadcrumb, fetchRCPaths, fetchFoundationTree]);

  const downstreamChains = useMemo(
    () => (focused ? buildDownstreamChains(focusRCPathData, focused.eid) : []),
    [focusRCPathData, focused],
  );

  // 目前展開拓墣中,所有帶 skill 的 RC eid(waist + downstream 皆可能命中)
  const skillEids = useMemo(
    () => new Set(
      (focusRCPathData?.nodes ?? [])
        .filter((n) => n.has_skill)
        .map((n) => n.id),
    ),
    [focusRCPathData],
  );

  const handleLaunchSkill = useCallback((eid: string, label: string) => {
    onLaunchSkill(eid, label);
  }, [onLaunchSkill]);

  return (
    <div style={{
      position: "relative", width, height, borderRadius: "20px", overflow: "hidden",
      background: "#030108", cursor: "default", flexShrink: 0,
    }}>

      {/* ── View Toggle ── */}
      <div style={{
        position: "absolute", top: 8, right: 12, display: "flex", gap: 4,
        zIndex: 10, pointerEvents: "auto",
      }}>
        {([
          { key: "hourglass", label: "沙漏" },
          { key: "2d",        label: "分佈" },
        ] as const).map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            style={{
              fontSize: "0.6rem", padding: "2px 8px", borderRadius: 10,
              border: `1px solid ${view === v.key ? "var(--purple)" : "rgba(127,119,221,0.2)"}`,
              background: view === v.key ? "rgba(127,119,221,0.15)" : "transparent",
              color: view === v.key ? "var(--purple)" : "rgba(127,119,221,0.4)",
              fontFamily: "IBM Plex Mono, monospace", cursor: "pointer", letterSpacing: "0.06em",
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* ── 沙漏剖面圖 ── */}
      {view === "hourglass" && (
        <HourglassViz
          focusedEid={focused?.eid ?? ""}
          focusedLabel={focused?.label ?? ""}
          foundationTree={focusFoundation}
          downstreamChains={downstreamChains}
          personalBrainEids={personalBrainEids}
          bertopicPosByEid={bertopicPosByEid}
          bertopicCentroid={bertopicCentroid}
          skillEids={skillEids}
          onLaunchSkill={handleLaunchSkill}
          loading={loadingFocus}
          onFocusNode={handleFocusNode}
          width={width}
          height={height}
        />
      )}

      {/* ── 2D BERTopic 分佈圖 ── */}
      {view === "2d" && bertopicLayout && (
        <BertopicMap data={bertopicLayout} personalBrain={personalBrain} width={width} height={height} />
      )}
      {view === "2d" && !bertopicLayout && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center", height: "100%",
          color: "rgba(127,119,221,0.25)", fontSize: "0.65rem", fontFamily: "monospace",
        }}>
          載入中…
        </div>
      )}

      {/* Title */}
      <div style={{
        position: "absolute", top: 10, left: 12, fontSize: "0.6rem",
        color: "rgba(127,119,221,0.4)", letterSpacing: "0.15em", textTransform: "uppercase",
        fontFamily: "monospace", pointerEvents: "none", userSelect: "none",
      }}>
        Personal Brain
      </div>

      {personalBrain.length > 0 && (
        <div style={{
          position: "absolute", top: 10, right: 90, fontSize: "0.6rem",
          color: "rgba(127,119,221,0.5)", fontFamily: "monospace",
          pointerEvents: "none", userSelect: "none",
        }}>
          {personalBrain.length} concepts
        </div>
      )}

      {!layout || personalBrain.length === 0 ? (
        <div style={{
          position: "absolute", inset: 0, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 6,
          color: "rgba(127,119,221,0.15)", fontSize: "0.65rem", letterSpacing: "0.14em",
          textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none",
        }}>
          <span>Personal Brain</span>
          <span style={{ fontSize: "0.55rem" }}>探索 concept,加入 foundations</span>
        </div>
      ) : null}

      {/* ── RC Tag row(Top-3 推薦)── */}
      {rcRanking.length > 0 && (
        <div style={{
          position: "absolute", top: 30, left: 0, right: 0,
          display: "flex", justifyContent: "center", gap: 6, padding: "4px 12px",
          flexWrap: "wrap", pointerEvents: "auto",
        }}>
          {rcRanking.slice(0, 3).map((rc, i) => {
            const isRoot = breadcrumb.length > 0 && breadcrumb[0].eid === rc.eid;
            const color  = ["#c060ff", "#7f77dd", "#534AB7"][i];
            return (
              <button
                key={rc.eid}
                onClick={() => handleTagClick(rc)}
                style={{
                  fontSize: "0.65rem", padding: "3px 10px", borderRadius: 16,
                  border: `1px solid ${isRoot ? color : "rgba(127,119,221,0.3)"}`,
                  background: isRoot ? "rgba(127,119,221,0.15)" : "rgba(3,1,8,0.7)",
                  color: isRoot ? color : "rgba(175,169,236,0.6)",
                  fontFamily: "monospace", cursor: "pointer", transition: "all 0.2s ease",
                  boxShadow: isRoot ? `0 0 8px ${color}44` : "none",
                  letterSpacing: "0.03em", backdropFilter: "blur(4px)",
                }}
              >
                {loadingFocus && isRoot && breadcrumb.length === 1 ? "…" : `${i + 1}. ${rc.label}`}
                <span style={{ marginLeft: 4, fontSize: "0.56rem", opacity: 0.7 }}>
                  {(rc.totalScore * 100).toFixed(0)}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* ── 鎖鏈麵包屑(顯示目前鑽到第幾層,可點回上一層)── */}
      {breadcrumb.length > 1 && (
        <div style={{
          position: "absolute", bottom: 14, left: 0, right: 0, padding: "0 12px",
          display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 4,
          pointerEvents: "auto",
        }}>
          {breadcrumb.map((c, i) => (
            <span key={c.eid} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {i > 0 && <span style={{ color: "rgba(79,142,247,0.4)", fontSize: "0.6rem" }}>⛓</span>}
              <button
                onClick={() => handleBreadcrumbClick(i)}
                style={{
                  fontSize: "0.62rem", padding: "2px 8px", borderRadius: 12,
                  border: `1px solid ${i === breadcrumb.length - 1 ? "#4f8ef7" : "rgba(79,142,247,0.25)"}`,
                  background: i === breadcrumb.length - 1 ? "rgba(79,142,247,0.15)" : "rgba(3,1,8,0.6)",
                  color: i === breadcrumb.length - 1 ? "#4f8ef7" : "rgba(79,142,247,0.5)",
                  fontFamily: "monospace", cursor: "pointer", backdropFilter: "blur(4px)",
                }}
              >
                {c.label}
              </button>
            </span>
          ))}
        </div>
      )}


    </div>
  );
}
