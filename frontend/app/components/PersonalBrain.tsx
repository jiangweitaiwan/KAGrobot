"use client";

/**
 * PersonalBrain.tsx(垂直剖面圖版)
 *
 * 與舊版的差異:
 *   - 移除整個 3D fMRI 場景(Three.js renderer / composer / bloom /
 *     point cloud / 呼吸動畫)。原本花了不少力氣做的按需渲染 /
 *     可見性偵測等效能優化,隨著場景整個拿掉一併移除。
 *   - 新增「聚焦(focus)」狀態機:不管是點 Top-3 推薦 RC 標籤、
 *     還是點鎖鏈上的任一個 downstream 節點(不分中繼或終點),
 *     一律呼叫 focusOn(eid, label),對該 eid 平行呼叫
 *     /graph/expand/{eid}(foundation tree)與 /graph/rc-paths/{eid}
 *     (downstream 鏈),取回後餵給 RCLineageViz 畫垂直剖面圖。
 *   - 鎖鏈本身用 buildDownstreamChains() 從 rc-paths 的 paths[] 裡,
 *     篩出「以目前聚焦 eid 為起點、屬於 Resulting_Concept 類型節點」
 *     的子序列取得,不需要額外的 adjacency 走訪,也不需要額外的
 *     後端改動。
 *   - Confirm 一律作用在「目前聚焦的 eid + 它自己 paths 裡的預設路徑」,
 *     語意等同於原本的 (selectedRCEid, selectedPathId) 呼叫方式,
 *     /chat 與 orchestrator 端完全不需要更動。
 *   - 視角切換從「3D / 2D」改為「垂直剖面 / 2D BERTopic」兩個選項。
 */

import { useCallback, useMemo, useState } from "react";
import type { FoundationTreeData, PersonalBrainConcept } from "./DetailPanel";
import type { GlobalLayout } from "./GlobalBrainExplorer";
import BertopicMap, { type BertopicLayoutData } from "./BertopicMap";
import RCLineageViz, { type ChainStep } from "./RCLineageViz";

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
  onConfirmRC:   (rcEid: string, rcLabel: string, pathId: number) => void;
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
 *
 * 做法:每條 path 的 node_ids 本身就是有序的完整節點序列,篩出屬於
 * Resulting_Concept 類型的子序列,找到 focusedEid 在其中的位置,
 * 往後 slice 即為它的 downstream 鏈(第一個元素就是 focusedEid 自己)。
 * 同一個終點(final eid)若有多條路徑,保留最短的一條。
 * tail 長度 <= 1(代表沒有走到別的 RC)則不計入。
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

/** Confirm 時的預設路徑:intent_score 較高者優先,其次選較短(較直接)的路徑。*/
function pickDefaultPath(paths: RCPathData["paths"]): RCPathData["paths"][number] | undefined {
  if (!paths.length) return undefined;
  return [...paths].sort((a, b) => {
    const scoreDiff = (b.intent_score ?? 0) - (a.intent_score ?? 0);
    if (scoreDiff !== 0) return scoreDiff;
    return a.node_ids.length - b.node_ids.length;
  })[0];
}

// ─────────────────────────────────────────────────────────────
// PersonalBrain
// ─────────────────────────────────────────────────────────────

export default function PersonalBrain({
  layout,
  bertopicLayout,
  personalBrain,
  rcRanking,
  onConfirmRC,
  fetchRCPaths,
  fetchFoundationTree,
  width  = 480,
  height = 480,
}: Props) {
  const [view, setView] = useState<"lineage" | "2d">("lineage");

  const [breadcrumb,      setBreadcrumb]      = useState<Crumb[]>([]);
  const [focusRCPathData, setFocusRCPathData] = useState<RCPathData | null>(null);
  const [focusFoundation, setFocusFoundation] = useState<FoundationTreeData | null>(null);
  const [loadingFocus,    setLoadingFocus]    = useState(false);

  const focused = breadcrumb.length > 0 ? breadcrumb[breadcrumb.length - 1] : null;

  const personalBrainEids = useMemo(
    () => new Set(personalBrain.map((c) => c.eid)),
    [personalBrain],
  );

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

  // ── 鎖鏈 / 剖面圖上的 downstream 節點點擊 → 一律重新聚焦 ──
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

  const handleConfirm = useCallback(() => {
    if (!focused || !focusRCPathData) return;
    const path = pickDefaultPath(focusRCPathData.paths);
    if (!path) return;
    onConfirmRC(focused.eid, path.final_label || focused.label, path.path_id);
  }, [focused, focusRCPathData, onConfirmRC]);

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
          { key: "lineage", label: "剖面" },
          { key: "2d",      label: "分佈" },
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

      {/* ── 垂直剖面圖 ── */}
      {view === "lineage" && (
        <RCLineageViz
          focusedEid={focused?.eid ?? ""}
          focusedLabel={focused?.label ?? ""}
          foundationTree={focusFoundation}
          downstreamChains={downstreamChains}
          personalBrainEids={personalBrainEids}
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
          position: "absolute", bottom: 52, left: 0, right: 0, padding: "0 12px",
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

      {/* ── Confirm button ── */}
      {focused && focusRCPathData && (
        <div style={{ position: "absolute", bottom: 10, left: 12, right: 12, pointerEvents: "auto" }}>
          <button
            onClick={handleConfirm}
            style={{
              width: "100%", padding: "8px",
              background: "linear-gradient(135deg, rgba(48,16,96,0.85), rgba(13,48,80,0.85))",
              border: "1px solid #7f77dd", color: "#afa9ec",
              fontSize: "0.75rem", fontWeight: 600, fontFamily: "monospace",
              borderRadius: 8, cursor: "pointer", letterSpacing: "0.04em",
              backdropFilter: "blur(6px)", transition: "box-shadow 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 0 14px rgba(127,119,221,0.3)")}
            onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "none")}
          >
            確認「{focused.label}」→ 開始分析
          </button>
        </div>
      )}
    </div>
  );
}
