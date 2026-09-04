"use client";

/**
 * WalkDock.tsx — 懸浮的「慢想紀錄」面板。
 *
 * 職責很單純：包住既有的 RandomWalkPanel，提供固定定位 + 收合/展開 +
 * 關閉三個外殼行為。不碰 RandomWalkPanel 內部的 PathRow / StatusBadge
 * 邏輯——那些顯示邏輯本來就與「是否懸浮」無關。
 *
 * 由 page.tsx 控制何時掛載（walkResultAvailable && !walkStreaming）、
 * 何時收合（walkDockCollapsed）、何時整個關閉（onDismiss 會連
 * walkPaths 一併清空）。
 */

import RandomWalkPanel, { type WalkPathState } from "./RandomWalkPanel";

type Props = {
  paths:      WalkPathState[];   // 已依 pathId 排序
  pathColors: Map<number, string>;
  collapsed:  boolean;
  onToggle:   () => void;
  onDismiss:  () => void;
};

export default function WalkDock({
  paths,
  pathColors,
  collapsed,
  onToggle,
  onDismiss,
}: Props) {
  if (paths.length === 0) return null;

  return (
    <div
      style={{
        position:   "fixed",
        right:      16,
        bottom:     16,
        zIndex:     500,
        maxWidth:   "calc(100vw - 32px)",
        fontFamily: "IBM Plex Mono, monospace",
      }}
    >
      {collapsed ? (
        <button
          type="button"
          onClick={onToggle}
          style={{
            display:        "flex",
            alignItems:     "center",
            gap:            6,
            borderRadius:   20,
            padding:        "8px 14px",
            border:         "1px solid rgba(0,212,170,0.4)",
            background:     "rgba(13,18,32,0.9)",
            color:          "var(--teal)",
            fontSize:       "0.72rem",
            boxShadow:      "0 0 12px rgba(0,212,170,0.15)",
            backdropFilter: "blur(4px)",
            cursor:         "pointer",
          }}
        >
          🧠 慢想紀錄 · {paths.length} 條路徑 ▲
        </button>
      ) : (
        <div
          style={{
            width:         320,
            maxHeight:     "60vh",
            display:       "flex",
            flexDirection: "column",
            border:        "1px solid rgba(0,212,170,0.3)",
            borderRadius:  14,
            background:    "rgba(13,18,32,0.92)",
            backdropFilter:"blur(6px)",
            boxShadow:     "0 4px 24px rgba(0,0,0,0.4)",
            overflow:      "hidden",
          }}
        >
          <div
            style={{
              display:        "flex",
              justifyContent: "space-between",
              alignItems:     "center",
              padding:        "8px 10px",
              borderBottom:   "1px solid rgba(0,212,170,0.2)",
              flexShrink:     0,
            }}
          >
            <span style={{ fontSize: "0.68rem", color: "var(--teal)" }}>
              🧠 慢想紀錄
            </span>
            <div style={{ display: "flex", gap: 4 }}>
              <button
                type="button"
                onClick={onToggle}
                style={{ fontSize: "0.6rem", padding: "2px 7px" }}
              >
                ▾ 收合
              </button>
              <button
                type="button"
                onClick={onDismiss}
                style={{ fontSize: "0.6rem", padding: "2px 7px" }}
              >
                ✕
              </button>
            </div>
          </div>
          <div style={{ padding: 10, overflowY: "auto", minHeight: 0 }}>
            <RandomWalkPanel
              paths={paths}
              pathColors={pathColors}
              status="complete"
              pendingQuestion={null}
            />
          </div>
        </div>
      )}
    </div>
  );
}