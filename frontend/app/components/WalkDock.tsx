"use client";

import { useState } from "react";
import type {
  WalkPathState, WalkQAEntry, WalkOverallStatus,
} from "./walkTypes";

type Props = {
  paths: WalkPathState[];
  pathColors: Map<number, string>;
  status: WalkOverallStatus;
  pendingQuestion: { pathId: number; question: string } | null;
  qaLog: WalkQAEntry[];
  onSubmitAnswer: (answer: string) => void;
  answerSubmitting?: boolean;
  collapsed: boolean;
  onToggle: () => void;
  onDismiss: () => void;
  brainModeOn: boolean;               
  onToggleBrainMode: () => void;     
  brainModeDisabled?: boolean;        
};

export default function WalkDock({
  paths, pathColors, status, pendingQuestion, qaLog, onSubmitAnswer,
  answerSubmitting = false, collapsed, onToggle, onDismiss,
  brainModeOn, onToggleBrainMode, brainModeDisabled = false,   // ← 新增
}: Props) {
  const [draft, setDraft] = useState("");

  if (paths.length === 0) return null;

  const submit = () => {
    const text = draft.trim();
    if (!text || answerSubmitting) return;
    onSubmitAnswer(text);
    setDraft("");
  };

  return (
    <div style={{
      position: "fixed", right: 16, bottom: 16, zIndex: 500,
      maxWidth: "calc(100vw - 32px)", fontFamily: "IBM Plex Mono, monospace",
    }}>
      {collapsed ? (
        <button type="button" onClick={onToggle} style={{
          display: "flex", alignItems: "center", gap: 6, borderRadius: 20,
          padding: "8px 14px", border: `1px solid ${pendingQuestion ? "var(--orange)" : "rgba(0,212,170,0.4)"}`,
          background: "rgba(13,18,32,0.9)", color: pendingQuestion ? "var(--orange)" : "var(--teal)",
          fontSize: "0.72rem",
          boxShadow: pendingQuestion ? "0 0 12px rgba(232,160,48,0.25)" : "0 0 12px rgba(0,212,170,0.15)",
          backdropFilter: "blur(4px)", cursor: "pointer",
        }}>
          {pendingQuestion
            ? "🤔 有問題待回答"
            : `${brainModeOn ? "🧠" : "🌌"} 慢想紀錄 · ${paths.length} 條路徑`} ▲
        </button>
      ) : (
        <div style={{
          width: 340, maxHeight: "70vh", display: "flex", flexDirection: "column",
          border: "1px solid rgba(0,212,170,0.3)", borderRadius: 14,
          background: "rgba(13,18,32,0.92)", backdropFilter: "blur(6px)",
          boxShadow: "0 4px 24px rgba(0,0,0,0.4)", overflow: "hidden",
        }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "8px 10px", borderBottom: "1px solid rgba(0,212,170,0.2)", flexShrink: 0,
          }}>
            <span style={{ fontSize: "0.68rem", color: "var(--teal)" }}>
              🧠 慢想紀錄 <StatusTag status={status} />
            </span>
            <div style={{ display: "flex", gap: 4 }}>
              <button
                type="button"
                onClick={onToggleBrainMode}
                disabled={brainModeDisabled}
                title={brainModeDisabled ? "尚無激活節點" : (brainModeOn ? "切回全域視圖" : "切換到腦內模式")}
                style={{
                  fontSize:   "0.6rem",
                  padding:    "2px 8px",
                  border:     `1px solid ${brainModeOn ? "var(--teal)" : "rgba(127,119,221,0.3)"}`,
                  background: brainModeOn ? "rgba(0,212,170,0.15)" : "transparent",
                  color:      brainModeOn ? "var(--teal)" : "rgba(175,169,236,0.6)",
                  opacity:    brainModeDisabled ? 0.4 : 1,
                  cursor:     brainModeDisabled ? "not-allowed" : "pointer",
                }}
              >
                {brainModeOn ? "🧠 腦內模式" : "🌌 全域視圖"}
              </button>
              <button type="button" onClick={onToggle} style={{ fontSize: "0.6rem", padding: "2px 7px" }}>▾ 收合</button>
              <button type="button" onClick={onDismiss} style={{ fontSize: "0.6rem", padding: "2px 7px" }}>✕</button>
            </div>
          </div>

          <div style={{ padding: 10, overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 10 }}>

            {/* ── 內心獨白 ── */}
            {(qaLog.length > 0 || pendingQuestion) && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span style={{ fontSize: "0.58rem", letterSpacing: "0.14em", color: "rgba(232,160,48,0.6)" }}>
                  🤔 內心獨白
                </span>
                {qaLog.filter((q) => q.answer !== undefined).map((q, i) => (
                  <div key={i} style={{
                    fontSize: "0.68rem", fontStyle: "italic", color: "rgba(200,218,240,0.55)",
                    border: "1px dashed rgba(127,119,221,0.25)", borderRadius: 8, padding: "6px 9px",
                  }}>
                    <div>「{q.question}」</div>
                    <div style={{ color: "var(--teal)", marginTop: 2 }}>→ {q.answer}</div>
                  </div>
                ))}
                {pendingQuestion && (
                  <div style={{
                    border: "1px dashed rgba(232,160,48,0.5)", borderRadius: 8, padding: "8px 9px",
                    background: "rgba(232,160,48,0.06)", display: "flex", flexDirection: "column", gap: 6,
                  }}>
                    <span style={{ fontSize: "0.7rem", fontStyle: "italic", color: "var(--orange)" }}>
                      「{pendingQuestion.question}」
                    </span>
                    <div style={{ display: "flex", gap: 4 }}>
                      <input
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
                        placeholder="回答…"
                        disabled={answerSubmitting}
                        style={{
                          flex: 1, fontSize: "0.68rem", padding: "4px 8px", borderRadius: 8,
                          border: "1px solid rgba(232,160,48,0.35)", background: "rgba(13,18,32,0.7)",
                          color: "var(--text)", fontFamily: "IBM Plex Mono, monospace",
                        }}
                      />
                      <button type="button" onClick={submit} disabled={answerSubmitting || !draft.trim()}
                        style={{ fontSize: "0.65rem", padding: "4px 10px" }}>
                        送出
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── 路徑圖示 ── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10, overflowX: "auto" }}>
              {paths.map((p) => (
                <PathIconStepper key={p.pathId} path={p} color={pathColors.get(p.pathId) ?? "var(--teal)"} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusTag({ status }: { status: WalkOverallStatus }) {
  const label: Record<WalkOverallStatus, string> = {
    connecting: "連線中", walking: "走動中", awaiting_answer: "等待回答",
    complete: "已完成", error: "錯誤",
  };
  return <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.6rem" }}>· {label[status]}</span>;
}

// ── 迷你 node-link stepper ──────────────────────────────────
function PathIconStepper({ path, color }: { path: WalkPathState; color: string }) {
  const GAP = 30, PAD = 16, Y = 24, H = 56;
  const nodes = path.visited;
  const xOf = (i: number) => PAD + i * GAP;
  const width = Math.max(200, PAD * 2 + (nodes.length - 1) * GAP);
  const isDead = path.status === "dead_end";

  return (
    <div>
      <svg width={width} height={H} style={{ display: "block" }}>
        {nodes.length > 1 && (
          <line x1={xOf(0)} y1={Y} x2={xOf(nodes.length - 1)} y2={Y}
            stroke={color} strokeWidth={1.5} opacity={isDead ? 0.3 : 0.85} />
        )}

        {/* 已解決但沒被選中的分支（樹突） */}
        {path.frontierHistory.map((step, si) => {
          const idx = nodes.findIndex((n) => n.eid === step.fromEid);
          if (idx === -1) return null;
          const bx = xOf(idx);
          return step.candidates.map((c, ci) => {
            const dx = (ci - (step.candidates.length - 1) / 2) * 9;
            return (
              <g key={`${si}-${c.eid}`} opacity={0.3}>
                <line x1={bx} y1={Y} x2={bx + dx} y2={Y - 13} stroke={color} strokeWidth={1} strokeDasharray="2,2" />
                <circle cx={bx + dx} cy={Y - 13} r={2.5} fill="none" stroke={color} strokeWidth={1} />
                <title>{c.label}</title>
              </g>
            );
          });
        })}

        {/* 目前這一步還在猶豫的候選 */}
        {path.status === "active" && path.frontierPreview.map((c, ci) => {
          const bx = xOf(nodes.length - 1);
          const dx = (ci - (path.frontierPreview.length - 1) / 2) * 9;
          return (
            <g key={c.eid} opacity={0.65}>
              <line x1={bx} y1={Y} x2={bx + dx} y2={Y + 13} stroke={color} strokeWidth={1} strokeDasharray="2,2" />
              {c.is_rc ? (
                <polygon points={`${bx+dx},${Y+8} ${bx+dx+3},${Y+13} ${bx+dx},${Y+18} ${bx+dx-3},${Y+13}`}
                  fill="none" stroke={color} strokeWidth={1.2} />
              ) : (
                <circle cx={bx + dx} cy={Y + 13} r={3} fill="none" stroke={color} strokeWidth={1.2} />
              )}
              <title>{c.label}</title>
            </g>
          );
        })}

        {/* 走過的節點 */}
        {nodes.map((n, i) => {
          const isCurrent = i === nodes.length - 1 && !isDead;
          return (
            <g key={n.eid}>
              {isCurrent && (
                <circle cx={xOf(i)} cy={Y} r={8} fill={color} opacity={0.3}
                  style={{ animation: "wdock-pulse 1.6s ease-in-out infinite" }} />
              )}
              <circle cx={xOf(i)} cy={Y} r={n.isRc ? 5 : 4}
                fill={isDead && i === nodes.length - 1 ? "none" : color}
                stroke={color} strokeWidth={1.2}
                opacity={isDead && i !== nodes.length - 1 ? 0.4 : 1} />
              <title>{n.label}</title>
              {(isCurrent || (isDead && i === nodes.length - 1)) && (
                <text x={xOf(i)} y={Y + 20} textAnchor="middle" fontSize={9} fill={color}>
                  {n.label.length > 10 ? n.label.slice(0, 9) + "…" : n.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {isDead && path.deadEndReason && (
        <div style={{ fontSize: "0.58rem", color: "var(--muted)", opacity: 0.7 }}>
          終止：{path.deadEndReason === "no_downstream" ? "無下游" : "達深度上限"}
        </div>
      )}
      <style>{`
        @keyframes wdock-pulse { 0%,100% { r: 6; opacity: 0.35; } 50% { r: 10; opacity: 0.08; } }
      `}</style>
    </div>
  );
}