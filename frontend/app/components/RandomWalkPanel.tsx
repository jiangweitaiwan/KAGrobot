"use client";

/**
 * RandomWalkPanel.tsx — Random Walk Mode 的探索過程展示面板（純展示版）。
 *
 * 不再自己開 SSE 連線或呼叫 ranking API：所有網路層與事件消費統一
 * 收斂到 page.tsx 的 handleWalkEvent，這裡只負責把 walkPaths /
 * pendingQuestion 畫出來。回答問題也交還給既有的 chat textarea
 * （page.tsx::submitMessage 在 phase === "walk_awaiting_answer" 時
 * 會轉呼叫 submitWalkAnswer），這裡只顯示問題文字，不再有自己的輸入框。
 *
 * 排名同樣交還：walk_complete 後 page.tsx 會重新拉
 * /brain/rc-semantic-scores，由 PersonalBrain 的 Top-3 標籤呈現，
 * 這個 panel 走完就淡出，不顯示任何排名結果。
 */

export type WalkPathStatus = "pending" | "active" | "dead_end";

export type FrontierPreviewItem = {
  eid:   string;
  label: string;
  is_rc: boolean;
};

export type WalkPathState = {
  pathId:          number;
  currentEid:      string;
  currentLabel:    string;
  visitedLabels:   string[];   // 不含目前 currentLabel，純粹是走過的軌跡
  status:          WalkPathStatus;
  deadEndReason?:  string;
  frontierPreview: FrontierPreviewItem[];
};

export type WalkOverallStatus = "connecting" | "walking" | "awaiting_answer" | "complete" | "error";

type Props = {
  paths:            WalkPathState[];              // 已依 pathId 排序
  pathColors:       Map<number, string>;          // pathId → hex color，與 GlobalBrainExplorer 共用
  status:           WalkOverallStatus;
  pendingQuestion:  { pathId: number; question: string } | null;
  error?:           string;
};

// ─────────────────────────────────────────────────────────────
// RandomWalkPanel
// ─────────────────────────────────────────────────────────────

export default function RandomWalkPanel({
  paths,
  pathColors,
  status,
  pendingQuestion,
  error,
}: Props) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 10,
      height: "100%", minHeight: 0, overflow: "hidden",
    }}>
      <div style={{
        fontSize: "0.62rem", letterSpacing: "0.14em", textTransform: "uppercase",
        color: "var(--orange)", opacity: 0.7, flexShrink: 0,
        display: "flex", alignItems: "center", gap: 6,
      }}>
        🧠 慢想中 · <StatusBadge status={status} />
      </div>

      {error && (
        <div className="status" style={{ flexShrink: 0 }}>錯誤：{error}</div>
      )}

      {/* ── K 條 path 的走動軌跡 ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, overflowY: "auto", flex: 1, minHeight: 0 }}>
        {paths.length === 0 && status === "connecting" && (
          <div className="meta">正在連線並取得起點…</div>
        )}
        {paths.map((p) => (
          <PathRow key={p.pathId} path={p} color={pathColors.get(p.pathId) ?? "var(--teal)"} />
        ))}
      </div>

      {/* ── 提問區（暫停點，顯示用；作答走下方 chat 輸入框）── */}
      {pendingQuestion && (
        <div style={{
          flexShrink: 0, border: "1px solid rgba(232,160,48,0.4)", borderRadius: 10,
          padding: "8px 10px", background: "rgba(232,160,48,0.06)",
        }}>
          <span style={{ fontSize: "0.7rem", color: "var(--orange)" }}>
            ❓ path {pendingQuestion.pathId}：{pendingQuestion.question}
          </span>
          <div style={{ fontSize: "0.6rem", color: "var(--muted)", marginTop: 4 }}>
            在下方輸入框回答即可繼續
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: WalkOverallStatus }) {
  const label: Record<WalkOverallStatus, string> = {
    connecting:      "連線中",
    walking:         "走動中",
    awaiting_answer: "等待回答",
    complete:        "已完成",
    error:           "錯誤",
  };
  const color: Record<WalkOverallStatus, string> = {
    connecting:      "var(--muted)",
    walking:         "var(--teal)",
    awaiting_answer: "var(--orange)",
    complete:        "var(--purple)",
    error:           "#e85d60",
  };
  return <span style={{ color: color[status] }}>{label[status]}</span>;
}

function PathRow({ path, color }: { path: WalkPathState; color: string }) {
  const dotColor =
    path.status === "dead_end" ? "#e85d60" :
    path.status === "active"   ? color : "var(--muted)";

  return (
    <div style={{
      border: `1px solid ${path.status === "dead_end" ? "var(--border)" : `${color}55`}`,
      borderRadius: 10, padding: "8px 10px",
      background: "rgba(13,18,32,0.55)", display: "flex", flexDirection: "column", gap: 4,
      opacity: path.status === "dead_end" ? 0.6 : 1,
      transition: "opacity 0.3s ease, border-color 0.3s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.62rem", color: "var(--muted)" }}>
        <span style={{
          display: "inline-block", width: 6, height: 6, borderRadius: "50%",
          background: dotColor, flexShrink: 0,
        }} />
        <span style={{ color }}>path {path.pathId}</span>
        {path.status === "dead_end" && path.deadEndReason && (
          <span style={{ marginLeft: "auto", opacity: 0.7 }}>
            終止：{path.deadEndReason === "no_downstream" ? "無下游" : "達深度上限"}
          </span>
        )}
      </div>

      <div style={{ fontSize: "0.72rem", color: "var(--text)", lineHeight: 1.6, wordBreak: "break-word" }}>
        {[...path.visitedLabels, path.currentLabel].filter(Boolean).map((label, i, arr) => (
          <span key={i}>
            <span style={{ color: i === arr.length - 1 ? color : "var(--text)" }}>{label}</span>
            {i < arr.length - 1 && <span style={{ color: "var(--muted)", margin: "0 4px" }}>→</span>}
          </span>
        ))}
      </div>

      {path.status === "active" && path.frontierPreview.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 2 }}>
          {path.frontierPreview.map((c) => (
            <span
              key={c.eid}
              title="候選下一步（由 LLM 自動判斷，不可手動選取）"
              style={{
                fontSize: "0.58rem", padding: "1px 7px", borderRadius: 10,
                border: `1px dashed ${c.is_rc ? color : "rgba(127,119,221,0.4)"}`,
                color: c.is_rc ? color : "rgba(175,169,236,0.6)",
              }}
            >
              {c.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}