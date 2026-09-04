"use client";

/**
 * RandomWalkPanel.tsx — Random Walk Mode 的對話面板。
 *
 * 與既有 dialogue mode（page.tsx 的 chat-column + SocratesChatBot）並列的
 * 第二種對話模式，透過 SSE 訂閱 core/walk_stream.py 的事件，逐步呈現
 * K 條 path 的走動過程，最終顯示 RC 推薦排名。
 *
 * ── GlobalBrain 即時渲染的整合方式（刻意的最小改動設計）──────────
 *   GlobalBrainExplorer.tsx 已經有 activeCandidateEids prop，會對每個
 *   eid 畫標示環（見該檔案 ringGroupRef 那段）。這裡不需要另外開一條
 *   「random walk 專用視覺化」管線——每次收到 frontier_expanded 事件，
 *   把該 path 的 currentEid 更新，整批 activeCandidateEids 陣列變化時，
 *   GlobalBrainExplorer 既有的 useEffect 會自動重繪標示環位置，達到
 *   「socratic 走動」的視覺效果，不需要改動 GlobalBrainExplorer 本身。
 *   呼叫端（page.tsx）只需要在 random walk mode 時，把這裡的
 *   onActiveNodesChange 接到 GlobalBrainExplorer 的 activeCandidateEids
 *   prop 即可。
 *
 * ── 為什麼 walk_complete 事件本身不直接帶排名結果 ────────────────
 *   core/walk_stream.py 的 walk_complete.ranking 欄位目前只是
 *   WalkSession.all_foundation_eids()（每條 path 走過的原始節點清單），
 *   不是套用 graph_service/walk_ranking.py 公式算出的最終排名——算排名
 *   需要 rc_foundation_map / rc_semantic_scores 這兩份資料（來自既有的
 *   /graph/expand/batch、/brain/rc-semantic-scores 機制），這些不屬於
 *   單一 walk session 的狀態，放進 SSE event payload 沒有必要地放大
 *   每個事件的大小。因此排名改成 walk_complete 觸發後另外呼叫
 *   fetchRanking()（由呼叫端注入，比照 page.tsx 既有的
 *   fetchRCPaths/fetchFoundationTree 等 fetch-prop 慣例）。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { streamSSE, type SSEEvent } from "../../lib/sseFetch";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

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

export type WalkRankedRC = {
  eid:               string;
  label:             string;
  totalScore:        number;
  semantic:          number;
  interScore:        number;
  convergenceCount:  number;
  approx:            boolean;   // 對應後端 WalkRCRanked.approx：透過路徑探索找到、不在原本語意 pool 內
};

type WalkOverallStatus = "connecting" | "walking" | "awaiting_answer" | "complete" | "error";

type Props = {
  walkId:    string;
  streamUrl: string;   // 例如 `${API_BASE}/walk/stream/${walkId}`
  answerUrl: string;   // 例如 `${API_BASE}/walk/answer/${walkId}`

  /** 每次任一 path 的目前位置改變時呼叫，餵給 GlobalBrainExplorer 的
   *  activeCandidateEids prop，達成節點即時點亮的效果。 */
  onActiveNodesChange: (eids: string[]) => void;

  /** walk_complete 後呼叫，取得套用 walk_ranking.py 公式算出的真正排名。 */
  fetchRanking: () => Promise<WalkRankedRC[]>;

  /** 使用者在最終排名中選定一個 RC，呼應既有 PersonalBrain 的
   *  onLaunchSkill 流程（把選定的 RC 送進 main_agent 開始分析）。 */
  onSelectFinalRC?: (eid: string, label: string) => void;
};

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function upsertPath(
  prev: Record<number, WalkPathState>,
  pathId: number,
  patch: Partial<WalkPathState>,
): Record<number, WalkPathState> {
  const existing = prev[pathId] ?? {
    pathId,
    currentEid: "",
    currentLabel: "",
    visitedLabels: [],
    status: "pending" as WalkPathStatus,
    frontierPreview: [],
  };
  return { ...prev, [pathId]: { ...existing, ...patch } };
}

function activeEidsOf(paths: Record<number, WalkPathState>): string[] {
  return Object.values(paths)
    .map((p) => p.currentEid)
    .filter((eid): eid is string => !!eid);
}

// ─────────────────────────────────────────────────────────────
// RandomWalkPanel
// ─────────────────────────────────────────────────────────────

export default function RandomWalkPanel({
  walkId,
  streamUrl,
  answerUrl,
  onActiveNodesChange,
  fetchRanking,
  onSelectFinalRC,
}: Props) {
  const [paths, setPaths]   = useState<Record<number, WalkPathState>>({});
  const [status, setStatus] = useState<WalkOverallStatus>("connecting");
  const [error, setError]   = useState<string>("");

  const [pendingQuestion, setPendingQuestion] = useState<{ pathId: number; question: string } | null>(null);
  const [answerInput, setAnswerInput]         = useState("");
  const [submittingAnswer, setSubmittingAnswer] = useState(false);

  const [ranking, setRanking]           = useState<WalkRankedRC[] | null>(null);
  const [loadingRanking, setLoadingRanking] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  // ── 統一的事件處理，GET 初始連線與 POST resume 都走這裡 ──────
  const handleEvent = useCallback((event: SSEEvent) => {
    switch (event.type) {
      case "frontier_ready": {
        setPaths((prev) => upsertPath(prev, event.path_id, {
          status: "active",
          frontierPreview: (event.candidates ?? []) as FrontierPreviewItem[],
        }));
        break;
      }

      case "frontier_expanded": {
        setPaths((prev) => {
          const existing = prev[event.path_id];
          const visitedLabels = existing
            ? [...existing.visitedLabels, existing.currentLabel].filter(Boolean)
            : [];
          return upsertPath(prev, event.path_id, {
            currentEid:      event.to_eid,
            currentLabel:    event.to_label,
            visitedLabels,
            status:          "pending",   // 下一輪要先取得新的 frontier
            frontierPreview: [],
          });
        });
        break;
      }

      case "path_dead_end": {
        setPaths((prev) => upsertPath(prev, event.path_id, {
          status: "dead_end",
          deadEndReason: event.reason,
          frontierPreview: [],
        }));
        break;
      }

      case "user_question": {
        setPendingQuestion({ pathId: event.path_id, question: event.question });
        setStatus("awaiting_answer");
        break;
      }

      case "walk_complete": {
        setStatus("complete");
        setLoadingRanking(true);
        fetchRanking()
          .then((r) => setRanking([...r].sort((a, b) => b.totalScore - a.totalScore)))
          .catch((err) => console.warn("[RandomWalkPanel] fetchRanking failed:", err))
          .finally(() => setLoadingRanking(false));
        break;
      }

      case "walk_error": {
        setStatus("error");
        setError(event.message || "random walk 發生未預期錯誤");
        break;
      }

      default:
        console.warn("[RandomWalkPanel] unknown event type:", event.type);
    }
  }, [fetchRanking]);

  // 每次 paths 變化，把目前所有 path 的位置往上回報，餵給 GlobalBrainExplorer
  useEffect(() => {
    onActiveNodesChange(activeEidsOf(paths));
  }, [paths, onActiveNodesChange]);

  // ── 初始連線：mount 時開一段 GET stream ──────────────────────
  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("connecting");
    setError("");

    streamSSE(streamUrl, handleEvent, { signal: controller.signal })
      .then(() => {
        // stream 自然結束（停在 question 或 walk_complete），
        // 實際狀態已經由 handleEvent 更新，這裡不需要再動 status——
        // 除非兩者都沒發生就結束了（例如連線中途被伺服器關閉），
        // 此時保守起見標記錯誤，避免畫面卡在 "connecting" 不動。
        setStatus((cur) => (cur === "connecting" ? "error" : cur));
      })
      .catch((err) => {
        if (controller.signal.aborted) return;   // unmount 造成的中止，不當錯誤處理
        console.warn("[RandomWalkPanel] stream failed:", err);
        setStatus("error");
        setError(err instanceof Error ? err.message : "連線失敗");
      });

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamUrl]);

  // ── 使用者回答問題 → POST /walk/answer，回應本身就是新一段 stream ──
  const handleSubmitAnswer = useCallback(async () => {
    if (!pendingQuestion || !answerInput.trim() || submittingAnswer) return;

    setSubmittingAnswer(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamSSE(answerUrl, handleEvent, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ path_id: pendingQuestion.pathId, answer: answerInput.trim() }),
        signal:  controller.signal,
      });
      setPendingQuestion(null);
      setAnswerInput("");
      setStatus((cur) => (cur === "awaiting_answer" ? "walking" : cur));
    } catch (err) {
      console.warn("[RandomWalkPanel] answer stream failed:", err);
      setStatus("error");
      setError(err instanceof Error ? err.message : "回答提交失敗");
    } finally {
      setSubmittingAnswer(false);
    }
  }, [pendingQuestion, answerInput, submittingAnswer, answerUrl, handleEvent]);

  const orderedPaths = Object.values(paths).sort((a, b) => a.pathId - b.pathId);

  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 10,
      height: "100%", minHeight: 0, overflow: "hidden",
    }}>
      <div style={{
        fontSize: "0.62rem", letterSpacing: "0.14em", textTransform: "uppercase",
        color: "var(--orange)", opacity: 0.7, flexShrink: 0,
      }}>
        Random Walk · {walkId.slice(0, 12)}… · <StatusBadge status={status} />
      </div>

      {error && (
        <div className="status" style={{ flexShrink: 0 }}>錯誤：{error}</div>
      )}

      {/* ── K 條 path 的走動軌跡 ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, overflowY: "auto", flex: 1, minHeight: 0 }}>
        {orderedPaths.length === 0 && status === "connecting" && (
          <div className="meta">正在連線並取得起點…</div>
        )}
        {orderedPaths.map((p) => (
          <PathRow key={p.pathId} path={p} />
        ))}
      </div>

      {/* ── 提問區（暫停點）── */}
      {pendingQuestion && (
        <div style={{
          flexShrink: 0, border: "1px solid rgba(232,160,48,0.4)", borderRadius: 10,
          padding: "8px 10px", background: "rgba(232,160,48,0.06)",
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          <span style={{ fontSize: "0.7rem", color: "var(--orange)" }}>
            ❓ path {pendingQuestion.pathId}：{pendingQuestion.question}
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={answerInput}
              onChange={(e) => setAnswerInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSubmitAnswer(); }}
              placeholder="輸入你的回答…"
              disabled={submittingAnswer}
              style={{
                flex: 1, border: "1px solid var(--border)", borderRadius: 8,
                background: "rgba(13,18,32,0.8)", color: "var(--text)",
                padding: "6px 10px", fontSize: "0.75rem", fontFamily: "IBM Plex Mono, monospace",
              }}
            />
            <button
              type="button"
              onClick={handleSubmitAnswer}
              disabled={submittingAnswer || !answerInput.trim()}
            >
              {submittingAnswer ? "送出中…" : "回答"}
            </button>
          </div>
        </div>
      )}

      {/* ── 最終排名 ── */}
      {status === "complete" && (
        <div style={{ flexShrink: 0, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
          <div className="detail-section-label" style={{ marginBottom: 6 }}>
            推薦 Resulting Concept
          </div>
          {loadingRanking ? (
            <div className="detail-loading-bar" />
          ) : ranking && ranking.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {ranking.slice(0, 5).map((r, i) => (
                <RankedRCRow
                  key={r.eid}
                  rank={i + 1}
                  rc={r}
                  onSelect={onSelectFinalRC ? () => onSelectFinalRC(r.eid, r.label) : undefined}
                />
              ))}
            </div>
          ) : (
            <div className="meta">沒有找到合適的 RC。</div>
          )}
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

function PathRow({ path }: { path: WalkPathState }) {
  const dotColor =
    path.status === "dead_end" ? "#e85d60" :
    path.status === "active"   ? "var(--teal)" : "var(--muted)";

  return (
    <div style={{
      border: "1px solid var(--border)", borderRadius: 10, padding: "8px 10px",
      background: "rgba(13,18,32,0.55)", display: "flex", flexDirection: "column", gap: 4,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.62rem", color: "var(--muted)" }}>
        <span style={{
          display: "inline-block", width: 6, height: 6, borderRadius: "50%",
          background: dotColor, flexShrink: 0,
        }} />
        path {path.pathId}
        {path.status === "dead_end" && path.deadEndReason && (
          <span style={{ marginLeft: "auto", opacity: 0.7 }}>
            終止：{path.deadEndReason === "no_downstream" ? "無下游" : "達深度上限"}
          </span>
        )}
      </div>

      <div style={{ fontSize: "0.72rem", color: "var(--text)", lineHeight: 1.6, wordBreak: "break-word" }}>
        {[...path.visitedLabels, path.currentLabel].filter(Boolean).map((label, i, arr) => (
          <span key={i}>
            <span style={{ color: i === arr.length - 1 ? "var(--teal)" : "var(--text)" }}>{label}</span>
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
                border: `1px dashed ${c.is_rc ? "var(--teal)" : "rgba(127,119,221,0.4)"}`,
                color: c.is_rc ? "var(--teal)" : "rgba(175,169,236,0.6)",
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

function RankedRCRow({
  rank, rc, onSelect,
}: {
  rank: number;
  rc: WalkRankedRC;
  onSelect?: () => void;
}) {
  const pct = Math.max(0, Math.min(100, rc.totalScore * 100));
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      border: "1px solid rgba(127,119,221,0.25)", borderRadius: 10,
      padding: "6px 10px", cursor: onSelect ? "pointer" : "default",
    }}
      onClick={onSelect}
    >
      <span style={{ fontSize: "0.62rem", color: "var(--muted)", width: 14, flexShrink: 0 }}>{rank}.</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
          <span style={{
            fontSize: "0.72rem", color: "var(--text)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {rc.label}
          </span>
          {rc.approx && (
            <span title="透過路徑探索找到，未在原本語意排名前段" style={{
              fontSize: "0.55rem", padding: "0 5px", borderRadius: 8,
              border: "1px solid var(--orange)", color: "var(--orange)",
            }}>
              探索發現
            </span>
          )}
          {rc.convergenceCount >= 2 && (
            <span title={`${rc.convergenceCount} 條路徑收斂到此`} style={{
              fontSize: "0.55rem", padding: "0 5px", borderRadius: 8,
              border: "1px solid var(--teal)", color: "var(--teal)",
            }}>
              ⛓ 收斂×{rc.convergenceCount}
            </span>
          )}
        </div>
        <div style={{ display: "flex", height: 5, borderRadius: 3, overflow: "hidden", background: "rgba(83,74,183,0.1)" }}>
          <div style={{ width: `${pct}%`, background: "linear-gradient(90deg,#534AB7,#00d4aa)" }} />
        </div>
      </div>
      <span style={{ fontSize: "0.6rem", color: "#afa9ec", flexShrink: 0 }}>
        {rc.totalScore.toFixed(2)}
      </span>
    </div>
  );
}