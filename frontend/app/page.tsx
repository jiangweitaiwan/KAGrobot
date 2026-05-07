"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GraphData } from "./components/KnowledgeGraph";

const KnowledgeGraph = dynamic(
  () => import("./components/KnowledgeGraph"),
  { ssr: false }
);

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

type Role = "user" | "ai";

type ChatMessage = {
  id:      string;
  role:    Role;
  content: string;
};

type ChatResponse = {
  reply:          string;
  phase:          string;
  socrates_phase?: string | null;
  all_graphs?:    Record<string, GraphData | null> | null;
};

// ─────────────────────────────────────────────────────────────
// Storage keys
// ─────────────────────────────────────────────────────────────

const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY   = "kagrobot_phase";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto)
    return crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** path 的方向標記前綴 */
function directionPrefix(direction?: string): string {
  return direction === "upstream" ? "↑ " : "↓ ";
}

// ─────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────

export default function Page() {
  const [sessionId, setSessionId] = useState("");
  const [phase,     setPhase]     = useState("socratic");
  const [messages,  setMessages]  = useState<ChatMessage[]>([]);
  const [input,     setInput]     = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error,     setError]     = useState("");

  // ── graph state ───────────────────────────────────────────
  const [allGraphs,       setAllGraphs]       = useState<Record<number, GraphData | null>>({});
  const [activeChipIndex, setActiveChipIndex] = useState<number | null>(null);
  const [graphData,       setGraphData]       = useState<GraphData | null>(null);
  const [selectedPathId,  setSelectedPathId]  = useState<number | null>(null);
  const [selectedLabel,   setSelectedLabel]   = useState<string | null>(null);
  const [confirmedPathId, setConfirmedPathId] = useState<number | null>(null);
  const [graphVisible,    setGraphVisible]    = useState(false);

  const boxRef = useRef<HTMLDivElement | null>(null);

  // ── init from localStorage ────────────────────────────────
  useEffect(() => {
    const storedSession = localStorage.getItem(SESSION_KEY);
    const sid           = storedSession || createSessionId();
    if (!storedSession) localStorage.setItem(SESSION_KEY, sid);

    const storedPhase = localStorage.getItem(PHASE_KEY) || "socratic";
    const rawHistory  = localStorage.getItem(HISTORY_KEY);
    const history     = rawHistory ? (JSON.parse(rawHistory) as ChatMessage[]) : [];

    setSessionId(sid);
    setPhase(storedPhase);
    setMessages(history);
  }, []);

  // ── auto-scroll ───────────────────────────────────────────
  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, isLoading]);

  const canSend = useMemo(
    () => !!input.trim() && !isLoading && !!sessionId,
    [input, isLoading, sessionId]
  );

  // ── typewriter ────────────────────────────────────────────
  async function typewriterAppend(fullText: string) {
    const aiId = createSessionId();
    setMessages((prev) => [...prev, { id: aiId, role: "ai", content: "" }]);
    for (let i = 1; i <= fullText.length; i++) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiId ? { ...msg, content: fullText.slice(0, i) } : msg
        )
      );
      await wait(10);
    }
  }

  // ── chip 切換（純前端，零後端請求）───────────────────────
  function handleChipClick(index: number) {
    if (isLoading) return;
    const g = allGraphs[index] ?? null;
    setActiveChipIndex(index);
    setGraphData(g);
    if (g) {
      const defaultPathId = typeof g.selected_path_id === "number"
        ? g.selected_path_id : null;
      setSelectedPathId(defaultPathId);
      const defaultPath = g.paths?.find((p) => p.path_id === defaultPathId);
      setSelectedLabel(defaultPath?.final_label ?? null);
    } else {
      setSelectedPathId(null);
      setSelectedLabel(null);
    }
  }

  // ── 使用者在圖上或 legend 點選路徑 ───────────────────────
  function handlePathSelect(pathId: number, finalLabel: string) {
    setSelectedPathId(pathId);
    setSelectedLabel(finalLabel);
  }

  // ── 初始化圖面板（收到 all_graphs 時呼叫）────────────────
  function initGraphPanel(parsed: Record<number, GraphData | null>) {
    setAllGraphs(parsed);
    const keys      = Object.keys(parsed).map(Number).filter((k) => parsed[k]);
    const firstIdx  = keys.length ? Math.min(...keys) : null;

    if (firstIdx !== null) {
      setActiveChipIndex(firstIdx);
      const firstGraph = parsed[firstIdx];
      setGraphData(firstGraph ?? null);
      setGraphVisible(true);

      if (firstGraph) {
        const defaultPathId = typeof firstGraph.selected_path_id === "number"
          ? firstGraph.selected_path_id : null;
        setSelectedPathId(defaultPathId);
        const defaultPath = firstGraph.paths?.find((p) => p.path_id === defaultPathId);
        setSelectedLabel(defaultPath?.final_label ?? null);
      }
    }
  }

  // ── 一般對話送出 ──────────────────────────────────────────
  async function submitMessage(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;
    if (!API_BASE) { setError("尚未設定 NEXT_PUBLIC_API_BASE_URL。"); return; }

    const text = input.trim();
    setInput("");
    setError("");
    setIsLoading(true);

    const userMsg: ChatMessage = { id: createSessionId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await fetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message:    text,
          phase:      phase === "concept_resolved" ? "concept_resolved" : "socratic",
        }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);

      const data      = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "socratic";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);

      if (data.all_graphs) {
        const parsed: Record<number, GraphData | null> = {};
        for (const [k, v] of Object.entries(data.all_graphs)) {
          parsed[Number(k)] = v;
        }
        initGraphPanel(parsed);
      }

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [
        ...prev,
        { id: createSessionId(), role: "ai", content: "抱歉，連線失敗，請稍後再試。" },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  // ── 確認路徑 ─────────────────────────────────────────────
  async function confirmPathSelection() {
    if (selectedPathId === null || !sessionId || isLoading) return;
    if (!API_BASE) { setError("尚未設定 NEXT_PUBLIC_API_BASE_URL。"); return; }
    if (selectedPathId === confirmedPathId && phase === "main_agent") return;

    setError("");
    setIsLoading(true);

    const isReselect = phase === "main_agent" && confirmedPathId !== null;
    const confirmMsg: ChatMessage = {
      id:      createSessionId(),
      role:    "user",
      content: isReselect
        ? `切換至路徑：${selectedLabel || `路徑 ${selectedPathId + 1}`}`
        : `確認選擇路徑：${selectedLabel || `路徑 ${selectedPathId + 1}`}`,
    };
    setMessages((prev) => [...prev, confirmMsg]);

    const sendPhase = isReselect ? "main_agent" : "concept_resolved";

    try {
      const response = await fetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id:       sessionId,
          message:          confirmMsg.content,
          phase:            sendPhase,
          selected_path_id: selectedPathId,
        }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);

      const data      = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "main_agent";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);
      setConfirmedPathId(selectedPathId);

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [
        ...prev,
        { id: createSessionId(), role: "ai", content: "抱歉，連線失敗，請稍後再試。" },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  // ── persist messages ──────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
  }, [sessionId, messages]);

  // ── reset ─────────────────────────────────────────────────
  function resetConversation() {
    const sid = createSessionId();
    setSessionId(sid);
    setPhase("socratic");
    setMessages([]);
    setError("");
    setAllGraphs({});
    setActiveChipIndex(null);
    setGraphData(null);
    setSelectedPathId(null);
    setSelectedLabel(null);
    setConfirmedPathId(null);
    setGraphVisible(false);
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
  }

  // ─────────────────────────────────────────────────────────
  // Derived display state
  // ─────────────────────────────────────────────────────────

  const showGraph = graphVisible && graphData && graphData.nodes.length > 0;

  const chipList = Object.entries(allGraphs)
    .map(([k, g]) => ({
      index:   Number(k),
      label:   g?.query_concepts?.[0]?.name ?? `選項 ${k}`,
      hasData: !!g,
    }))
    .sort((a, b) => a.index - b.index);

  const isReselect      = phase === "main_agent" && confirmedPathId !== null;
  const hasNewSelection = selectedPathId !== null && selectedPathId !== confirmedPathId;

  const showConfirm = showGraph && selectedLabel !== null && (
    phase === "concept_resolved" || (isReselect && hasNewSelection)
  );

  const confirmBtnText = isLoading
    ? "分析中…"
    : isReselect
      ? `切換至「${selectedLabel}」，重新分析 →`
      : `確認選擇「${selectedLabel}」，開始分析 →`;

  const inputPlaceholder = phase === "concept_resolved"
    ? "可詢問路徑含義，或直接在右側圖譜選擇路徑後按確認…"
    : phase === "main_agent"
      ? "繼續詢問分析細節，或在右側切換路徑…"
      : "輸入你的問題…";

  // path legend：上游排前，下游排後，同方向內保持 path_id 順序
  const sortedPaths = useMemo(() => {
    if (!graphData?.paths) return [];
    return [...graphData.paths].sort((a, b) => {
      const da = a.direction === "upstream" ? 0 : 1;
      const db = b.direction === "upstream" ? 0 : 1;
      if (da !== db) return da - db;
      return a.path_id - b.path_id;
    });
  }, [graphData]);

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <main className={showGraph ? "with-graph" : ""}>
      {/* ── header ── */}
      <header>
        <div>
          <h1>KAGrobot 對話機器人</h1>
          <div className="session">session: {sessionId || "初始化中…"}</div>
          <div className="meta">phase: {phase}</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {showGraph && (
            <button
              type="button"
              onClick={() => setGraphVisible((v) => !v)}
              style={{ fontSize: "0.8rem" }}
            >
              {graphVisible ? "隱藏圖譜" : "顯示圖譜"}
            </button>
          )}
          <button type="button" onClick={resetConversation} disabled={isLoading}>
            開新對話
          </button>
        </div>
      </header>

      {/* ── content area ── */}
      <div className="content-area">
        {/* chat column */}
        <div className="chat-column">
          <div className="chat-box" ref={boxRef}>
            {messages.length === 0 ? (
              <div className="meta">請輸入需求開始對話。</div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  {msg.content}
                </div>
              ))
            )}
            {isLoading && <div className="meta">AI 正在思考中…</div>}
          </div>

          <div>
            <form onSubmit={submitMessage}>
              <textarea
                placeholder={inputPlaceholder}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button type="submit" disabled={!canSend}>送出</button>
            </form>
            {error && <div className="status">錯誤：{error}</div>}
          </div>
        </div>

        {/* graph panel */}
        {showGraph && (
          <div className="graph-panel">
            <div className="graph-header">
              <span className="graph-title">知識圖譜</span>
              {selectedLabel ? (
                <span className="graph-selected">
                  {selectedPathId === confirmedPathId ? "✓ 已確認：" : "◉ 已選："}
                  {selectedLabel}
                </span>
              ) : (
                <span className="meta">點擊節點選取路徑</span>
              )}
            </div>

            {/* concept chips */}
            {chipList.length > 1 && (
              <div className="chip-legend">
                {chipList.map((chip) => (
                  <button
                    key={chip.index}
                    className={`concept-chip ${activeChipIndex === chip.index ? "active" : ""}`}
                    onClick={() => handleChipClick(chip.index)}
                    disabled={!chip.hasData || isLoading}
                    style={{
                      borderColor: activeChipIndex === chip.index
                        ? CHIP_COLORS[chip.index % CHIP_COLORS.length]
                        : undefined,
                      color: activeChipIndex === chip.index
                        ? CHIP_COLORS[chip.index % CHIP_COLORS.length]
                        : undefined,
                    }}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            )}

            {/* path legend：上游在前，下游在後，各自顯示方向前綴 */}
            {sortedPaths.length > 0 && (
              <div className="path-legend">
                {sortedPaths.map((p) => {
                  const prefix    = directionPrefix(p.direction);
                  const color     = p.direction === "upstream"
                    ? UPSTREAM_COLORS[p.path_id % UPSTREAM_COLORS.length]
                    : PATH_COLORS[p.path_id % PATH_COLORS.length];
                  const isActive  = selectedPathId === p.path_id;
                  const isConfirmed = p.path_id === confirmedPathId;

                  return (
                    <button
                      key={p.path_id}
                      className={`path-chip ${isActive ? "active" : ""}`}
                      onClick={() => handlePathSelect(p.path_id, p.final_label)}
                      style={{
                        borderColor: isActive ? color : undefined,
                        color:       isActive ? color : undefined,
                      }}
                    >
                      {isConfirmed ? "✓ " : ""}{prefix}
                      {p.final_label || `路徑 ${p.path_id + 1}`}
                    </button>
                  );
                })}
              </div>
            )}

            <KnowledgeGraph
              data={graphData}
              selectedPathId={selectedPathId}
              onPathSelect={handlePathSelect}
              width={520}
              height={400}
            />

            {/* 確認/重選按鈕 */}
            {showConfirm && (
              <button
                className="confirm-btn"
                onClick={confirmPathSelection}
                disabled={isLoading}
                style={isReselect ? {
                  background:  "linear-gradient(135deg, #301060, #0a3a4a)",
                  borderColor: "#a060f0",
                  color:       "#a060f0",
                } : undefined}
              >
                {confirmBtnText}
              </button>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

const CHIP_COLORS = [
  "#4f8ef7", "#00d4aa", "#e8a030",
];

// 下游路徑色系（暖色）
const PATH_COLORS = [
  "#4f8ef7", "#00d4aa", "#e8a030", "#e85d60",
  "#a060f0", "#3dd090", "#f07050", "#60b0f0",
];

// 上游路徑色系（冷色，與下游區隔）
const UPSTREAM_COLORS = [
  "#94a3b8", "#7dd3fc", "#a5b4fc", "#86efac",
  "#c4b5fd", "#93c5fd", "#6ee7b7", "#bfdbfe",
];
