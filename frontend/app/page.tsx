"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GraphData } from "./components/KnowledgeGraph";

// D3 touches the DOM — load only on client
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
  graph_data?:    GraphData | null;
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

// ─────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────

export default function Page() {
  const [sessionId,      setSessionId]      = useState("");
  const [phase,          setPhase]          = useState("socratic");
  const [messages,       setMessages]       = useState<ChatMessage[]>([]);
  const [input,          setInput]          = useState("");
  const [isLoading,      setIsLoading]      = useState(false);
  const [error,          setError]          = useState("");

  // ── graph state ───────────────────────────────────────────
  const [graphData,      setGraphData]      = useState<GraphData | null>(null);
  const [selectedPathId, setSelectedPathId] = useState<number | null>(null);
  const [selectedLabel,  setSelectedLabel]  = useState<string | null>(null);
  const [graphVisible,   setGraphVisible]   = useState(false);

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

  // ── submit ────────────────────────────────────────────────
  async function submitMessage(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;
    if (!API_BASE) {
      setError("尚未設定 NEXT_PUBLIC_API_BASE_URL。");
      return;
    }

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
        body:    JSON.stringify({ session_id: sessionId, message: text, phase }),
      });

      if (!response.ok) throw new Error(`API error ${response.status}`);

      const data = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "socratic";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);

      // ── graph data ──────────────────────────────────────
      if (data.graph_data) {
        setGraphData(data.graph_data);
        setGraphVisible(true);
        // if server already resolved a selected path, reflect it
        if (typeof data.graph_data.selected_path_id === "number") {
          const pid = data.graph_data.selected_path_id;
          setSelectedPathId(pid);
          const p = data.graph_data.paths.find((x) => x.path_id === pid);
          if (p) setSelectedLabel(p.final_label);
        }
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

  // ── when user manually selects a path in the graph ────────
  function handlePathSelect(pathId: number, finalLabel: string) {
    setSelectedPathId(pathId);
    setSelectedLabel(finalLabel);
    // If still in socratic phase, auto-fill the input with the selection
    if (phase === "socratic") {
      setInput(`我選擇「${finalLabel}」`);
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
    setGraphData(null);
    setSelectedPathId(null);
    setSelectedLabel(null);
    setGraphVisible(false);
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
  }

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  const showGraph = graphVisible && graphData && graphData.nodes.length > 0;

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

      {/* ── content area: chat + graph side by side ── */}
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
                placeholder="輸入你的問題…"
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
              {selectedLabel && (
                <span className="graph-selected">
                  ◉ 已選：{selectedLabel}
                </span>
              )}
              {!selectedLabel && (
                <span className="meta">點擊節點選取路徑</span>
              )}
            </div>

            {/* path legend */}
            {graphData.paths.length > 0 && (
              <div className="path-legend">
                {graphData.paths.map((p) => (
                  <button
                    key={p.path_id}
                    className={`path-chip ${selectedPathId === p.path_id ? "active" : ""}`}
                    onClick={() => handlePathSelect(p.path_id, p.final_label)}
                    style={{
                      borderColor: selectedPathId === p.path_id
                        ? PATH_COLORS[p.path_id % PATH_COLORS.length]
                        : undefined,
                      color: selectedPathId === p.path_id
                        ? PATH_COLORS[p.path_id % PATH_COLORS.length]
                        : undefined,
                    }}
                  >
                    {p.final_label || `路徑 ${p.path_id + 1}`}
                  </button>
                ))}
              </div>
            )}

            <KnowledgeGraph
              data={graphData}
              selectedPathId={selectedPathId}
              onPathSelect={handlePathSelect}
              width={520}
              height={400}
            />

            {/* confirm selection button (socratic phase only) */}
            {phase === "socratic" && selectedLabel && (
              <button
                className="confirm-btn"
                onClick={() => {
                  setInput(`我選擇「${selectedLabel}」`);
                }}
              >
                確認選擇「{selectedLabel}」→
              </button>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

// path colour palette (matches KnowledgeGraph.tsx TOKEN.path_colors)
const PATH_COLORS = [
  "#4f8ef7", "#00d4aa", "#e8a030", "#e85d60",
  "#a060f0", "#3dd090", "#f07050", "#60b0f0",
];
