"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GraphData } from "./components/KnowledgeGraph";
import type { GlobalLayout, FoundationTree } from "./components/GlobalBrain";

const KnowledgeGraph = dynamic(() => import("./components/KnowledgeGraph"), { ssr: false });
const GlobalBrain    = dynamic(() => import("./components/GlobalBrain"),    { ssr: false });

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
// Constants
// ─────────────────────────────────────────────────────────────

const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY   = "kagrobot_phase";
const MAX_HISTORY = 60;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function saveHistory(msgs: ChatMessage[]): void {
  const trimmed = msgs.slice(-MAX_HISTORY);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed.slice(-20))); }
    catch { localStorage.removeItem(HISTORY_KEY); }
  }
}

function extractCandidateEids(parsed: Record<number, GraphData | null>): string[] {
  const eids: string[] = [];
  for (const g of Object.values(parsed)) {
    if (!g) continue;
    const qc = (g.query_concepts ?? [])[0];
    if (qc?.id && !eids.includes(qc.id)) eids.push(qc.id);
    for (const node of g.nodes ?? []) {
      if (node.matched && !eids.includes(node.id)) eids.push(node.id);
    }
  }
  return eids;
}

// ─────────────────────────────────────────────────────────────
// useBrainSize
// ─────────────────────────────────────────────────────────────

function useBrainSize(ref: React.RefObject<HTMLDivElement>) {
  const [size, setSize] = useState({ w: 560, h: 560 });

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({
          w: Math.floor(width)  || 560,
          h: Math.floor(height) || 560,
        });
      }
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);

  return size;
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

  // ── KnowledgeGraph state ───────────────────────────────────
  const [graphData,       setGraphData]       = useState<GraphData | null>(null);
  const [activeChipIndex, setActiveChipIndex] = useState<number | null>(null);
  const [selectedPathId,  setSelectedPathId]  = useState<number | null>(null);
  const [selectedLabel,   setSelectedLabel]   = useState<string | null>(null);
  const [confirmedPathId, setConfirmedPathId] = useState<number | null>(null);
  const [graphVisible,    setGraphVisible]    = useState(false);
  const [chipList, setChipList] = useState<
    { index: number; label: string; hasData: boolean }[]
  >([]);

  // ── GlobalBrain state ──────────────────────────────────────
  const [globalLayout,        setGlobalLayout]        = useState<GlobalLayout | null>(null);
  const [globalLayoutFetched, setGlobalLayoutFetched] = useState(false);
  const [activeCandidateEids, setActiveCandidateEids] = useState<string[]>([]);
  // selected-path activation
  const [activePathNodeEids,  setActivePathNodeEids]  = useState<string[]>([]);
  const [activePathEdges,     setActivePathEdges]     = useState<{source:string;target:string}[]>([]);
  const [activeFoundationTree, setActiveFoundationTree] = useState<FoundationTree>({ nodes: [], edges: [] });

  const allGraphsRef = useRef<Record<number, GraphData | null>>({});
  const isTypingRef  = useRef(false);
  const messagesRef  = useRef<ChatMessage[]>([]);
  const boxRef       = useRef<HTMLDivElement | null>(null);

  const brainColRef = useRef<HTMLDivElement>(null);
  const brainSize   = useBrainSize(brainColRef as React.RefObject<HTMLDivElement>);

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

  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, isLoading]);

  // ── fetch global layout ────────────────────────────────────
  useEffect(() => {
    if (messages.length === 0 || globalLayoutFetched || !API_BASE) return;
    setGlobalLayoutFetched(true);
    fetch(`${API_BASE.replace(/\/$/, "")}/graph/global-layout`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: GlobalLayout) => setGlobalLayout(d))
      .catch((err) => console.warn("[GlobalBrain] layout fetch failed:", err));
  }, [messages, globalLayoutFetched]);

  // ── selectedPathId → GlobalBrain active sets ──────────────
  useEffect(() => {
    if (selectedPathId === null || !graphData) {
      setActivePathNodeEids([]);
      setActivePathEdges([]);
      setActiveFoundationTree({ nodes: [], edges: [] });
      return;
    }

    const path = graphData.paths?.find((p) => p.path_id === selectedPathId);
    if (!path) {
      setActivePathNodeEids([]);
      setActivePathEdges([]);
      setActiveFoundationTree({ nodes: [], edges: [] });
      return;
    }

    // Collect concept node eids for the selected downstream path
    const pathConceptEids = (path.node_ids ?? []).filter((nid) => {
      const node = graphData.nodes?.find((n) => n.id === nid);
      return node?.node_type === "concept";
    });

    // Collect the edges for the selected path (source→target pairs from graphData.edges)
    const pathEdgeIdSet = new Set(path.edge_ids ?? []);
    const pathEdgePairs = (graphData.edges ?? [])
      .filter((e) => pathEdgeIdSet.has(e.id))
      .map((e) => ({ source: e.source as string, target: e.target as string }));

    // Foundation tree from graphData (all upstream intermediate nodes + edges)
    const foundationTree: FoundationTree = (graphData as any).foundation_tree ?? { nodes: [], edges: [] };

    setActivePathNodeEids(pathConceptEids);
    setActivePathEdges(pathEdgePairs);
    setActiveFoundationTree(foundationTree);
  }, [selectedPathId, graphData]);

  const canSend = !!input.trim() && !isLoading && !!sessionId;

  // ── typewriter ────────────────────────────────────────────
  async function typewriterAppend(fullText: string): Promise<void> {
    isTypingRef.current = true;
    const aiId = createSessionId();
    setMessages((prev) => [...prev, { id: aiId, role: "ai", content: "" }]);
    for (let i = 1; i <= fullText.length; i++) {
      setMessages((prev) =>
        prev.map((msg) => msg.id === aiId ? { ...msg, content: fullText.slice(0, i) } : msg)
      );
      await wait(10);
    }
    isTypingRef.current = false;
    saveHistory(messagesRef.current);
  }

  // ── chip 切換 ─────────────────────────────────────────────
  function handleChipClick(index: number) {
    if (isLoading) return;
    const g = allGraphsRef.current[index] ?? null;
    setActiveChipIndex(index);
    setGraphData(g);
    if (g) {
      const defaultPathId = typeof g.selected_path_id === "number" ? g.selected_path_id : null;
      setSelectedPathId(defaultPathId);
      const defaultPath = g.paths?.find((p) => p.path_id === defaultPathId);
      setSelectedLabel(defaultPath?.final_label ?? null);
    } else {
      setSelectedPathId(null);
      setSelectedLabel(null);
    }
  }

  function handlePathSelect(pathId: number, finalLabel: string) {
    setSelectedPathId(pathId);
    setSelectedLabel(finalLabel);
  }

  function initGraphPanel(parsed: Record<number, GraphData | null>) {
    allGraphsRef.current = parsed;
    const keys     = Object.keys(parsed).map(Number).filter((k) => parsed[k]);
    const firstIdx = keys.length ? Math.min(...keys) : null;
    const chips = Object.entries(parsed)
      .map(([k, g]) => ({
        index:   Number(k),
        label:   g?.query_concepts?.[0]?.name ?? `選項 ${k}`,
        hasData: !!g,
      }))
      .sort((a, b) => a.index - b.index);
    setChipList(chips);

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

    const candidateEids = extractCandidateEids(parsed);
    setActiveCandidateEids(candidateEids);
    // reset path highlights when new graphs arrive
    setActivePathNodeEids([]);
    setActivePathEdges([]);
    setActiveFoundationTree({ nodes: [], edges: [] });
  }

  // ── 送出訊息 ──────────────────────────────────────────────
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
        body:    JSON.stringify({
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
        for (const [k, v] of Object.entries(data.all_graphs)) parsed[Number(k)] = v;
        initGraphPanel(parsed);
      }

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [
        ...prev,
        { id: createSessionId(), role: "ai", content: "抱歉，連線失敗，請稍後再試。" },
      ]);
      saveHistory(messagesRef.current);
    } finally {
      setIsLoading(false);
      if (!isTypingRef.current) saveHistory(messagesRef.current);
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

    try {
      const response = await fetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:       sessionId,
          message:          confirmMsg.content,
          phase:            isReselect ? "main_agent" : "concept_resolved",
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
      saveHistory(messagesRef.current);
    } finally {
      setIsLoading(false);
      if (!isTypingRef.current) saveHistory(messagesRef.current);
    }
  }

  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem(SESSION_KEY, sessionId);
  }, [sessionId]);

  function resetConversation() {
    const sid = createSessionId();
    setSessionId(sid);
    setPhase("socratic");
    setMessages([]);
    setError("");
    allGraphsRef.current = {};
    setChipList([]);
    setActiveChipIndex(null);
    setGraphData(null);
    setSelectedPathId(null);
    setSelectedLabel(null);
    setConfirmedPathId(null);
    setGraphVisible(false);
    setActiveCandidateEids([]);
    setActivePathNodeEids([]);
    setActivePathEdges([]);
    setActiveFoundationTree({ nodes: [], edges: [] });
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
  }

  // ─────────────────────────────────────────────────────────
  // Derived display state
  // ─────────────────────────────────────────────────────────

  const showGraph = graphVisible && !!graphData && graphData.nodes.length > 0;

  const isReselect      = phase === "main_agent" && confirmedPathId !== null;
  const hasNewSelection = selectedPathId !== null && selectedPathId !== confirmedPathId;
  const showConfirm     = showGraph && selectedLabel !== null && (
    phase === "concept_resolved" || (isReselect && hasNewSelection)
  );

  const confirmBtnText = isLoading
    ? "分析中…"
    : isReselect
      ? `切換至「${selectedLabel}」→`
      : `確認「${selectedLabel}」，開始分析 →`;

  const inputPlaceholder = phase === "concept_resolved"
    ? "詢問路徑含義，或在右側選擇後確認…"
    : phase === "main_agent"
      ? "繼續詢問，或切換路徑…"
      : "輸入你的問題…";

  const sortedPaths = useMemo(() => {
    if (!graphData?.paths) return [];
    return [...graphData.paths].sort((a, b) => a.path_id - b.path_id);
  }, [graphData]);

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <main>
      {/* ── header ── */}
      <header>
        <div>
          <h1>KAG Robot</h1>
          <div className="session">
            {sessionId ? `${sessionId.slice(0, 20)}…` : "初始化中…"}
            {" · "}
            <span style={{ color: "var(--orange)", opacity: 0.7 }}>{phase}</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {showGraph && (
            <button type="button" onClick={() => setGraphVisible((v) => !v)}>
              {graphVisible ? "隱藏圖譜" : "顯示圖譜"}
            </button>
          )}
          <button type="button" onClick={resetConversation} disabled={isLoading}>
            開新對話
          </button>
        </div>
      </header>

      {/* ── three-column content area ── */}
      <div className="content-area">

        {/* ══ LEFT: chat column ══ */}
        <div className="chat-column">
          <div className="chat-box" ref={boxRef}>
            {messages.length === 0 ? (
              <div className="meta" style={{ textAlign: "center", marginTop: 24 }}>
                輸入問題，開始與知識場域對話
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  {msg.content}
                </div>
              ))
            )}
            {isLoading && (
              <div className="meta">
                <span style={{ color: "var(--orange)", opacity: 0.7 }}>⬡ 思考中…</span>
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <form onSubmit={submitMessage}>
              <textarea
                placeholder={inputPlaceholder}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (canSend) submitMessage(e as unknown as FormEvent);
                  }
                }}
                disabled={isLoading}
              />
              <button type="submit" disabled={!canSend}
                style={{ alignSelf: "flex-end" }}
              >
                送出
              </button>
            </form>
            {error && <div className="status">錯誤：{error}</div>}
          </div>
        </div>

        {/* ══ CENTRE: Brain column ══ */}
        <div className="brain-panel" ref={brainColRef}>
          <div className="brain-panel-label">Global Knowledge Field</div>
          <GlobalBrain
            layout={globalLayout}
            activeCandidateEids={activeCandidateEids}
            activePathNodeEids={activePathNodeEids}
            activePathEdges={activePathEdges}
            activeFoundationTree={activeFoundationTree}
            width={brainSize.w}
            height={Math.max(brainSize.h - 32, 300)}
          />
        </div>

        {/* ══ RIGHT: KnowledgeGraph panel ══ */}
        {showGraph && (
          <div className="graph-panel">
            <div className="graph-header">
              <span className="graph-title">知識圖譜</span>
              {selectedLabel ? (
                <span className="graph-selected">
                  {selectedPathId === confirmedPathId ? "✓ " : "◉ "}
                  {selectedLabel}
                </span>
              ) : (
                <span className="meta">點擊節點選取路徑</span>
              )}
            </div>

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
                        ? CHIP_COLORS[chip.index % CHIP_COLORS.length] : undefined,
                      color: activeChipIndex === chip.index
                        ? CHIP_COLORS[chip.index % CHIP_COLORS.length] : undefined,
                    }}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            )}

            {sortedPaths.length > 0 && (
              <div className="path-legend">
                {sortedPaths.map((p) => {
                  const color       = PATH_COLORS[p.path_id % PATH_COLORS.length];
                  const isActive    = selectedPathId === p.path_id;
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
                      {isConfirmed ? "✓ " : "↓ "}
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
              width={456}
              height={graphVisible && showConfirm ? 340 : 400}
            />

            {showConfirm && (
              <button
                className="confirm-btn"
                onClick={confirmPathSelection}
                disabled={isLoading}
                style={isReselect ? {
                  background:  "linear-gradient(135deg, rgba(48, 16, 96, 0.8), rgba(10, 58, 74, 0.8))",
                  borderColor: "#a060f0",
                  color:       "#a060f0",
                } : undefined}
              >
                {confirmBtnText}
              </button>
            )}
          </div>
        )}

        {/* ── graph panel 佔位 ── */}
        {!showGraph && (
          <div style={{
            display:        "flex",
            alignItems:     "center",
            justifyContent: "center",
            border:         "1px dashed rgba(30, 46, 72, 0.3)",
            borderRadius:   "14px",
            color:          "rgba(122, 138, 154, 0.25)",
            fontSize:       "0.65rem",
            letterSpacing:  "0.15em",
            textTransform:  "uppercase",
            fontFamily:     "monospace",
          }}>
            圖譜面板
          </div>
        )}

      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────
// Color tokens
// ─────────────────────────────────────────────────────────────

const CHIP_COLORS = ["#4f8ef7", "#00d4aa", "#e8a030"];

const PATH_COLORS = [
  "#4f8ef7", "#00d4aa", "#e8a030", "#e85d60",
  "#a060f0", "#3dd090", "#f07050", "#60b0f0",
];
