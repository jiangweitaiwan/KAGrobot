"use client";

import {
  FormEvent, useCallback, useEffect, useMemo, useRef, useState,
} from "react";
import dynamic from "next/dynamic";
import type { GlobalLayout, GlobalLayoutNode } from "./components/GlobalBrainExplorer";
import type {
  PersonalBrainConcept,
  ConceptDetail,
  FoundationTreeData,
  RCSemanticScore,
  RCFoundationMap,
} from "./components/DetailPanel";
import type { RCRankedItem, RCPathData } from "./components/PersonalBrain";

const GlobalBrainExplorer = dynamic(() => import("./components/GlobalBrainExplorer"), { ssr: false });
const PersonalBrain       = dynamic(() => import("./components/PersonalBrain"),       { ssr: false });
const DetailPanel         = dynamic(() => import("./components/DetailPanel"),         { ssr: false });

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

type Role = "user" | "ai";
type ChatMessage = { id: string; role: Role; content: string };

type ChatResponse = {
  reply:          string;
  phase:          string;
  socrates_phase?: string | null;
  all_graphs?:    Record<string, GraphDataLite | null> | null;
};

// Minimal graph type needed for rcFoundationMap extraction
type GraphDataLite = {
  query_concepts?: { id: string; name: string }[];
  nodes?:          { id: string; node_type: string; matched?: boolean }[];
  paths?: {
    path_id:       number;
    node_ids:      string[];
    edge_ids:      string[];
    final_concept: string;
    final_label:   string;
    foundations?:  { element_id: string; label: string }[];
  }[];
  foundations?: { element_id: string; label: string }[];
};

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY   = "kagrobot_phase";
const USER_ID_KEY = "kagrobot_user_id";
const MAX_HISTORY = 60;
const ALPHA       = 0.5;   // weight for semantic_sim in score(RC)

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function createId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function wait(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function saveHistory(msgs: ChatMessage[]): void {
  const trimmed = msgs.slice(-MAX_HISTORY);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed)); }
  catch { localStorage.removeItem(HISTORY_KEY); }
}

function extractCandidateEids(allGraphs: Record<number, GraphDataLite | null>): string[] {
  const eids: string[] = [];
  for (const g of Object.values(allGraphs)) {
    if (!g) continue;
    const qc = (g.query_concepts ?? [])[0];
    if (qc?.id && !eids.includes(qc.id)) eids.push(qc.id);
    for (const node of g.nodes ?? []) {
      if (node.matched && !eids.includes(node.id)) eids.push(node.id);
    }
  }
  return eids;
}

/**
 * Extract foundation EIDs per RC from all_graphs.
 * The backend already ran UpstreamBFS and stored foundations on each graph.
 */
function buildRCFoundationMap(allGraphs: Record<number, GraphDataLite | null>): RCFoundationMap {
  const map: RCFoundationMap = new Map();
  for (const g of Object.values(allGraphs)) {
    if (!g) continue;
    const graphFoundations = (g.foundations ?? []).map((f) => f.element_id);
    for (const path of g.paths ?? []) {
      if (!path.final_concept) continue;
      const fEids = graphFoundations.length > 0
        ? graphFoundations
        : (path.foundations ?? []).map((f) => f.element_id);
      if (!map.has(path.final_concept)) {
        map.set(path.final_concept, {
          label:         path.final_label || path.final_concept,
          foundationEids: fEids,
        });
      }
    }
  }
  return map;
}

/**
 * Compute RC ranking from personal brain + semantic scores.
 * α * semantic_sim + (1-α) * weighted_inter_score
 */
function computeRCRanking(
  personalBrain:    PersonalBrainConcept[],
  rcSemanticScores: RCSemanticScore[],
  rcFoundationMap:  RCFoundationMap,
  alpha:            number,
): RCRankedItem[] {
  const weightMap  = new Map(personalBrain.map((c) => [c.eid, c.weight]));
  const semMap     = new Map(rcSemanticScores.map((s) => [s.eid, s.score]));
  const results: RCRankedItem[] = [];

  rcFoundationMap.forEach(({ label, foundationEids }, rcEid) => {
    const semantic = semMap.get(rcEid) ?? 0;
    const n        = foundationEids.length;
    let   weightedSum = 0;
    for (const feid of foundationEids) {
      const w = weightMap.get(feid);
      if (w !== undefined) weightedSum += w;
    }
    const interScore  = n > 0 ? weightedSum / n : 0;
    const totalScore  = alpha * semantic + (1 - alpha) * interScore;
    results.push({ eid: rcEid, label, totalScore });
  });

  return results.sort((a, b) => b.totalScore - a.totalScore).slice(0, 3);
}

// ─────────────────────────────────────────────────────────────
// usePanelSize
// ─────────────────────────────────────────────────────────────

function usePanelSize(ref: React.RefObject<HTMLDivElement>) {
  const [size, setSize] = useState({ w: 400, h: 400 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({ w: Math.floor(width) || 400, h: Math.floor(height) || 400 });
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
  // ── Identity ───────────────────────────────────────────────
  const [sessionId, setSessionId] = useState("");
  const [userId,    setUserId]    = useState("");
  const [phase,     setPhase]     = useState("socratic");

  // ── Chat ───────────────────────────────────────────────────
  const [messages,  setMessages]  = useState<ChatMessage[]>([]);
  const [input,     setInput]     = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error,     setError]     = useState("");

  // ── Global layout ──────────────────────────────────────────
  const [globalLayout,        setGlobalLayout]        = useState<GlobalLayout | null>(null);
  const [globalLayoutFetched, setGlobalLayoutFetched] = useState(false);
  const [activeCandidateEids, setActiveCandidateEids] = useState<string[]>([]);

  // ── RC Intelligence ────────────────────────────────────────
  const [rcSemanticScores, setRCSemanticScores] = useState<RCSemanticScore[]>([]);
  const [rcFoundationMap,  setRCFoundationMap]  = useState<RCFoundationMap>(new Map());
  const allGraphsRef = useRef<Record<number, GraphDataLite | null>>({});

  // ── Personal Brain ─────────────────────────────────────────
  const [personalBrain, setPersonalBrain] = useState<PersonalBrainConcept[]>([]);
  const [brainLoaded,   setBrainLoaded]   = useState(false);

  // ── Detail Panel ───────────────────────────────────────────
  const [selectedNode,   setSelectedNode]   = useState<GlobalLayoutNode | null>(null);
  const [detail,         setDetail]         = useState<ConceptDetail | null>(null);
  const [foundationTree, setFoundationTree] = useState<FoundationTreeData | null>(null);
  const [loadingTree,    setLoadingTree]    = useState(false);

  // ── Refs ───────────────────────────────────────────────────
  const isTypingRef   = useRef(false);
  const messagesRef   = useRef<ChatMessage[]>([]);
  const boxRef        = useRef<HTMLDivElement | null>(null);
  const explorerColRef  = useRef<HTMLDivElement>(null);
  const personalColRef  = useRef<HTMLDivElement>(null);
  const explorerSize    = usePanelSize(explorerColRef as React.RefObject<HTMLDivElement>);
  const personalSize    = usePanelSize(personalColRef  as React.RefObject<HTMLDivElement>);

  // ── RC Ranking (derived) ───────────────────────────────────
  const rcRanking = useMemo(
    () => computeRCRanking(personalBrain, rcSemanticScores, rcFoundationMap, ALPHA),
    [personalBrain, rcSemanticScores, rcFoundationMap],
  );

  // ──────────────────────────────────────────────────────────
  // Init from localStorage
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    let uid = localStorage.getItem(USER_ID_KEY);
    if (!uid) { uid = createId(); localStorage.setItem(USER_ID_KEY, uid); }
    setUserId(uid);

    const storedSession = localStorage.getItem(SESSION_KEY);
    const sid           = storedSession || createId();
    if (!storedSession) localStorage.setItem(SESSION_KEY, sid);
    setSessionId(sid);

    setPhase(localStorage.getItem(PHASE_KEY) || "socratic");
    const rawHistory = localStorage.getItem(HISTORY_KEY);
    setMessages(rawHistory ? JSON.parse(rawHistory) : []);
  }, []);

  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, isLoading]);

  // ── Load Personal Brain ────────────────────────────────────
  useEffect(() => {
    if (!userId || !API_BASE || brainLoaded) return;
    setBrainLoaded(true);
    fetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => { if (d.concepts?.length) setPersonalBrain(d.concepts); })
      .catch((err) => console.warn("[PersonalBrain] load failed:", err));
  }, [userId, brainLoaded]);

  // ── Fetch global layout ────────────────────────────────────
  useEffect(() => {
    if (messages.length === 0 || globalLayoutFetched || !API_BASE) return;
    setGlobalLayoutFetched(true);
    fetch(`${API_BASE.replace(/\/$/, "")}/graph/global-layout`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: GlobalLayout) => setGlobalLayout(d))
      .catch((err) => console.warn("[GlobalBrain] layout fetch failed:", err));
  }, [messages, globalLayoutFetched]);

  // ──────────────────────────────────────────────────────────
  // GlobalBrainExplorer node click
  // ──────────────────────────────────────────────────────────

  const handleNodeClick = useCallback(async (node: GlobalLayoutNode) => {
    setSelectedNode(node);
    setDetail(null);
    setFoundationTree(null);
    setLoadingTree(true);

    if (!API_BASE) { setLoadingTree(false); return; }
    const base = API_BASE.replace(/\/$/, "");

    const [detailRes, treeRes] = await Promise.allSettled([
      fetch(`${base}/concept/detail/${node.id}`).then((r) => r.json() as Promise<ConceptDetail>),
      fetch(`${base}/graph/expand/${node.id}`).then((r) => r.json() as Promise<FoundationTreeData>),
    ]);

    if (detailRes.status === "fulfilled") setDetail(detailRes.value);
    if (treeRes.status === "fulfilled")   setFoundationTree(treeRes.value);
    setLoadingTree(false);
  }, []);

  // ──────────────────────────────────────────────────────────
  // Personal Brain CRUD
  // ──────────────────────────────────────────────────────────

  const persistBrainEntry = useCallback((entry: PersonalBrainConcept) => {
    if (!API_BASE || !userId) return;
    fetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).catch((err) => console.warn("[brain/add]", err));
  }, [userId]);

  const addFoundationsToBrain = useCallback((
    items: { eid: string; label: string }[],
  ) => {
    setPersonalBrain((prev) => {
      const updated = [...prev];
      const brainMap = new Map(updated.map((c) => [c.eid, c]));
      for (const { eid, label } of items) {
        const existing = brainMap.get(eid);
        const entry: PersonalBrainConcept = existing
          ? { ...existing, weight: Math.min(1.0, existing.weight + 0.1) }
          : { eid, label, source: "manual", weight: 0.5, addedAt: Date.now() };
        brainMap.set(eid, entry);
        persistBrainEntry(entry);
      }
      return [...brainMap.values()];
    });
  }, [persistBrainEntry]);

  const removeFromBrain = useCallback((eid: string) => {
    setPersonalBrain((prev) => prev.filter((c) => c.eid !== eid));
    if (!API_BASE || !userId) return;
    fetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts/${eid}`, { method: "DELETE" })
      .catch((err) => console.warn("[brain/remove]", err));
  }, [userId]);

  const updateWeight = useCallback((eid: string, weight: number) => {
    setPersonalBrain((prev) => prev.map((c) => c.eid === eid ? { ...c, weight } : c));
    if (!API_BASE || !userId) return;
    fetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts/${eid}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weight }),
    }).catch((err) => console.warn("[brain/weight]", err));
  }, [userId]);

  // ──────────────────────────────────────────────────────────
  // PersonalBrain: fetch RC paths
  // ──────────────────────────────────────────────────────────

  const fetchRCPaths = useCallback(async (eid: string): Promise<RCPathData | null> => {
    if (!API_BASE) return null;
    try {
      const r = await fetch(`${API_BASE.replace(/\/$/, "")}/graph/rc-paths/${eid}`);
      if (!r.ok) throw new Error(`${r.status}`);
      return await r.json();
    } catch (err) {
      console.warn("[rc-paths]", err);
      return null;
    }
  }, []);

  // ──────────────────────────────────────────────────────────
  // PersonalBrain: confirm RC → MainAgent
  // ──────────────────────────────────────────────────────────

  const handleConfirmRC = useCallback(async (
    rcEid:   string,
    rcLabel: string,
    pathId:  number,
  ) => {
    if (!sessionId || isLoading || !API_BASE) return;

    // Auto-add RC to brain on confirm
    addFoundationsToBrain([{ eid: rcEid, label: rcLabel }]);

    const confirmMsg: ChatMessage = {
      id: createId(), role: "user",
      content: `確認選擇路徑：${rcLabel}`,
    };
    setMessages((prev) => [...prev, confirmMsg]);
    setIsLoading(true);
    setError("");

    try {
      const resp = await fetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:       sessionId,
          message:          confirmMsg.content,
          phase:            phase === "main_agent" ? "main_agent" : "concept_resolved",
          selected_path_id: pathId,
        }),
      });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = (await resp.json()) as ChatResponse;
      setPhase(data.phase || "main_agent");
      localStorage.setItem(PHASE_KEY, data.phase || "main_agent");
      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [...prev, { id: createId(), role: "ai", content: "抱歉，連線失敗。" }]);
    } finally {
      setIsLoading(false);
      if (!isTypingRef.current) saveHistory(messagesRef.current);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, isLoading, phase, addFoundationsToBrain]);

  // ──────────────────────────────────────────────────────────
  // Chat
  // ──────────────────────────────────────────────────────────

  const canSend = !!input.trim() && !isLoading && !!sessionId;

  async function typewriterAppend(fullText: string): Promise<void> {
    isTypingRef.current = true;
    const aiId = createId();
    setMessages((prev) => [...prev, { id: aiId, role: "ai", content: "" }]);
    for (let i = 1; i <= fullText.length; i++) {
      setMessages((prev) =>
        prev.map((msg) => msg.id === aiId ? { ...msg, content: fullText.slice(0, i) } : msg),
      );
      await wait(10);
    }
    isTypingRef.current = false;
    saveHistory(messagesRef.current);
  }

  async function submitMessage(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;
    if (!API_BASE) { setError("尚未設定 NEXT_PUBLIC_API_BASE_URL。"); return; }

    const text = input.trim();
    setInput("");
    setError("");
    setIsLoading(true);

    const userMsg: ChatMessage = { id: createId(), role: "user", content: text };
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
        const parsed: Record<number, GraphDataLite | null> = {};
        for (const [k, v] of Object.entries(data.all_graphs)) parsed[Number(k)] = v;
        allGraphsRef.current = parsed;

        const newMap = buildRCFoundationMap(parsed);
        setRCFoundationMap(newMap);
        setActiveCandidateEids(extractCandidateEids(parsed));

        // Fetch semantic scores using the query from socrates
        // We derive query from the latest chat message
        const query = text;
        if (query && API_BASE) {
          fetch(`${API_BASE.replace(/\/$/, "")}/brain/rc-semantic-scores?query=${encodeURIComponent(query)}`)
            .then((r) => r.json())
            .then((d) => setRCSemanticScores(d.scores ?? []))
            .catch((err) => console.warn("[rc-semantic-scores]", err));
        }
      }

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [...prev, { id: createId(), role: "ai", content: "抱歉，連線失敗，請稍後再試。" }]);
      saveHistory(messagesRef.current);
    } finally {
      setIsLoading(false);
      if (!isTypingRef.current) saveHistory(messagesRef.current);
    }
  }

  function resetConversation() {
    const sid = createId();
    setSessionId(sid);
    setPhase("socratic");
    setMessages([]);
    setError("");
    allGraphsRef.current = {};
    setActiveCandidateEids([]);
    setRCFoundationMap(new Map());
    setRCSemanticScores([]);
    setSelectedNode(null);
    setDetail(null);
    setFoundationTree(null);
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
  }

  const inputPlaceholder =
    phase === "concept_resolved" ? "詢問路徑含義…" :
    phase === "main_agent"       ? "繼續詢問分析…" :
                                   "輸入你的問題…";

  // ──────────────────────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────────────────────

  return (
    <main>
      {/* ── Header ── */}
      <header>
        <div>
          <h1>KAG Robot</h1>
          <div className="session">
            {sessionId ? `${sessionId.slice(0, 20)}…` : "初始化中…"}
            {" · "}
            <span style={{ color: "var(--orange)", opacity: 0.7 }}>{phase}</span>
            {personalBrain.length > 0 && (
              <>
                {" · "}
                <span style={{ color: "var(--purple)", opacity: 0.8 }}>
                  ◉ {personalBrain.length} in brain
                </span>
              </>
            )}
          </div>
        </div>
        <button type="button" onClick={resetConversation} disabled={isLoading}>
          開新對話
        </button>
      </header>

      {/* ── Four-column layout: chat | GlobalBrainExplorer | DetailPanel | PersonalBrain ── */}
      <div className="content-area">

        {/* ══ Chat ══ */}
        <div className="chat-column">
          <div className="chat-box" ref={boxRef}>
            {messages.length === 0 ? (
              <div className="meta" style={{ textAlign: "center", marginTop: 24 }}>
                輸入問題，開始與知識場域對話
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>{msg.content}</div>
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
              <button type="submit" disabled={!canSend} style={{ alignSelf: "flex-end" }}>
                送出
              </button>
            </form>
            {error && <div className="status">錯誤：{error}</div>}
          </div>
        </div>

        {/* ══ GlobalBrainExplorer ══ */}
        <div className="brain-panel" ref={explorerColRef}>
          <div className="brain-panel-label">Global Knowledge Field</div>
          <GlobalBrainExplorer
            layout={globalLayout}
            activeCandidateEids={activeCandidateEids}
            onNodeClick={handleNodeClick}
            width={explorerSize.w}
            height={Math.max(explorerSize.h - 32, 280)}
          />
        </div>

        {/* ══ DetailPanel ══ */}
        <DetailPanel
          node={selectedNode}
          detail={detail}
          foundationTree={foundationTree}
          loadingTree={loadingTree}
          personalBrain={personalBrain}
          rcSemanticScores={rcSemanticScores}
          rcFoundationMap={rcFoundationMap}
          alpha={ALPHA}
          onAddFoundations={addFoundationsToBrain}
          onRemove={removeFromBrain}
          onUpdateWeight={updateWeight}
        />

        {/* ══ PersonalBrain ══ */}
        <div className="brain-panel" ref={personalColRef}>
          <div className="brain-panel-label">Personal Brain</div>
          <PersonalBrain
            layout={globalLayout}
            personalBrain={personalBrain}
            rcRanking={rcRanking}
            onConfirmRC={handleConfirmRC}
            fetchRCPaths={fetchRCPaths}
            width={personalSize.w}
            height={Math.max(personalSize.h - 32, 280)}
          />
        </div>

      </div>
    </main>
  );
}
