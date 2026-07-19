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

import type { BertopicLayoutData } from "./components/BertopicMap";

import SemanticPanel from "./components/SemanticPanel";

import type { SemanticContextData, RecommendedFoundation } from "./components/SemanticPanel";


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
  skill_used?:    string | null;
};

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY   = "kagrobot_phase";
const USER_ID_KEY = "kagrobot_user_id";
const MAX_HISTORY = 60;
const ALPHA       = 0.5;

// ── 打字機效能參數 ───────────────────────────────────────────
const TYPEWRITER_CHARS_PER_TICK = 3;
const TYPEWRITER_TICK_MS        = 20;

// ── Resize 防抖(毫秒)──────────────────────────────────────
const RESIZE_DEBOUNCE_MS = 200;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
const RC_RANKING_POOL_SIZE = 15;

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
    const interScore = n > 0 ? weightedSum / n : 0;
    const totalScore = alpha * semantic + (1 - alpha) * interScore;
    results.push({ eid: rcEid, label, totalScore });
  });

  return results.sort((a, b) => b.totalScore - a.totalScore).slice(0, 3);
}

// ─────────────────────────────────────────────────────────────
// usePanelSize(加入 resize 防抖)
// ─────────────────────────────────────────────────────────────

function usePanelSize(ref: React.RefObject<HTMLDivElement>) {
  const [size, setSize] = useState({ w: 400, h: 400 });
  useEffect(() => {
    if (!ref.current) return;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const ro = new ResizeObserver((entries) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          setSize({ w: Math.floor(width) || 400, h: Math.floor(height) || 400 });
        }
      }, RESIZE_DEBOUNCE_MS);
    });
    ro.observe(ref.current);

    return () => {
      if (timer) clearTimeout(timer);
      ro.disconnect();
    };
  }, [ref]);
  return size;
}

// ─────────────────────────────────────────────────────────────
// Semantic 機制 fetch helpers
// ─────────────────────────────────────────────────────────────

async function fetchSemanticContextData(sessionId: string): Promise<SemanticContextData | null> {
  if (!API_BASE || !sessionId) return null;
  try {
    const r = await fetch(
      `${API_BASE.replace(/\/$/, "")}/brain/semantic-context?session_id=${encodeURIComponent(sessionId)}`,
    );
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    return { query: d.query ?? "", intent_types: d.intent_types ?? [], summary: d.summary ?? "" };
  } catch (err) {
    console.warn("[semantic-context]", err);
    return null;
  }
}

async function fetchSemanticScores(sessionId: string): Promise<RCSemanticScore[]> {
  if (!API_BASE || !sessionId) return [];
  try {
    const r = await fetch(
      `${API_BASE.replace(/\/$/, "")}/brain/rc-semantic-scores?session_id=${encodeURIComponent(sessionId)}`,
    );
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    return d.scores ?? [];
  } catch (err) {
    console.warn("[rc-semantic-scores]", err);
    return [];
  }
}

async function fetchFoundationRecommendations(sessionId: string): Promise<RecommendedFoundation[]> {
  if (!API_BASE || !sessionId) return [];
  try {
    const r = await fetch(
      `${API_BASE.replace(/\/$/, "")}/brain/foundation-recommendations?session_id=${encodeURIComponent(sessionId)}`,
    );
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    return d.recommendations ?? [];
  } catch (err) {
    console.warn("[foundation-recommendations]", err);
    return [];
  }
}

async function fetchExpandBatch(
  eids: string[],
): Promise<Record<string, {
  anchor_eid: string;
  anchor_label: string;
  foundations: { element_id: string; label: string }[];
  nodes: { element_id: string; label: string }[];
  edges: { source: string; target: string }[];
}>> {
  if (!API_BASE || eids.length === 0) return {};
  try {
    const r = await fetch(`${API_BASE.replace(/\/$/, "")}/graph/expand/batch`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ eids }),
    });
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    return d.results ?? {};
  } catch (err) {
    console.warn("[expand/batch]", err);
    return {};
  }
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
  // ── BERTopic 2D layout ─────────────────────────────────────
  const [bertopicLayout, setBertopicLayout] = useState<BertopicLayoutData | null>(null);
  const [globalLayoutFetched, setGlobalLayoutFetched] = useState(false);

  // ── RC Intelligence ────────────────────────────────────────
  // rcFoundationMap 一律由 /graph/expand/batch 補齊(不再有 all_graphs 直接建構的分支)。
  const [rcSemanticScores, setRCSemanticScores] = useState<RCSemanticScore[]>([]);
  const [rcFoundationMap,  setRCFoundationMap]  = useState<RCFoundationMap>(new Map());
  const [semanticContext, setSemanticContext] = useState<SemanticContextData | null>(null);
  const [recommendedFoundations, setRecommendedFoundations] = useState<RecommendedFoundation[]>([]);

  // ── Personal Brain ─────────────────────────────────────────
  const [personalBrain, setPersonalBrain] = useState<PersonalBrainConcept[]>([]);
  const [brainLoaded,   setBrainLoaded]   = useState(false);

  // ── Detail Panel ───────────────────────────────────────────
  const [selectedNode,   setSelectedNode]   = useState<GlobalLayoutNode | null>(null);
  const [detail,         setDetail]         = useState<ConceptDetail | null>(null);
  const [foundationTree, setFoundationTree] = useState<FoundationTreeData | null>(null);
  const [loadingTree,    setLoadingTree]    = useState(false);

  // ── Refs ───────────────────────────────────────────────────
  const isTypingRef    = useRef(false);
  const messagesRef    = useRef<ChatMessage[]>([]);
  const boxRef         = useRef<HTMLDivElement | null>(null);
  const explorerColRef = useRef<HTMLDivElement>(null);
  const personalColRef = useRef<HTMLDivElement>(null);
  const explorerSize   = usePanelSize(explorerColRef as React.RefObject<HTMLDivElement>);
  const personalSize   = usePanelSize(personalColRef  as React.RefObject<HTMLDivElement>);

  // ── 避免同一個 eid 重複觸發 batch expand ──────────────────
  const expandFetchedRef = useRef<Set<string>>(new Set());

  // ── activeCandidateEids：不再依賴 chat response 的 all_graphs，
  //    直接從 rcSemanticScores 取 Top-3（GlobalBrainExplorer 點亮用）──
  const activeCandidateEids = useMemo(
    () => [...rcSemanticScores]
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((s) => s.eid),
    [rcSemanticScores],
  );

  // ── RC Ranking (derived, 前端即時計算) ────────────────────
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

  // ── Fetch BERTopic layout(一次性,和 globalLayout 同步觸發)──
  useEffect(() => {
    if (!API_BASE || bertopicLayout !== null) return;
    fetch(`${API_BASE.replace(/\/$/, "")}/graph/bertopic-layout`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: BertopicLayoutData) => setBertopicLayout(d))
      .catch((err) => console.warn("[BertopicLayout] fetch failed:", err));
  }, [bertopicLayout]);

  // ── RCFoundationMap 統一補齊：一律走 /graph/expand/batch，merge-only，
  //    不再有「先用 all_graphs 建、再補洞」的雙軌邏輯。
  //    wanted 集合 = 目前點亮的 Top-3 anchor ∪ semantic pool 前 15 名。──
  useEffect(() => {
    if (rcSemanticScores.length === 0 && activeCandidateEids.length === 0) return;

    const wanted = new Set<string>(activeCandidateEids);
    [...rcSemanticScores]
      .sort((a, b) => b.score - a.score)
      .slice(0, RC_RANKING_POOL_SIZE)
      .forEach((s) => wanted.add(s.eid));

    const missing = [...wanted].filter(
      (eid) => !rcFoundationMap.has(eid) && !expandFetchedRef.current.has(eid),
    );
    if (missing.length === 0) return;

    missing.forEach((eid) => expandFetchedRef.current.add(eid));

    fetchExpandBatch(missing).then((results) => {
      if (Object.keys(results).length === 0) return;
      setRCFoundationMap((prev) => {
        const next = new Map(prev);
        for (const [eid, r] of Object.entries(results)) {
          next.set(eid, {
            label:          r.anchor_label || eid,
            foundationEids: r.foundations.map((f) => f.element_id),
            treeNodes:      r.nodes,
            treeEdges:      r.edges,
          });
        }
        return next;
      });
    });
  }, [activeCandidateEids, rcSemanticScores, rcFoundationMap]);

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
  // 方案 B：「詢問這個」→ 把 eid 以人類可讀明文預填進輸入框，
  // 讓 socrates 在下一輪呼叫 retrieve_concept_detail 時能精確帶入 eid，
  // 不需要模糊比對 rc_name。
  // ──────────────────────────────────────────────────────────

  const handleAskAboutNode = useCallback((eid: string, label: string) => {
    setInput(`關於「${label}」(eid: ${eid})，`);
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

  const addFoundationsToBrain = useCallback((items: { eid: string; label: string }[]) => {
    setPersonalBrain((prev) => {
      const updated  = [...prev];
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

  const fetchConceptDetail = useCallback(async (eid: string): Promise<ConceptDetail | null> => {
    if (!API_BASE) return null;
    try {
      const r = await fetch(`${API_BASE.replace(/\/$/, "")}/concept/detail/${eid}`);
      if (!r.ok) throw new Error(`${r.status}`);
      return await r.json();
    } catch (err) {
      console.warn("[concept/detail]", err);
      return null;
    }
  }, []);

  // ──────────────────────────────────────────────────────────
  // PersonalBrain: fetch foundation tree(垂直剖面圖用)
  // ──────────────────────────────────────────────────────────

  const fetchFoundationTree = useCallback(async (eid: string): Promise<FoundationTreeData | null> => {
    if (!API_BASE) return null;
    try {
      const r = await fetch(`${API_BASE.replace(/\/$/, "")}/graph/expand/${eid}`);
      if (!r.ok) throw new Error(`${r.status}`);
      return await r.json();
    } catch (err) {
      console.warn("[graph/expand]", err);
      return null;
    }
  }, []);

  // ──────────────────────────────────────────────────────────
  // PersonalBrain: confirm RC → MainAgent
  // ──────────────────────────────────────────────────────────

  const handleConfirmRC = useCallback(async (
    rcEid: string, rcLabel: string, pathId: number,
  ) => {
    if (!sessionId || isLoading || !API_BASE) return;

    addFoundationsToBrain([{ eid: rcEid, label: rcLabel }]);

    const confirmMsg: ChatMessage = {
      id: createId(), role: "user",
      content: `確認選擇路徑:${rcLabel}`,
    };
    setMessages((prev) => [...prev, confirmMsg]);
    setIsLoading(true);
    setError("");

    try {
      const resp = await fetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:        sessionId,
          message:           confirmMsg.content,
          phase:             phase === "main_agent" ? "main_agent" : "socratic",
          selected_path_id:  pathId,
          selected_rc_eid:   rcEid,
          selected_rc_label: rcLabel,
        }),
      });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = (await resp.json()) as ChatResponse;
      setPhase(data.phase || "main_agent");
      localStorage.setItem(PHASE_KEY, data.phase || "main_agent");
      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗,請稍後重試");
      setMessages((prev) => [...prev, { id: createId(), role: "ai", content: "抱歉,連線失敗。" }]);
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

    let i = TYPEWRITER_CHARS_PER_TICK;
    for (; i <= fullText.length; i += TYPEWRITER_CHARS_PER_TICK) {
      setMessages((prev) =>
        prev.map((msg) => msg.id === aiId ? { ...msg, content: fullText.slice(0, i) } : msg),
      );
      await wait(TYPEWRITER_TICK_MS);
    }
    setMessages((prev) =>
      prev.map((msg) => msg.id === aiId ? { ...msg, content: fullText } : msg),
    );

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
          phase:      phase === "main_agent" ? "main_agent" : "socratic",
        }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);

      const data      = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "socratic";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);

      // ── Semantic 機制：每輪都刷新。GlobalBrain 點亮 / RCFoundationMap 補齊
      //    / SemanticPanel 顯示，全部由這三支 fetch 驅動，不再依賴 chat
      //    response 裡的任何圖譜欄位。──
      fetchSemanticContextData(sessionId).then(setSemanticContext);
      fetchSemanticScores(sessionId).then(setRCSemanticScores);
      fetchFoundationRecommendations(sessionId).then(setRecommendedFoundations);

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗,請稍後重試");
      setMessages((prev) => [...prev, { id: createId(), role: "ai", content: "抱歉,連線失敗,請稍後再試。" }]);
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
    setRCFoundationMap(new Map());
    setRCSemanticScores([]);
    expandFetchedRef.current = new Set();
    setSelectedNode(null);
    setDetail(null);
    setFoundationTree(null);
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
    setSemanticContext(null);
    setRecommendedFoundations([]);
  }

  const inputPlaceholder =
    phase === "main_agent" ? "繼續詢問分析…" : "輸入你的問題…";

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

      {/* ── Four-column layout ── */}
      <div className="content-area">

        {/* ══ Chat ══ */}
        <div className="chat-column">
          <div className="chat-box" ref={boxRef}>
            {messages.length === 0 ? (
              <div className="meta" style={{ textAlign: "center", marginTop: 24 }}>
                輸入問題,開始與知識場域對話
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
            {error && <div className="status">錯誤:{error}</div>}
          </div>
        </div>

        {/* ══ GlobalBrainExplorer ══ */}
        <div className="brain-panel" ref={explorerColRef}>
          <div className="brain-panel-label">Global Knowledge Field</div>
          <GlobalBrainExplorer
            layout={globalLayout}
            activeCandidateEids={activeCandidateEids}
            rcFoundationMap={rcFoundationMap}
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
          fetchConceptDetail={fetchConceptDetail}
          onAddFoundations={addFoundationsToBrain}
          onRemove={removeFromBrain}
          onUpdateWeight={updateWeight}
          onAskAboutNode={handleAskAboutNode}
        />

        <SemanticPanel
          context={semanticContext}
          rcScores={rcSemanticScores}
          recommendedFoundations={recommendedFoundations}
          onAddFoundation={(eid, label) => addFoundationsToBrain([{ eid, label }])}
        />

        {/* ══ PersonalBrain ══ */}
        <div className="brain-panel" ref={personalColRef}>
          <div className="brain-panel-label">Personal Brain</div>
          <PersonalBrain
            layout={globalLayout}
            personalBrain={personalBrain}
            rcRanking={rcRanking}
            bertopicLayout={bertopicLayout}
            fetchFoundationTree={fetchFoundationTree}
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