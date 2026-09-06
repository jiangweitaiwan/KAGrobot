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

import { AttachmentCard, AttachmentTray, type Attachment } from "./components/AttachmentCard";



import { useAuth } from "./components/AuthProvider";
import AuthPanel from "./components/AuthPanel";
import { apiFetch } from "../lib/apiFetch";
import { streamSSE } from "../lib/sseFetch"; 
import  {
  type WalkPathState,
  type FrontierPreviewItem,
  type WalkOverallStatus,
  type WalkQAEntry,
} from "./components/walkTypes";
import WalkDock from "./components/WalkDock";

const GlobalBrainExplorer = dynamic(() => import("./components/GlobalBrainExplorer"), { ssr: false });
const PersonalBrain       = dynamic(() => import("./components/PersonalBrain"),       { ssr: false });
const DetailPanel         = dynamic(() => import("./components/DetailPanel"),         { ssr: false });
const WALK_PATH_COLORS = ["#00d4aa", "#7f77dd", "#e8a030"]; // teal / purple / amber
// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

type Role = "user" | "ai";
type ChatMessage = { id: string; role: Role; content: string; attachments?: Attachment[] };

type ChatResponse = {
  reply:          string;
  attachments?:   Attachment[];
  phase:          string;
  socrates_phase?: string | null;
  skill_used?:    string | null;
};

type SessionSummary = {
  session_id: string;
  phase?:     string | null;
  summary:    string;
  updated_at?: number | null;
  created_at?: number | null;
};

type SessionDisplayMessage = {
  role:        string;
  content:     string;
  attachments: Attachment[];
  created_at:  number;
};

type SessionDetail = {
  session_id:             string;
  phase:                  string;
  display_messages:       SessionDisplayMessage[];
  selected_rc_eid?:       string | null;
  selected_rc_label?:     string | null;
  walk_session?:          WalkSessionDto | null;
  walk_pending_question?: { path_id: number; question: string } | null;
};

type WalkStatusResponse = {
  phase: string;
  has_pending_question: boolean;
  pending_question: { path_id: number; question: string } | null;
};
type WalkNodeRefDto = { eid: string; label: string; is_rc: boolean };
type FrontierCandidateDto = { eid: string; label: string; definition?: string | null; mechanism?: string | null; node_type?: string | null };
type DecisionLogEntryDto = { from_eid: string; to_eid: string; reason: string; qa?: { question: string; answer: string } | null };

type PathStateDto = {
  path_id: number;
  start_eid: string;
  start_label: string;
  current_eid: string;
  current_label: string;
  visited: WalkNodeRefDto[];
  frontier: FrontierCandidateDto[];
  status: "pending" | "active" | "dead_end";
  dead_end_reason: string | null;
  depth: number;
  max_depth: number;
  decision_log: DecisionLogEntryDto[];
  pending_qa: { question: string; answer: string } | null;
};

type WalkSessionDto = {
  session_id: string;
  paths: PathStateDto[];
  max_depth: number;
};


// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

// 注意：使用者身分（uid）不再由本檔案自行產生／存進 localStorage，
// 一律來自 Firebase Auth（見 AuthProvider）——匿名使用者也有真正的
// Firebase uid，登入/升級帳號時 uid 不變，資料自動延續。
const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY   = "kagrobot_phase";
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
// Attachment placeholder helpers
// ─────────────────────────────────────────────────────────────

// 對應後端 get_sandbox_file 要求 LLM 插入的 [[attachment:N]] 佔位符。
const ATTACHMENT_TOKEN_RE = /\[\[attachment:(\d+)\]\]/g;

type ContentPart =
  | { type: "text"; value: string }
  | { type: "ref"; ref: string };

/** 把訊息文字依 [[attachment:N]] 佔位符切成文字段落與附件參照段落。 */
function splitByPlaceholder(content: string): ContentPart[] {
  const parts: ContentPart[] = [];
  let lastIndex = 0;
  ATTACHMENT_TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ATTACHMENT_TOKEN_RE.exec(content)) !== null) {
    if (m.index > lastIndex) parts.push({ type: "text", value: content.slice(lastIndex, m.index) });
    parts.push({ type: "ref", ref: m[1] });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < content.length) parts.push({ type: "text", value: content.slice(lastIndex) });
  return parts;
}

/**
 * 打字機逐字顯示時，若切點剛好落在 [[attachment:N]] 佔位符中間，
 * 直接跳到該 token 結尾，避免使用者看到一閃而過的 "[[attachm" 這種
 * 未完成的原始標記文字。
 */
function revealUpTo(fullText: string, i: number): string {
  ATTACHMENT_TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ATTACHMENT_TOKEN_RE.exec(fullText)) !== null) {
    const start = m.index;
    const end   = start + m[0].length;
    if (i > start && i < end) return fullText.slice(0, end);
    if (i <= start) break;
  }
  return fullText.slice(0, i);
}

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

function walkSessionDtoToPathStates(dto: WalkSessionDto): Record<number, WalkPathState> {
  const out: Record<number, WalkPathState> = {};
  for (const p of dto.paths) {
    out[p.path_id] = {
      pathId: p.path_id,
      visited: p.visited.map((n) => ({ eid: n.eid, label: n.label, isRc: n.is_rc })),
      currentEid: p.current_eid,
      currentLabel: p.current_label,
      status: p.status,
      deadEndReason: p.dead_end_reason ?? undefined,
      // frontier / frontierHistory 只有「即時走動」時才有意義的分支預覽，
      // 重播模式下沒有 live interrupt 可互動，一律留空，只呈現主幹軌跡。
      frontierPreview: [],
      frontierHistory: [],
    };
  }
  return out;
}

function walkSessionDtoToQaLog(dto: WalkSessionDto): WalkQAEntry[] {
  const out: WalkQAEntry[] = [];
  for (const p of dto.paths) {
    for (const entry of p.decision_log) {
      if (entry.qa) {
        out.push({ pathId: p.path_id, question: entry.qa.question, answer: entry.qa.answer });
      }
    }
  }
  return out;
}

/** eid → {category, color} 查表，供 DetailPanel 標題列 / PersonalBrain
 *  Top-3 推薦標籤 / HourglassViz waist 標籤共用。資料來源是
 *  globalLayout.nodes 上的 category / category_color 欄位
 *  （由後端 /graph/global-layout 合併 GCS rc_categories.json 產生），
 *  不需要額外 fetch。*/
function buildRCCategoryMap(
    layout: GlobalLayout | null,
  ): Map<string, { category: string; color?: string | null }> {
    const m = new Map<string, { category: string; color?: string | null }>();
    layout?.nodes.forEach((n) => {
      if (n.category) m.set(n.id, { category: n.category, color: n.category_color });
    });
    return m;
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
// Semantic 機制 fetch helpers（一律透過 apiFetch 帶上登入憑證）
// ─────────────────────────────────────────────────────────────

async function fetchSemanticContextData(sessionId: string): Promise<SemanticContextData | null> {
  if (!API_BASE || !sessionId) return null;
  try {
    const r = await apiFetch(
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
    const r = await apiFetch(
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
    const r = await apiFetch(
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
    const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/graph/expand/batch`, {
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

async function fetchMySessions(): Promise<SessionSummary[]> {
  if (!API_BASE) return [];
  try {
    const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/me/sessions`);
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    return d.sessions ?? [];
  } catch (err) {
    console.warn("[me/sessions]", err);
    return [];
  }
}

async function fetchSessionDetail(sessionId: string): Promise<SessionDetail | null> {
  if (!API_BASE) return null;
  try {
    const r = await apiFetch(
      `${API_BASE.replace(/\/$/, "")}/me/sessions/${encodeURIComponent(sessionId)}`,
    );
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch (err) {
    console.warn("[me/sessions/detail]", err);
    return null;
  }
}


// ─────────────────────────────────────────────────────────────
// Random Walk（慢想）fetch helpers
// ─────────────────────────────────────────────────────────────

async function fetchWalkStatus(sessionId: string): Promise<WalkStatusResponse | null> {
  if (!API_BASE || !sessionId) return null;
  try {
    const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/walk/status/${sessionId}`);
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch (err) {
    console.warn("[walk/status]", err);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────
// ChatBubble — 拆解 [[attachment:N]] 佔位符為附件卡片
// ─────────────────────────────────────────────────────────────

function ChatBubble({ msg }: { msg: ChatMessage }) {
  const attachments = msg.attachments ?? [];
  const parts        = useMemo(() => splitByPlaceholder(msg.content), [msg.content]);
  const referencedRefs = useMemo(
    () => new Set(parts.filter((p): p is Extract<ContentPart, { type: "ref" }> => p.type === "ref").map((p) => p.ref)),
    [parts],
  );
  // 保底：若模型忘記插入佔位符（attachments 非空但文字裡一個都沒引用），
  // 仍把所有附件附加在回覆最後面，不讓產出的圖片/檔案憑空消失。
  const unreferenced = attachments.filter((a) => !referencedRefs.has(a.ref));

  return (
    <div className={`msg ${msg.role}`}>
      {parts.map((p, i) => {
        if (p.type === "text") {
          return p.value ? <span key={i}>{p.value}</span> : null;
        }
        const att = attachments.find((a) => a.ref === p.ref);
        return att ? (
          <span key={i} style={{ display: "inline-block", verticalAlign: "top" }}>
            <AttachmentCard attachment={att} />
          </span>
        ) : null;
      })}
      {unreferenced.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", marginTop: 6 }}>
          {unreferenced.map((a) => (
            <AttachmentCard key={a.ref} attachment={a} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────

export default function Page() {
  // ── Identity（uid 來自 Firebase Auth，見 AuthProvider）──────
  const { user, loading: authLoading } = useAuth();
  const userId = user?.uid ?? "";

  const [sessionId, setSessionId] = useState("");
  const [phase,     setPhase]     = useState("socratic");

  // ── Chat ───────────────────────────────────────────────────
  const [messages,  setMessages]  = useState<ChatMessage[]>([]);
  const [input,     setInput]     = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error,     setError]     = useState("");

  const [historySessions, setHistorySessions] = useState<SessionSummary[]>([]);
  const [showHistory,     setShowHistory]     = useState(false);
  const [loadingHistory,  setLoadingHistory]  = useState(false);


  // ── Random Walk（慢想）Mode ───────────────────────────────────
 
  const [walkEnabled, setWalkEnabled]     = useState(false);
  const [walkPaths, setWalkPaths] = useState<Record<number, WalkPathState>>({});
  const [walkQaLog, setWalkQaLog] = useState<WalkQAEntry[]>([]);
  const [pendingWalkQuestion, setPendingWalkQuestion] = useState<{ pathId: number; question: string } | null>(null);
  const [walkStreaming, setWalkStreaming]   = useState(false);
  const [pendingWalkPathId, setPendingWalkPathId] = useState<number>(0);
  const [walkRankingJustUpdated, setWalkRankingJustUpdated] = useState(false);

    // walk 結果是否仍應顯示（獨立於 walkStreaming，由使用者主動關閉才清空）
  const [walkResultAvailable, setWalkResultAvailable] = useState(false);
  const [walkDockCollapsed,  setWalkDockCollapsed]    = useState(false);
  const [brainModeOn,        setBrainModeOn]          = useState(false);
  const [walkReadOnly,       setWalkReadOnly]         = useState(false);


  const walkPathColorMap = useMemo(() => {
    const m = new Map<number, string>();
    Object.keys(walkPaths).forEach((k) => {
      const pid = Number(k);
      m.set(pid, WALK_PATH_COLORS[pid % WALK_PATH_COLORS.length]);
    });
    return m;
  }, [walkPaths]);


  // walk 路徑的視覺化不再受「WalkDock 是否還開著」影響：只要 walkPaths 有資料
  // 就一直畫在 GlobalBrain 上，直到使用者按 WalkDock 的 ✕、開新對話、或載入
  // 別的歷史對話。是否顯示 WalkDock 面板本身則另外用 walkResultAvailable 等
  // 既有旗標控制，兩者徹底脫鉤。
  const walkIllumination = useMemo(() => {
    if (Object.keys(walkPaths).length === 0) return undefined;
    return Object.values(walkPaths).map((p) => ({
      pathId: p.pathId,
      color: walkPathColorMap.get(p.pathId) ?? WALK_PATH_COLORS[0],
      visitedEids: p.visited.map((n) => n.eid),
      frontierPreview: p.frontierPreview.map((c) => ({ eid: c.eid, isRc: c.is_rc })),
      frontierHistory: p.frontierHistory.map((step) => ({
        fromEid: step.fromEid,
        candidates: step.candidates.map((c) => ({ eid: c.eid, isRc: c.is_rc })),
      })),
      status: p.status,
    }));
  }, [walkPaths, walkPathColorMap]);

  const walkOverallStatus: WalkOverallStatus =
    phase === "walk_awaiting_answer" ? "awaiting_answer" :
    walkStreaming                     ? "walking" :
    "connecting";

  // 每次開新的 walk SSE 連線前，先中止上一條，避免同時存在多條
  // reader 造成事件重複處理；元件卸載時一併中止，避免背景無人消費
  // 的 stream 持續佔用連線。
  const walkAbortRef = useRef<AbortController | null>(null);  


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
  // ── 是否有任何可高亮的節點（語意推薦候選 or 慢想路徑）。
  //    沒有的話「切換背景」按鈕要 disable，避免使用者切換後畫面整片灰。  
  const hasHighlightableNodes = useMemo(() => {
    const anyCandidates = activeCandidateEids.length > 0;
    const anyWalk = Object.values(walkPaths).some((p) => p.visited.length > 0);
    return anyCandidates || anyWalk;
  }, [activeCandidateEids, walkPaths]);

  // 激活節點消失（例如換了對話、走完的路徑被清除）時，自動退回全域視圖。
  useEffect(() => {
    if (!hasHighlightableNodes && brainModeOn) setBrainModeOn(false);
  }, [hasHighlightableNodes, brainModeOn]);

  // ── GlobalBrainExplorer 實際要點亮的節點集合：random walk modal 開著
  //    時，優先顯示 walk 即時走訪的節點（大腦被激活的視覺效果）；
  //    否則沿用 dialogue mode 既有的語意排名 Top-3。modal 關閉時
  //    RandomWalkPanel 會被 unmount，SSE 連線隨之中止（後端 walk 狀態
  //    本身透過 orchestrator checkpointer 安全保存、不受影響，只是
  //    前端暫時看不到即時進度），因此這裡的判斷同時也隱含「只有
  //    modal 開著才顯示 walk 高亮」。


  // ── RC Ranking (derived, 前端即時計算) ────────────────────
  const rcRanking = useMemo(
    () => computeRCRanking(personalBrain, rcSemanticScores, rcFoundationMap, ALPHA),
    [personalBrain, rcSemanticScores, rcFoundationMap],
  );
  // ── RC 分類標籤查表（GlobalBrain hover / DetailPanel / PersonalBrain
  //    推薦標籤 / HourglassViz waist 共用，見 buildRCCategoryMap）──────
  const rcCategoryMap = useMemo(
    () => buildRCCategoryMap(globalLayout),
    [globalLayout],
  );

  // ── 本次 session 至今所有附件（供橫向 tray 彙總顯示）───────
  const allAttachments = useMemo(
    () => messages.flatMap((m) => m.attachments ?? []),
    [messages],
  );

  // ──────────────────────────────────────────────────────────
  // Init from localStorage（session/phase/history 仍是本機裝置層級的
  // 「目前開著哪個對話」設定，與登入身分無關；身分本身完全交給
  // AuthProvider／Firebase 管理，這裡不再產生或讀寫任何 uid）
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const storedSession = localStorage.getItem(SESSION_KEY);
    const sid           = storedSession || createId();
    if (!storedSession) localStorage.setItem(SESSION_KEY, sid);
    setSessionId(sid);

    const storedPhase = localStorage.getItem(PHASE_KEY) || "socratic";
    setPhase(storedPhase);
    const rawHistory = localStorage.getItem(HISTORY_KEY);
    setMessages(rawHistory ? JSON.parse(rawHistory) : []);

    // 若上次關閉頁面時卡在 walk_pending，等 userId 就緒後再檢查
    // （見下方另一個 effect，因為 recoverWalkIfNeeded 需要已登入）。
  }, []);

  // ── walk 恢復檢查：需要 userId（apiFetch 才帶得出 Authorization
  //    header）與 sessionId 都就緒後才能呼叫 ──────────────────────
  useEffect(() => {
    if (!userId || !sessionId) return;
    const storedPhase = localStorage.getItem(PHASE_KEY) || "socratic";
    if (storedPhase === "walk_pending") {
      recoverWalkIfNeeded(sessionId, storedPhase);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, sessionId]);

  // ── unmount 時中止任何仍在進行的 walk SSE 連線 ──────────────────
  useEffect(() => {
    return () => { walkAbortRef.current?.abort(); };
  }, []);

  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, isLoading]);

  // ── Load Personal Brain（等待 Firebase 給出 uid 後才觸發，
  //    匿名使用者一樣有 uid，不會因為還沒登入正式帳號而卡住）──
  useEffect(() => {
    if (!userId || !API_BASE || brainLoaded) return;
    setBrainLoaded(true);
    apiFetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => { if (d.concepts?.length) setPersonalBrain(d.concepts); })
      .catch((err) => console.warn("[PersonalBrain] load failed:", err));
  }, [userId, brainLoaded]);

  // ── Fetch global layout ────────────────────────────────────
  useEffect(() => {
    if (messages.length === 0 || globalLayoutFetched || !API_BASE || !userId) return;
    setGlobalLayoutFetched(true);
    apiFetch(`${API_BASE.replace(/\/$/, "")}/graph/global-layout`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: GlobalLayout) => setGlobalLayout(d))
      .catch((err) => console.warn("[GlobalBrain] layout fetch failed:", err));
  }, [messages, globalLayoutFetched, userId]);

  // ── Fetch BERTopic layout(一次性,和 globalLayout 同步觸發)──
  useEffect(() => {
    if (!API_BASE || !userId || bertopicLayout !== null) return;
    apiFetch(`${API_BASE.replace(/\/$/, "")}/graph/bertopic-layout`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: BertopicLayoutData) => setBertopicLayout(d))
      .catch((err) => console.warn("[BertopicLayout] fetch failed:", err));
  }, [bertopicLayout, userId]);



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
      apiFetch(`${base}/concept/detail/${node.id}`).then((r) => r.json() as Promise<ConceptDetail>),
      apiFetch(`${base}/graph/expand/${node.id}`).then((r) => r.json() as Promise<FoundationTreeData>),
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
  // Random Walk（慢想）：SSE 事件消費與狀態機
  // ──────────────────────────────────────────────────────────
  const upsertWalkPath = useCallback((pathId: number, patch: Partial<WalkPathState>) => {
    setWalkPaths((prev) => {
      const existing = prev[pathId] ?? {
        pathId, visited: [], currentEid: "", currentLabel: "",
        status: "pending" as const, frontierPreview: [], frontierHistory: [],
      };
      return { ...prev, [pathId]: { ...existing, ...patch } };
    });
  }, []);
  
  const handleWalkEvent = useCallback((event: any) => {
    switch (event.type) {
      case "walk_started": {
        const paths: Record<number, WalkPathState> = {};
        (event.start_concepts ?? []).forEach((c: any) => {
          paths[c.path_id] = {
            pathId: c.path_id,
            visited: [{ eid: c.eid, label: c.label, isRc: false }],
            currentEid: c.eid, currentLabel: c.label,
            status: "pending", frontierPreview: [], frontierHistory: [],
          };
        });
        setWalkPaths(paths);
        setWalkQaLog([]);
        setWalkResultAvailable(false);
        setWalkDockCollapsed(false);
        setWalkReadOnly(false); 
        break;
      }
      case "frontier_ready": {
        upsertWalkPath(event.path_id, {
          status: "active",
          frontierPreview: (event.candidates ?? []) as FrontierPreviewItem[],
        });
        break;
      }
      case "frontier_expanded": {
        setWalkPaths((prev) => {
          const existing = prev[event.path_id];
          if (!existing) return prev;
  
          const chosen = existing.frontierPreview.find((c) => c.eid === event.to_eid);
          const unchosen = existing.frontierPreview.filter((c) => c.eid !== event.to_eid);
  
          const frontierHistory = unchosen.length > 0
            ? [...existing.frontierHistory, { fromEid: event.from_eid, candidates: unchosen }]
            : existing.frontierHistory;
  
          return {
            ...prev,
            [event.path_id]: {
              ...existing,
              visited: [...existing.visited, {
                eid: event.to_eid, label: event.to_label, isRc: chosen?.is_rc ?? false,
              }],
              currentEid: event.to_eid,
              currentLabel: event.to_label,
              status: "pending",
              frontierPreview: [],
              frontierHistory,
            },
          };
        });
        break;
      }
      case "path_dead_end": {
        upsertWalkPath(event.path_id, {
          status: "dead_end", deadEndReason: event.reason, frontierPreview: [],
        });
        break;
      }
      case "user_question": {
        const pathId = event.path_id ?? 0;
        setPendingWalkPathId(pathId);
        setPendingWalkQuestion({ pathId, question: event.question });
        setWalkQaLog((prev) => [...prev, { pathId, question: event.question }]);
        setPhase("walk_awaiting_answer");
        setWalkDockCollapsed(false);   // never let a question hide behind a collapsed dock
        break;
      }
      case "walk_complete": {
        setPhase("socratic");
        localStorage.setItem(PHASE_KEY, "socratic");
        setWalkStreaming(false);
        setWalkResultAvailable(true);
        setWalkDockCollapsed(false);
        setPendingWalkQuestion(null);
        typewriterAppend(event.summary || "慢想完成。", []);
        fetchSemanticContextData(sessionId).then(setSemanticContext);
        fetchSemanticScores(sessionId).then((scores) => {
          setRCSemanticScores(scores);
          setWalkRankingJustUpdated(true);
          setTimeout(() => setWalkRankingJustUpdated(false), 2200);
        });
        fetchFoundationRecommendations(sessionId).then(setRecommendedFoundations);
        break;
      }
      case "walk_aborted": {
        setPhase("socratic");
        localStorage.setItem(PHASE_KEY, "socratic");
        setWalkStreaming(false);
        setWalkPaths({});
        setWalkQaLog([]);
        setWalkResultAvailable(false);
        setPendingWalkQuestion(null);
        typewriterAppend(event.message || "慢想未能開始，我們繼續聊聊。", []);
        break;
      }
      case "walk_error": {
        setWalkStreaming(false);
        setError(event.message || "慢想過程發生錯誤");
        break;
      }
      default:
        console.warn("[walk] unknown event type:", event.type);
    }
  }, [sessionId, upsertWalkPath]); 

  /** 開一段 /walk/stream，驅動後端真正跑 walk 迴圈。呼叫前一律先中止
   *  上一條連線（若還存在），避免重複消費。*/
  const runWalkStream = useCallback(async () => {
    if (!API_BASE || !sessionId) return;
    walkAbortRef.current?.abort();
    const controller = new AbortController();
    walkAbortRef.current = controller;

    setWalkStreaming(true);
    try {
      await streamSSE(
        `${API_BASE.replace(/\/$/, "")}/walk/stream/${sessionId}`,
        handleWalkEvent,
        { signal: controller.signal },
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        console.warn("[walk/stream]", err);
        setError(err instanceof Error ? err.message : "慢想連線失敗");
      }
    } finally {
      if (walkAbortRef.current === controller) setWalkStreaming(false);
    }
  }, [sessionId, handleWalkEvent]);

  /** 回答 walk 提出的問題：POST /walk/answer，回應本身就是新一段
   *  SSE stream，沿用同一套事件處理。*/
  const submitWalkAnswer = useCallback(async (answerText: string) => {
    if (!API_BASE || !sessionId) return;
    walkAbortRef.current?.abort();
    const controller = new AbortController();
    walkAbortRef.current = controller;
  
    const answeredPathId = pendingWalkPathId;
    const answeredQuestion = pendingWalkQuestion?.question ?? "";
    setWalkQaLog((prev) => {
      const idx = [...prev].reverse().findIndex(
        (q) => q.pathId === answeredPathId && q.question === answeredQuestion && q.answer === undefined,
      );
      if (idx === -1) return prev;
      const realIdx = prev.length - 1 - idx;
      const next = [...prev];
      next[realIdx] = { ...next[realIdx], answer: answerText };
      return next;
    });
    setPendingWalkQuestion(null);
    setWalkStreaming(true);
    try {
      await streamSSE(
        `${API_BASE.replace(/\/$/, "")}/walk/answer/${sessionId}`,
        handleWalkEvent,
        {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path_id: answeredPathId, answer: answerText }),
          signal: controller.signal,
        },
      );
      setPhase((cur) => (cur === "walk_awaiting_answer" ? "walk_pending" : cur));
    } catch (err) {
      if (!controller.signal.aborted) {
        console.warn("[walk/answer]", err);
        setError(err instanceof Error ? err.message : "回答提交失敗");
      }
    } finally {
      if (walkAbortRef.current === controller) setWalkStreaming(false);
    }
  }, [sessionId, pendingWalkPathId, pendingWalkQuestion, handleWalkEvent]);

  /** 頁面重新整理後（非冷啟動）的恢復：若本機記得上次 phase 是
   *  walk_pending，先問後端目前真實狀態，再決定：
   *    - 有 pending question → 直接呈現問題，等待回答（不重新送 SSE
   *      input，避免干擾正在暫停的 interrupt）。
   *    - 沒有 pending question 但仍是 walk_pending → 安全地重新接上
   *      SSE 繼續消費事件。
   *    - 後端已不是 walk_pending（多半是冷啟動後被 /chat 的降級邏輯
   *      打回 socratic）→ 校正本機 phase，放棄恢復。
   */
  const recoverWalkIfNeeded = useCallback(async (sid: string, localPhase: string) => {
    if (localPhase !== "walk_pending" || !sid) return;
    const status = await fetchWalkStatus(sid);
    if (!status) return;

    if (status.phase !== "walk_pending") {
      setPhase("socratic");
      localStorage.setItem(PHASE_KEY, "socratic");
      return;
    }
    if (status.has_pending_question && status.pending_question) {
      const pathId = status.pending_question.path_id ?? 0;
      setPendingWalkPathId(pathId);
      setPendingWalkQuestion({ pathId, question: status.pending_question.question });
      setPhase("walk_awaiting_answer");
      return;
    }
    runWalkStream();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ──────────────────────────────────────────────────────────
  // Personal Brain CRUD
  // ──────────────────────────────────────────────────────────

  const persistBrainEntry = useCallback((entry: PersonalBrainConcept) => {
    if (!API_BASE || !userId) return;
    apiFetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts`, {
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
    apiFetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts/${eid}`, { method: "DELETE" })
      .catch((err) => console.warn("[brain/remove]", err));
  }, [userId]);

  const updateWeight = useCallback((eid: string, weight: number) => {
    setPersonalBrain((prev) => prev.map((c) => c.eid === eid ? { ...c, weight } : c));
    if (!API_BASE || !userId) return;
    apiFetch(`${API_BASE.replace(/\/$/, "")}/user/${userId}/brain/concepts/${eid}`, {
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
      const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/graph/rc-paths/${eid}`);
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
      const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/concept/detail/${eid}`);
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
      const r = await apiFetch(`${API_BASE.replace(/\/$/, "")}/graph/expand/${eid}`);
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

  const handleLaunchSkill = useCallback(async (
    rcEid: string, rcLabel: string, sessionIdOverride?: string,
  ) => {
    // sessionIdOverride 只有在「從 random walk 排名結果選定最終 RC」
    // 時才會有值（見 handleSelectFinalRC），此時它就是 walkId 本身；
    // 一般 dialogue mode（PersonalBrain 的 Top-3 標籤點擊）不帶這個
    // 參數，行為與改動前完全一致。
    const targetSessionId = sessionIdOverride ?? sessionId;
    if (!targetSessionId || isLoading || !API_BASE || !userId) return;

    addFoundationsToBrain([{ eid: rcEid, label: rcLabel }]);

    const launchMsg: ChatMessage = {
      id: createId(), role: "user",
      content: `執行 skill：${rcLabel}`,
    };
    setMessages((prev) => [...prev, launchMsg]);
    setIsLoading(true);
    setError("");

    try {
      const resp = await apiFetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:        targetSessionId,
          message:           launchMsg.content,
          phase:             phase === "main_agent" ? "main_agent" : "socratic",
          selected_rc_eid:   rcEid,
          selected_rc_label: rcLabel,
        }),
      });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = (await resp.json()) as ChatResponse;

      // 從 random walk 交接過來時，walk_id 從此變成這段對話往後的
      // session_id——orchestrator 那邊的 thread_id 本來就是同一個
      // "{uid}:{walk_id}" 字串（見 api/routes/routes_walk.py 與
      // api/routes/chat.py 共用的 thread_id 慣例），所以這裡只需要
      // 把本機狀態同步過去，完全不需要後端另外提供交接用的 endpoint。
      if (sessionIdOverride) {
        setSessionId(sessionIdOverride);
        localStorage.setItem(SESSION_KEY, sessionIdOverride);
      }

      setPhase(data.phase || "main_agent");
      localStorage.setItem(PHASE_KEY, data.phase || "main_agent");
      await typewriterAppend(data.reply || "", data.attachments ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗,請稍後重試");
      setMessages((prev) => [...prev, { id: createId(), role: "ai", content: "抱歉,連線失敗。" }]);
    } finally {
      setIsLoading(false);
      if (!isTypingRef.current) saveHistory(messagesRef.current);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, isLoading, phase, userId, addFoundationsToBrain]);



  // ──────────────────────────────────────────────────────────
  // Chat
  // ──────────────────────────────────────────────────────────

  const canSend = !!input.trim() && !isLoading && !!sessionId && !!userId
  && !walkStreaming && phase !== "walk_awaiting_answer";

  async function typewriterAppend(fullText: string, attachments: Attachment[] = []): Promise<void> {
    isTypingRef.current = true;
    const aiId = createId();
    // attachments 一次性掛上（不隨打字機效果變化），文字則逐字揭露；
    // revealUpTo 確保揭露切點不會落在 [[attachment:N]] 佔位符中間，
    // 避免使用者看到一閃而過的原始標記文字。
    setMessages((prev) => [...prev, { id: aiId, role: "ai", content: "", attachments }]);

    let i = TYPEWRITER_CHARS_PER_TICK;
    for (; i <= fullText.length; i += TYPEWRITER_CHARS_PER_TICK) {
      const revealed = revealUpTo(fullText, i);
      setMessages((prev) =>
        prev.map((msg) => msg.id === aiId ? { ...msg, content: revealed } : msg),
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
      const response = await apiFetch(`${API_BASE.replace(/\/$/, "")}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:   sessionId,
          message:      text,
          phase:        phase === "main_agent" ? "main_agent" : "socratic",
          walk_enabled: walkEnabled,
        }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);

      const data      = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "socratic";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);

      await typewriterAppend(data.reply || "", data.attachments ?? []);

      if (nextPhase === "walk_pending") {
        // 慢想開始：這輪 /chat 只負責把 phase 轉場，真正的 walk 迴圈
        // 由這裡立刻接上的 SSE 驅動。
        runWalkStream();
      } else {
        fetchSemanticContextData(sessionId).then(setSemanticContext);
        fetchSemanticScores(sessionId).then(setRCSemanticScores);
        fetchFoundationRecommendations(sessionId).then(setRecommendedFoundations);
      }
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
    walkAbortRef.current?.abort();

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

    // 新對話與舊的慢想進度沒有關聯，一併清空。
    setWalkPaths({});
    setPendingWalkQuestion(null);
    setWalkStreaming(false);
    setPendingWalkPathId(0);
    setWalkResultAvailable(false);
    setWalkDockCollapsed(false);
    setBrainModeOn(false); 
    setWalkReadOnly(false);
    setWalkQaLog([]); 
  }

  const openHistory = useCallback(async () => {
    setShowHistory(true);
    setLoadingHistory(true);
    const list = await fetchMySessions();
    // 依最後更新時間新到舊（後端已排序，這裡保底再排一次）
    list.sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
    setHistorySessions(list);
    setLoadingHistory(false);
  }, []);

  const handleLoadSession = useCallback(async (sid: string) => {
    setShowHistory(false);
    setIsLoading(true);
    setError("");

    const sessionDetail = await fetchSessionDetail(sid);
    setIsLoading(false);

    if (!sessionDetail) {
      setError("讀取對話紀錄失敗，請稍後再試");
      return;
    }

    const restoredMessages: ChatMessage[] = sessionDetail.display_messages.map((m) => ({
      id:          createId(),
      role:        m.role === "user" ? "user" : "ai",
      content:     m.content,
      attachments: m.attachments ?? [],
    }));

    walkAbortRef.current?.abort();

    // 從歷史對話載入的 phase 若是暫時性的 walk 中繼狀態，代表當初
    // 存檔時剛好卡在慢想過程中——interrupt 的可執行狀態不可能被還原
    // （見後端 chat.py 對冷啟動的降級說明），這裡一律校正回 socratic，
    // 避免使用者下一句輸入被誤路由到 /walk/answer。
    const sanitizedPhase =
      sessionDetail.phase === "walk_pending" || sessionDetail.phase === "walk_awaiting_answer"
        ? "socratic"
        : sessionDetail.phase;

    setSessionId(sessionDetail.session_id);
    setPhase(sanitizedPhase);
    setMessages(restoredMessages);
    setWalkPaths({});
    setPendingWalkQuestion(null);
    setWalkStreaming(false);
    setWalkResultAvailable(false);
    setWalkDockCollapsed(false);
    setBrainModeOn(false); 
    // ── 還原慢想（Random Walk）唯讀重播狀態 ──────────────────
    if (sessionDetail.walk_session && sessionDetail.walk_session.paths.length > 0) {
      setWalkPaths(walkSessionDtoToPathStates(sessionDetail.walk_session));
      setWalkQaLog(walkSessionDtoToQaLog(sessionDetail.walk_session));
      setWalkResultAvailable(true);
      setWalkReadOnly(true);

      if (sessionDetail.walk_pending_question) {
        setPendingWalkQuestion({
          pathId: sessionDetail.walk_pending_question.path_id,
          question: sessionDetail.walk_pending_question.question,
        });
      } else {
        setPendingWalkQuestion(null);
      }
    } else {
      setWalkReadOnly(false);
    }
    // ── 清空與舊聚焦狀態相關的暫存，避免殘留上一個 session 的畫面 ──
    setRCFoundationMap(new Map());
    setRCSemanticScores([]);
    expandFetchedRef.current = new Set();
    setSelectedNode(null);
    setDetail(null);
    setFoundationTree(null);
    setSemanticContext(null);
    setRecommendedFoundations([]);

    localStorage.setItem(SESSION_KEY, sessionDetail.session_id);
    localStorage.setItem(PHASE_KEY, sanitizedPhase);
    saveHistory(restoredMessages);

    // ── 補齊 Semantic 機制的畫面（GlobalBrain 點亮 / SemanticPanel），
    //    這些不含在 display_messages 裡，需要額外重新拉一次 ──
    fetchSemanticContextData(sessionDetail.session_id).then(setSemanticContext);
    fetchSemanticScores(sessionDetail.session_id).then(setRCSemanticScores);
    fetchFoundationRecommendations(sessionDetail.session_id).then(setRecommendedFoundations);
  }, []);

  const inputPlaceholder =
    phase === "walk_awaiting_answer" ? "回答慢想過程中的問題…" :
    phase === "main_agent"            ? "繼續詢問分析…" :
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
            {walkStreaming && (
              <>
                {" · "}
                <span style={{ color: "var(--teal)" }}>🧠 慢想中…</span>
              </>
            )}            
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {!authLoading && <AuthPanel />}
          <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.7rem", color: "var(--muted)" }}>
            <input
              type="checkbox"
              checked={walkEnabled}
              onChange={(e) => setWalkEnabled(e.target.checked)}
              disabled={isLoading || walkStreaming || phase !== "socratic"}
            />
            🧠 慢想模式
          </label>
          <button type="button" onClick={openHistory} disabled={isLoading}>
            歷史對話
          </button>
          <button type="button" onClick={resetConversation} disabled={isLoading}>
            開新對話
          </button>
        </div>
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
              messages.map((msg) => <ChatBubble key={msg.id} msg={msg} />)
            )}
            {isLoading && !walkStreaming && (
              <div className="meta"><span style={{ color: "var(--orange)", opacity: 0.7 }}>⬡ 思考中…</span></div>
            )}
          </div>

          <AttachmentTray attachments={allAttachments} />

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
            walkIllumination={walkIllumination}
            rcFoundationMap={rcFoundationMap}
            onNodeClick={handleNodeClick}
            dimBackground={brainModeOn}
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
          rcCategoryMap={rcCategoryMap}
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
            rcCategoryMap={rcCategoryMap}
            justUpdated={walkRankingJustUpdated}
            fetchFoundationTree={fetchFoundationTree}
            onLaunchSkill={handleLaunchSkill}
            fetchRCPaths={fetchRCPaths}
            isSubmitting={isLoading}
            width={personalSize.w}
            height={Math.max(personalSize.h - 32, 280)}
          />
        </div>

      </div>
      
      {/* ── 懸浮的慢想紀錄面板 ── */}
      {(walkStreaming || phase === "walk_awaiting_answer" || walkResultAvailable) && (
        <WalkDock
          paths={Object.values(walkPaths).sort((a, b) => a.pathId - b.pathId)}
          pathColors={walkPathColorMap}
          status={walkOverallStatus}
          pendingQuestion={pendingWalkQuestion}
          qaLog={walkQaLog}
          onSubmitAnswer={submitWalkAnswer}
          readOnly={walkReadOnly}
          collapsed={walkDockCollapsed}
          onToggle={() => setWalkDockCollapsed((v) => !v)}
          brainModeOn={brainModeOn}
          onToggleBrainMode={() => setBrainModeOn((v) => !v)}
          brainModeDisabled={!hasHighlightableNodes}
          onDismiss={() => {
            setWalkResultAvailable(false);
            setWalkPaths({});
            setWalkQaLog([]);
            setBrainModeOn(false);
          }}
        />
      )}

      {/* ── 歷史對話 modal ── */}
      {showHistory && (
        <div
          onClick={() => setShowHistory(false)}
          style={{
            position: "fixed", inset: 0, background: "rgba(3,1,8,0.72)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 1000, padding: 20,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 380, maxWidth: "100%", maxHeight: "70vh", overflowY: "auto",
              background: "var(--panel)", border: "1px solid var(--border)",
              borderRadius: 14, padding: 16,
              display: "flex", flexDirection: "column", gap: 8,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{
                fontSize: "0.62rem", letterSpacing: "0.18em", textTransform: "uppercase",
                color: "var(--orange)",
              }}>
                歷史對話
              </span>
              <button type="button" onClick={() => setShowHistory(false)} style={{ padding: "2px 8px" }}>
                ✕
              </button>
            </div>

            {loadingHistory ? (
              <div className="meta">載入中…</div>
            ) : historySessions.length === 0 ? (
              <div className="meta">尚無歷史對話紀錄</div>
            ) : (
              historySessions.map((s) => (
                <button
                  key={s.session_id}
                  type="button"
                  onClick={() => handleLoadSession(s.session_id)}
                  style={{
                    textAlign: "left", display: "flex", flexDirection: "column", gap: 2,
                    padding: "8px 10px",
                  }}
                >
                  <span style={{
                    fontSize: "0.7rem", color: "var(--text)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {s.summary
                      ? (s.summary.length > 40 ? s.summary.slice(0, 40) + "…" : s.summary)
                      : `session ${s.session_id.slice(0, 8)}…`}
                  </span>
                  <span style={{ fontSize: "0.6rem", color: "var(--muted)" }}>
                    {s.phase ?? ""}
                    {s.updated_at ? ` · ${new Date(s.updated_at).toLocaleString()}` : ""}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}


    </main>
  );
}
