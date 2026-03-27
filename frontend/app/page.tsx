"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Role = "user" | "ai";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
};

type ChatResponse = {
  reply: string;
  phase: string;
  socrates_phase?: string | null;
};

const SESSION_KEY = "kagrobot_session_id";
const HISTORY_KEY = "kagrobot_history";
const PHASE_KEY = "kagrobot_phase";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function Page() {
  const [sessionId, setSessionId] = useState("");
  const [phase, setPhase] = useState("socratic");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const storedSession = localStorage.getItem(SESSION_KEY);
    const sid = storedSession || createSessionId();
    if (!storedSession) {
      localStorage.setItem(SESSION_KEY, sid);
    }

    const storedPhase = localStorage.getItem(PHASE_KEY) || "socratic";
    const rawHistory = localStorage.getItem(HISTORY_KEY);
    const history = rawHistory ? (JSON.parse(rawHistory) as ChatMessage[]) : [];

    setSessionId(sid);
    setPhase(storedPhase);
    setMessages(history);
  }, []);

  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, isLoading]);

  const canSend = useMemo(() => !!input.trim() && !isLoading && !!sessionId, [input, isLoading, sessionId]);

  async function typewriterAppend(fullText: string) {
    const aiId = createSessionId();
    setMessages((prev) => [...prev, { id: aiId, role: "ai", content: "" }]);

    for (let i = 1; i <= fullText.length; i += 1) {
      const chunk = fullText.slice(0, i);
      setMessages((prev) => prev.map((msg) => (msg.id === aiId ? { ...msg, content: chunk } : msg)));
      await wait(10);
    }
  }

  async function submitMessage(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;

    if (!API_BASE) {
      setError("尚未設定 NEXT_PUBLIC_API_BASE_URL，請先設定 Vercel 環境變數。");
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
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          phase,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error ${response.status}`);
      }

      const data = (await response.json()) as ChatResponse;
      const nextPhase = data.phase || "socratic";
      setPhase(nextPhase);
      localStorage.setItem(PHASE_KEY, nextPhase);

      await typewriterAppend(data.reply || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "請求失敗，請稍後重試");
      setMessages((prev) => [...prev, { id: createSessionId(), role: "ai", content: "抱歉，連線失敗，請稍後再試。" }]);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
  }, [sessionId, messages]);

  function resetConversation() {
    const sid = createSessionId();
    setSessionId(sid);
    setPhase("socratic");
    setMessages([]);
    setError("");
    localStorage.setItem(SESSION_KEY, sid);
    localStorage.setItem(PHASE_KEY, "socratic");
    localStorage.removeItem(HISTORY_KEY);
  }

  return (
    <main>
      <header>
        <div>
          <h1>KAGrobot 對話機器人</h1>
          <div className="session">session_id: {sessionId || "初始化中..."}</div>
          <div className="meta">phase: {phase}</div>
        </div>
        <button type="button" onClick={resetConversation} disabled={isLoading}>
          開新對話
        </button>
      </header>

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
        {isLoading && <div className="meta">AI 正在思考中...</div>}
      </div>

      <div>
        <form onSubmit={submitMessage}>
          <textarea
            placeholder="輸入你的問題..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={!canSend}>
            送出
          </button>
        </form>
        {error && <div className="status">錯誤：{error}</div>}
      </div>
    </main>
  );
}
