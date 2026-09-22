"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { accountApi, type SessionSummary } from "../../../lib/accountApi";
import { ConfirmButton, LS_HISTORY, LS_PHASE, LS_SESSION, compact, formatDateTime } from "./ui";

const PHASE_CHIP: Record<string, { text: string; cls: string }> = {
  main_agent:     { text: "分析中",   cls: "agent" },
  path_confirmed: { text: "分析中",   cls: "agent" },
  walk_pending:   { text: "慢想中斷", cls: "walk" },
};

function shorten(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export default function HistoryTab() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError]       = useState("");
  const [query, setQuery]       = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft]       = useState("");
  const [busy, setBusy]         = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setSessions(await accountApi.sessions());
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "讀取歷史對話失敗");
      setSessions((prev) => prev ?? []);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!sessions) return [];
    if (!q) return sessions;
    return sessions.filter((s) => `${s.title} ${s.summary}`.toLowerCase().includes(q));
  }, [sessions, query]);

  async function commitRename(sid: string) {
    const title = draft.trim();
    setBusy(sid);
    try {
      await accountApi.renameSession(sid, title);
      setSessions((prev) => prev?.map((s) => (s.session_id === sid ? { ...s, title } : s)) ?? prev);
      setRenaming(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "重新命名失敗");
    } finally {
      setBusy(null);
    }
  }

  async function remove(sid: string) {
    setBusy(sid);
    try {
      await accountApi.deleteSession(sid);
      setSessions((prev) => prev?.filter((s) => s.session_id !== sid) ?? prev);
      // 刪的是 dashboard 目前開著的那個對話：清掉本機暫存，避免回去還看到已刪除的內容。
      if (localStorage.getItem(LS_SESSION) === sid) {
        [LS_SESSION, LS_HISTORY, LS_PHASE].forEach((k) => localStorage.removeItem(k));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "刪除失敗");
    } finally {
      setBusy(null);
    }
  }

  if (sessions === null) return <div className="acct-loading">載入歷史對話…</div>;

  return (
    <div>
      <div className="acct-toolbar">
        <input
          className="acct-input"
          placeholder="搜尋對話名稱或需求摘要"
          aria-label="搜尋歷史對話"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="acct-note">共 {sessions.length} 個對話</span>
      </div>

      {error && <p className="acct-error">{error}</p>}

      {sessions.length === 0 ? (
        <div className="acct-empty">
          <strong>還沒有任何對話</strong>
          <span>到對話頁輸入你想分析的問題，對話會自動保存在這裡。</span>
          <Link href="/dashboard" className="acct-btn primary">開始第一個對話</Link>
        </div>
      ) : visible.length === 0 ? (
        <p className="acct-note">沒有符合「{query}」的對話。</p>
      ) : (
        <ul className="acct-list">
          {visible.map((s) => {
            const chip = s.phase ? PHASE_CHIP[s.phase] : undefined;
            const isBusy = busy === s.session_id;
            return (
              <li key={s.session_id} className="acct-row">
                <div className="acct-row-main">
                  {renaming === s.session_id ? (
                    <div className="acct-rename">
                      <input
                        className="acct-input"
                        value={draft}
                        maxLength={80}
                        autoFocus
                        aria-label="對話名稱"
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitRename(s.session_id);
                          if (e.key === "Escape") setRenaming(null);
                        }}
                      />
                      <button type="button" className="acct-btn primary" disabled={isBusy} onClick={() => commitRename(s.session_id)}>儲存</button>
                      <button type="button" className="acct-btn quiet" onClick={() => setRenaming(null)}>取消</button>
                    </div>
                  ) : (
                    <span className={`acct-row-title${s.title ? "" : " untitled"}`}>
                      {s.title || "未命名對話"}
                    </span>
                  )}
                  {s.summary && <span className="acct-row-sub" title={s.summary}>{shorten(s.summary, 110)}</span>}
                  <span className="acct-row-meta">
                    <span>{formatDateTime(s.updated_at)}</span>
                    <span>{s.total_tokens > 0 ? `${compact(s.total_tokens)} tokens` : "無用量紀錄"}</span>
                    {chip && <span className={`acct-chip ${chip.cls}`}>{chip.text}</span>}
                  </span>
                </div>

                <div className="acct-row-actions">
                  <Link href={`/dashboard?session=${encodeURIComponent(s.session_id)}`} className="acct-btn primary">開啟</Link>
                  <button
                    type="button"
                    className="acct-btn quiet"
                    disabled={isBusy}
                    onClick={() => { setRenaming(s.session_id); setDraft(s.title); }}
                  >
                    重新命名
                  </button>
                  <ConfirmButton
                    label="刪除"
                    disabled={isBusy}
                    className="acct-btn quiet"
                    onConfirm={() => remove(s.session_id)}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
