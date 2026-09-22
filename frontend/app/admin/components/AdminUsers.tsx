"use client";

import { useCallback, useEffect, useState } from "react";
import { adminApi, type AdminUserRow } from "../../../lib/adminApi";
import { fmt, formatDateTime } from "../../account/components/ui";
import { PlanChip, userLabel } from "./adminUi";

export default function AdminUsers({
  selectedUid, onSelect, refreshKey,
}: { selectedUid: string | null; onSelect: (uid: string) => void; refreshKey: number }) {
  const [input, setInput]       = useState("");
  const [query, setQuery]       = useState("");           // 已送出的搜尋
  const [guests, setGuests]     = useState(false);
  const [rows, setRows]         = useState<AdminUserRow[] | null>(null);
  const [nextToken, setNext]    = useState<string | null>(null);
  const [loadingMore, setMore]  = useState(false);
  const [error, setError]       = useState("");

  const load = useCallback(async (opts: { q: string; guests: boolean; token?: string | null }) => {
    try {
      const res = await adminApi.users({ q: opts.q, includeGuests: opts.guests, pageToken: opts.token, limit: 30 });
      setRows((prev) => (opts.token && prev ? [...prev, ...res.users] : res.users));
      setNext(res.next_page_token);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "讀取使用者失敗");
      setRows((prev) => prev ?? []);
    }
  }, []);

  // 條件（搜尋 / 含訪客）改變，或外部通知資料已變動（例如改了等級）→ 從第一頁重載。
  useEffect(() => { setRows(null); load({ q: query, guests }); }, [query, guests, refreshKey, load]);

  async function more() {
    if (!nextToken) return;
    setMore(true);
    await load({ q: query, guests, token: nextToken });
    setMore(false);
  }

  const searching = query !== "";

  return (
    <div>
      <div className="acct-toolbar">
        <div className="adm-search">
          <input
            className="acct-input"
            placeholder="輸入完整 Email 或 UID"
            aria-label="搜尋使用者"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") setQuery(input.trim()); }}
          />
          <button type="button" className="acct-btn primary" onClick={() => setQuery(input.trim())}>搜尋</button>
          {searching && <button type="button" className="acct-btn quiet" onClick={() => { setInput(""); setQuery(""); }}>清除</button>}
        </div>
        <label className="adm-check">
          <input type="checkbox" checked={guests} onChange={(e) => setGuests(e.target.checked)} disabled={searching} />
          包含訪客
        </label>
      </div>
      <p className="acct-note" style={{ margin: "0 0 12px" }}>
        Firebase 只支援完整 Email 或 UID 的精確查找，不支援模糊搜尋。
      </p>

      {error && <p className="acct-error">{error}</p>}
      {rows === null ? <div className="acct-loading">載入使用者…</div>
        : rows.length === 0 ? (
          <div className="acct-empty">
            <strong>{searching ? `找不到「${query}」` : "沒有符合條件的使用者"}</strong>
            {searching && <span>請確認 Email 拼寫，或改用 UID。</span>}
          </div>
        ) : (
          <>
            <div className="adm-table-wrap">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th scope="col">使用者</th><th scope="col">等級</th>
                    <th scope="col" className="num">本月 token</th><th scope="col">最近登入</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((u) => (
                    <tr key={u.uid} className={`clickable${u.uid === selectedUid ? " selected" : ""}`} onClick={() => onSelect(u.uid)}>
                      <td>
                        <button type="button" className="adm-link" onClick={(e) => { e.stopPropagation(); onSelect(u.uid); }}>
                          {userLabel(u)}
                        </button>
                        {u.is_admin && <span className="adm-chip admin">管理者</span>}
                        {u.display_name && u.email && <span className="adm-sub">{u.display_name}</span>}
                      </td>
                      <td>
                        <PlanChip plan={u.plan} />
                        {u.plan === "vip" && u.plan_expires_at && <span className="adm-sub">至 {new Date(u.plan_expires_at).toLocaleDateString("zh-TW")}</span>}
                      </td>
                      <td className="num">{u.plan === "guest" ? `${fmt(u.lifetime_tokens)}（累計）` : fmt(u.month_tokens)}</td>
                      <td className="muted">{formatDateTime(u.last_sign_in_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {nextToken && !searching && (
              <div style={{ textAlign: "center", marginTop: 14 }}>
                <button type="button" className="acct-btn" onClick={more} disabled={loadingMore}>{loadingMore ? "載入中…" : "顯示更多"}</button>
              </div>
            )}
          </>
        )}
    </div>
  );
}
