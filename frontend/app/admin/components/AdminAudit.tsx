"use client";

import { useEffect, useState } from "react";
import { adminApi, type AuditEntry } from "../../../lib/adminApi";
import { formatDateTime } from "../../account/components/ui";
import { describePlan } from "./adminUi";

const ACTION_LABEL: Record<string, string> = { "plan.set": "設定等級" };

export default function AdminAudit({ onOpenUser, refreshKey }: { onOpenUser: (uid: string) => void; refreshKey: number }) {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [error, setError]     = useState("");

  useEffect(() => {
    let alive = true;
    adminApi.audit({ limit: 100 })
      .then((e) => { if (alive) { setEntries(e); setError(""); } })
      .catch((err) => { if (alive) { setError(err instanceof Error ? err.message : "讀取操作紀錄失敗"); setEntries((p) => p ?? []); } });
    return () => { alive = false; };
  }, [refreshKey]);

  if (entries === null) return <div className="acct-loading">載入操作紀錄…</div>;

  return (
    <div>
      <p className="acct-note" style={{ margin: "0 0 14px" }}>管理者每次調整會員等級都會留下紀錄，最新在前，最多顯示 100 筆。</p>
      {error && <p className="acct-error">{error}</p>}
      {entries.length === 0 ? (
        <div className="acct-empty"><strong>還沒有任何操作紀錄</strong></div>
      ) : (
        <ul className="acct-list">
          {entries.map((e) => (
            <li key={e.id} className="acct-row">
              <div className="acct-row-main">
                <span className="acct-row-title">
                  {ACTION_LABEL[e.action] ?? e.action}：{describePlan(e.before.plan, e.before.plan_expires_at)} → {describePlan(e.after.plan, e.after.plan_expires_at)}
                </span>
                <span className="acct-row-meta">
                  <span>{formatDateTime(e.at)}</span>
                  <span>操作者 {e.admin_email ?? e.admin_uid ?? "—"}</span>
                  <span>對象 {e.target_uid}</span>
                </span>
              </div>
              <div className="acct-row-actions">
                <button type="button" className="acct-btn quiet" onClick={() => onOpenUser(e.target_uid)}>檢視使用者</button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
