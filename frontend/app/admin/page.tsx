"use client";

import Link from "next/link";
import { Cormorant_Garamond } from "next/font/google";
import { useCallback, useEffect, useState } from "react";
import AuthPanel from "../components/AuthPanel";
import { useAuth } from "../components/AuthProvider";
import { useMyUsage } from "../../lib/useMyUsage";
import "../account/account.css";
import "./admin.css";
import AdminAudit from "./components/AdminAudit";
import AdminOverview from "./components/AdminOverview";
import AdminUsers from "./components/AdminUsers";
import AdminVips from "./components/AdminVips";
import UserDetail from "./components/UserDetail";

const serif = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["500", "600"],
  style: ["normal", "italic"],
  variable: "--font-serif",
});

const TABS = [
  { key: "overview", label: "總覽" },
  { key: "users",    label: "使用者" },
  { key: "vips",     label: "VIP 名單" },
  { key: "audit",    label: "操作紀錄" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

function tabFromHash(): TabKey {
  const h = window.location.hash.replace("#", "");
  return TABS.find((t) => t.key === h)?.key ?? "overview";
}

export default function AdminPage() {
  const { user, loading: authLoading } = useAuth();
  const { usage, error: usageError } = useMyUsage(!!user);

  const [tab, setTab]                 = useState<TabKey>("overview");
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  // 任何一處改了資料（例如調整等級）就 +1，讓其他分頁的列表重新載入。
  const [refreshKey, setRefreshKey]   = useState(0);
  const bump = useCallback(() => setRefreshKey((k) => k + 1), []);

  useEffect(() => {
    setTab(tabFromHash());
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const selectTab = useCallback((key: TabKey) => {
    setTab(key);
    window.history.replaceState(null, "", `#${key}`);
  }, []);

  const isAdmin = !!usage?.is_admin;
  const checking = authLoading || !user || (!usage && !usageError);

  return (
    <div className={`acct ${serif.variable}`}>
      <div className="acct-wrap">
        <div className="acct-top">
          <Link href="/account" className="acct-back">← 會員中心</Link>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {isAdmin && <span className="adm-mode">管理者模式</span>}
            {!authLoading && <AuthPanel />}
          </div>
        </div>

        <h1 className="acct-title">管理後台</h1>

        {checking ? (
          <div className="acct-loading">正在確認權限…</div>
        ) : !isAdmin ? (
          <div className="acct-empty" style={{ marginTop: 24 }}>
            <strong>你沒有權限查看這個頁面</strong>
            <span>
              {usageError
                ? `無法確認權限：${usageError}`
                : "管理後台僅限管理者。若你應該有權限，請確認已用管理者帳號登入。"}
            </span>
            <Link href="/dashboard" className="acct-btn primary">回到對話</Link>
          </div>
        ) : (
          <>
            <p className="acct-sub">查看全站用量、管理使用者的會員等級與 VIP 名單。所有等級變更都會留下操作紀錄。</p>

            <div className="acct-tabs" role="tablist" aria-label="管理後台">
              {TABS.map((t) => (
                <button
                  key={t.key} id={`atab-${t.key}`} type="button" role="tab" className="acct-tab"
                  aria-selected={tab === t.key} aria-controls={`apanel-${t.key}`}
                  onClick={() => selectTab(t.key)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            <div className="acct-panel" role="tabpanel" id={`apanel-${tab}`} aria-labelledby={`atab-${tab}`}>
              {tab === "overview" ? <AdminOverview onOpenUser={setSelectedUid} />
                : tab === "users"  ? <AdminUsers selectedUid={selectedUid} onSelect={setSelectedUid} refreshKey={refreshKey} />
                : tab === "vips"   ? <AdminVips onOpenUser={setSelectedUid} onChanged={bump} refreshKey={refreshKey} />
                : <AdminAudit onOpenUser={setSelectedUid} refreshKey={refreshKey} />}
            </div>

            {selectedUid && (
              <UserDetail uid={selectedUid} onClose={() => setSelectedUid(null)} onChanged={bump} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
