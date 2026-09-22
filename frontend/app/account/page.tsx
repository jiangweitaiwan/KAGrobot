"use client";

import Link from "next/link";
import { Cormorant_Garamond } from "next/font/google";
import { useCallback, useEffect, useState } from "react";
import AuthPanel from "../components/AuthPanel";
import { useAuth } from "../components/AuthProvider";
import { useMyUsage } from "../../lib/useMyUsage";
import BrainTab from "./components/BrainTab";
import HistoryTab from "./components/HistoryTab";
import OverviewTab from "./components/OverviewTab";
import UsageTab from "./components/UsageTab";
import "./account.css";

// 與首頁相同的顯示字型（首頁在 .intro-root 上掛同名變數），這裡掛在 .acct 上。
const serif = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["500", "600"],
  style: ["normal", "italic"],
  variable: "--font-serif",
});

const TABS = [
  { key: "overview", label: "概覽" },
  { key: "history",  label: "歷史對話" },
  { key: "brain",    label: "Personal Brain" },
  { key: "usage",    label: "Token 用量" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

function tabFromHash(): TabKey {
  const h = window.location.hash.replace("#", "");
  return TABS.find((t) => t.key === h)?.key ?? "overview";
}

export default function AccountPage() {
  const { user, loading: authLoading } = useAuth();
  const { usage, error: usageError, refresh } = useMyUsage(!!user);

  // 初始值固定 "overview"，掛載後才讀 hash，避免 SSR / hydration 不一致。
  const [tab, setTab] = useState<TabKey>("overview");

  useEffect(() => {
    setTab(tabFromHash());
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const selectTab = useCallback((key: TabKey) => {
    setTab(key);
    window.history.replaceState(null, "", `#${key}`);
    if (key === "overview" || key === "usage") refresh();   // 切回這兩頁時順手更新額度
  }, [refresh]);

  const ready = !authLoading && !!user;

  return (
    <div className={`acct ${serif.variable}`}>
      <div className="acct-wrap">
        <div className="acct-top">
          <Link href="/dashboard" className="acct-back">← 回到對話</Link>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {usage?.is_admin && <Link href="/admin" className="acct-back">管理後台</Link>}
            {!authLoading && <AuthPanel />}
          </div>
        </div>

        <h1 className="acct-title">會員中心</h1>
        <p className="acct-sub">查看你的方案與額度、管理歷史對話與 Personal Brain，以及追蹤 token 用量。</p>

        <div className="acct-tabs" role="tablist" aria-label="會員中心">
          {TABS.map((t) => (
            <button
              key={t.key}
              id={`tab-${t.key}`}
              type="button"
              role="tab"
              className="acct-tab"
              aria-selected={tab === t.key}
              aria-controls={`panel-${t.key}`}
              onClick={() => selectTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="acct-panel" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
          {!ready ? (
            <div className="acct-loading">正在確認登入狀態…</div>
          ) : tab === "history" ? (
            <HistoryTab />
          ) : tab === "brain" ? (
            <BrainTab uid={user.uid} usage={usage} />
          ) : !usage ? (
            usageError
              ? <p className="acct-error">{usageError}　<button type="button" className="acct-btn" onClick={refresh}>重試</button></p>
              : <div className="acct-loading">載入方案與用量…</div>
          ) : tab === "usage" ? (
            <UsageTab usage={usage} />
          ) : (
            <OverviewTab usage={usage} />
          )}
        </div>
      </div>
    </div>
  );
}
