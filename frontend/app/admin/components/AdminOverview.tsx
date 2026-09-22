"use client";

import { useEffect, useState } from "react";
import { adminApi, asIndexRequired, shiftPeriod, type AdminOverview as Overview, type TopUser } from "../../../lib/adminApi";
import type { DailyUsageItem } from "../../../lib/accountApi";
import DailyBars from "../../account/components/DailyBars";
import { FEATURE_LABEL, PLAN_LABEL, fmt } from "../../account/components/ui";
import { PlanChip, userLabel } from "./adminUi";

const PLAN_ORDER = ["guest", "normal", "vip"] as const;

function periodText(key: string) { return `${key.slice(0, 4)} 年 ${Number(key.slice(4))} 月`; }

export default function AdminOverview({ onOpenUser }: { onOpenUser: (uid: string) => void }) {
  const [which, setWhich]     = useState<"current" | "previous">("current");
  const [currentKey, setCurrentKey] = useState<string | null>(null);   // 由後端回傳，避免用瀏覽器時區猜「本月」
  const [data, setData]       = useState<Overview | null>(null);
  const [daily, setDaily]     = useState<DailyUsageItem[] | null>(null);
  const [top, setTop]         = useState<TopUser[] | null>(null);
  const [topNote, setTopNote] = useState("");
  const [error, setError]     = useState("");

  useEffect(() => {
    let alive = true;
    const period = which === "previous" && currentKey ? shiftPeriod(currentKey, -1) : undefined;
    setData(null); setTop(null); setTopNote(""); setError("");

    adminApi.overview(period)
      .then((d) => { if (!alive) return; setData(d); if (which === "current") setCurrentKey(d.period); })
      .catch((e) => alive && setError(e instanceof Error ? e.message : "讀取總覽失敗"));

    // 排行榜獨立請求：缺少索引時只有這一塊失敗，總覽其餘部分照常顯示。
    adminApi.topUsers(period, 10)
      .then((u) => alive && setTop(u))
      .catch((e) => {
        if (!alive) return;
        setTop([]);
        setTopNote(asIndexRequired(e) ?? (e instanceof Error ? e.message : "讀取排行失敗"));
      });
    return () => { alive = false; };
  }, [which, currentKey]);

  useEffect(() => {
    let alive = true;
    adminApi.overviewDaily(30).then((d) => alive && setDaily(d)).catch(() => alive && setDaily([]));
    return () => { alive = false; };
  }, []);

  const features = data ? Object.entries(data.by_feature).filter(([, v]) => v.total_tokens > 0).sort((a, b) => b[1].total_tokens - a[1].total_tokens) : [];
  const featureMax = features[0]?.[1].total_tokens ?? 1;
  // 已知等級依固定順序，未預期的等級（例如舊版本寫入的 "unknown"）排在後面，不丟資料。
  const knownPlans: string[] = [...PLAN_ORDER];
  const plans = data
    ? [...knownPlans.filter((p) => data.by_plan[p]), ...Object.keys(data.by_plan).filter((p) => !knownPlans.includes(p))]
    : [];
  const planMax = data ? Math.max(1, ...Object.values(data.by_plan).map((v) => v.total_tokens)) : 1;

  return (
    <div>
      <div className="acct-toolbar">
        <div className="acct-range" role="group" aria-label="統計月份">
          <button type="button" aria-pressed={which === "current"} onClick={() => setWhich("current")}>本月</button>
          <button type="button" aria-pressed={which === "previous"} disabled={!currentKey} onClick={() => setWhich("previous")}>上月</button>
        </div>
        {data && <span className="acct-note">{periodText(data.period)}</span>}
      </div>

      {error && <p className="acct-error">{error}</p>}
      {!data && !error && <div className="acct-loading">載入總覽…</div>}

      {data && (
        <>
          <dl className="adm-stats">
            <div><dt>總 token</dt><dd>{fmt(data.total_tokens)}</dd></div>
            <div><dt>對話輪數</dt><dd>{fmt(data.turns)}</dd></div>
            <div><dt>模型呼叫</dt><dd>{fmt(data.calls)}</dd></div>
            <div><dt>有效 VIP</dt><dd>{fmt(data.vip_active)}</dd></div>
          </dl>

          {data.total_tokens === 0 && (
            <p className="acct-note" style={{ margin: "0 0 8px" }}>
              這個月還沒有用量紀錄。全站彙總是從這個版本上線後才開始累計，更早的用量不會出現在這裡。
            </p>
          )}

          <div className="adm-two">
            <section className="acct-section">
              <h2>依等級</h2>
              <p className="lede">用量發生當下的會員等級。</p>
              {plans.length === 0 ? <p className="acct-note">沒有資料。</p> : plans.map((p) => {
                const v = data.by_plan[p];
                return (
                  <div className="acct-feature" key={p}>
                    <span className="name">{PLAN_LABEL[p as keyof typeof PLAN_LABEL] ?? (p === "unknown" ? "未分類" : p)}</span>
                    <div className="acct-meter" aria-hidden="true"><span style={{ width: `${(v.total_tokens / planMax) * 100}%` }} /></div>
                    <span className="val">{fmt(v.total_tokens)}</span>
                  </div>
                );
              })}
            </section>

            <section className="acct-section">
              <h2>依功能</h2>
              <p className="lede">哪個功能最耗 token，決定額度該怎麼訂。</p>
              {features.length === 0 ? <p className="acct-note">沒有資料。</p> : features.map(([name, v]) => (
                <div className="acct-feature" key={name}>
                  <span className="name">{FEATURE_LABEL[name] ?? name}</span>
                  <div className="acct-meter" aria-hidden="true"><span style={{ width: `${(v.total_tokens / featureMax) * 100}%` }} /></div>
                  <span className="val">{fmt(v.total_tokens)}</span>
                </div>
              ))}
            </section>
          </div>
        </>
      )}

      <section className="acct-section">
        <h2>近 30 天趨勢</h2>
        {daily ? <DailyBars days={daily} /> : <div className="acct-loading">載入中…</div>}
      </section>

      <section className="acct-section">
        <h2>用量最高的使用者</h2>
        {top === null ? <div className="acct-loading">載入中…</div>
          : topNote ? <p className="acct-callout bad">{topNote}</p>
          : top.length === 0 ? <p className="acct-note">這個月還沒有使用者用量。</p>
          : (
            <div className="adm-table-wrap">
              <table className="adm-table">
                <thead><tr><th scope="col">使用者</th><th scope="col">等級</th><th scope="col" className="num">Token</th><th scope="col" className="num">輪數</th></tr></thead>
                <tbody>
                  {top.map((u) => (
                    <tr key={u.uid} className="clickable" onClick={() => onOpenUser(u.uid)}>
                      <td><button type="button" className="adm-link" onClick={(e) => { e.stopPropagation(); onOpenUser(u.uid); }}>{userLabel({ ...u, is_anonymous: u.plan === "guest" })}</button></td>
                      <td><PlanChip plan={u.plan} /></td>
                      <td className="num">{fmt(u.total_tokens)}</td>
                      <td className="num">{fmt(u.turns)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </section>
    </div>
  );
}
