"use client";

import { useEffect, useMemo, useState } from "react";
import { accountApi, type DailyUsageItem, type MyUsage } from "../../../lib/accountApi";
import DailyBars from "./DailyBars";
import { FEATURE_LABEL, fmt } from "./ui";

const RANGES = [7, 30, 90] as const;

export default function UsageTab({ usage }: { usage: MyUsage }) {
  const [days, setDays]     = useState<(typeof RANGES)[number]>(30);
  const [daily, setDaily]   = useState<DailyUsageItem[] | null>(null);
  const [error, setError]   = useState("");

  useEffect(() => {
    let alive = true;
    setDaily(null);
    accountApi.dailyUsage(days)
      .then((d) => { if (alive) { setDaily(d); setError(""); } })
      .catch((err) => { if (alive) setError(err instanceof Error ? err.message : "讀取每日用量失敗"); });
    return () => { alive = false; };
  }, [days]);

  const b = usage.breakdown;
  const isLife = usage.period === "lifetime";

  const features = useMemo(
    () => Object.entries(b.by_feature)
      .filter(([, v]) => v.total_tokens > 0)
      .sort((a, c) => c[1].total_tokens - a[1].total_tokens),
    [b.by_feature],
  );
  const featureMax = features[0]?.[1].total_tokens ?? 1;
  const ioTotal = b.input_tokens + b.output_tokens;

  const stats = useMemo(() => {
    if (!daily) return null;
    const total = daily.reduce((s, d) => s + d.total_tokens, 0);
    const peak  = daily.reduce((m, d) => (d.total_tokens > m.total_tokens ? d : m), daily[0]);
    return { total, avg: total / daily.length, peak };
  }, [daily]);

  return (
    <div>
      <div className="acct-usage-head">
        <div>
          <div className="acct-big">{fmt(usage.used)}<small>tokens</small></div>
          <span className="acct-note">{isLife ? "試用期間累計" : `本月（${usage.period_key?.slice(0, 4)}/${usage.period_key?.slice(4)}）`}</span>
        </div>
        <div>
          <div className="acct-big">{fmt(b.turns)}<small>輪對話</small></div>
          <span className="acct-note">{fmt(b.calls)} 次模型呼叫</span>
        </div>
      </div>

      {ioTotal > 0 && (
        <>
          <div className="acct-split" role="img" aria-label={`輸入 ${fmt(b.input_tokens)}，輸出 ${fmt(b.output_tokens)}`}>
            <span style={{ width: `${(b.input_tokens / ioTotal) * 100}%` }} />
            <span style={{ width: `${(b.output_tokens / ioTotal) * 100}%` }} />
          </div>
          <div className="acct-legend">
            <span><i style={{ background: "var(--a-ok)" }} />輸入 {fmt(b.input_tokens)}</span>
            <span><i style={{ background: "var(--orange)" }} />輸出 {fmt(b.output_tokens)}</span>
            {b.reasoning_tokens > 0 && <span>模型思考 {fmt(b.reasoning_tokens)}</span>}
          </div>
        </>
      )}

      <section className="acct-section" style={{ marginTop: 22 }}>
        <h2>用在哪裡</h2>
        <p className="lede">
          依功能分解{isLife ? "試用期間" : "本月"}的用量。額度以模型回報的 token 數計算，含模型內部思考。
        </p>
        {features.length === 0 ? (
          <p className="acct-note">還沒有用量。開始一段對話後，這裡會顯示每個功能用了多少。</p>
        ) : (
          features.map(([name, v]) => (
            <div className="acct-feature" key={name}>
              <span className="name">{FEATURE_LABEL[name] ?? name}</span>
              <div className="acct-meter" aria-hidden="true"><span style={{ width: `${(v.total_tokens / featureMax) * 100}%` }} /></div>
              <span className="val">{fmt(v.total_tokens)}</span>
            </div>
          ))
        )}
      </section>

      <section className="acct-section">
        <div className="acct-toolbar" style={{ marginBottom: 6 }}>
          <h2 style={{ margin: 0 }}>每日用量</h2>
          <div className="acct-range" role="group" aria-label="時間範圍">
            {RANGES.map((r) => (
              <button key={r} type="button" aria-pressed={days === r} onClick={() => setDays(r)}>{r} 天</button>
            ))}
          </div>
        </div>

        {error ? (
          <p className="acct-error">{error}</p>
        ) : !daily || !stats ? (
          <div className="acct-loading">載入中…</div>
        ) : (
          <>
            <p className="lede" style={{ marginBottom: 10 }}>
              近 {days} 天共 {fmt(stats.total)} tokens，日均 {fmt(stats.avg)}
              {stats.peak.total_tokens > 0 && <>；用量最高是 {stats.peak.date.slice(5).replace("-", "/")}（{fmt(stats.peak.total_tokens)}）</>}。
            </p>
            <DailyBars days={daily} />
          </>
        )}
        <p className="acct-note" style={{ marginTop: 10 }}>日期依伺服器設定的時區切分（預設為台北時間）。</p>
      </section>
    </div>
  );
}
