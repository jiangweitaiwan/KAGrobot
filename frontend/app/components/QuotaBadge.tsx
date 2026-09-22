"use client";

import Link from "next/link";
import type { MyUsage } from "../../lib/accountApi";

/**
 * dashboard header 的額度指示。點擊進入會員中心。
 * 未啟用限額（enforced=false）時不使用警示色，避免使用者誤以為快被擋。
 */
export default function QuotaBadge({ usage }: { usage: MyUsage | null }) {
  if (!usage) return null;

  const ratio = Math.min(1, usage.ratio);
  const pct = Math.round(ratio * 100);
  const state = !usage.enforced ? "ok" : usage.exceeded ? "bad" : usage.warn ? "warn" : "ok";
  const color = state === "bad" ? "#c4574a" : state === "warn" ? "var(--orange)" : "var(--teal)";
  const label = usage.period === "lifetime" ? "試用" : "本月";

  return (
    <Link
      href="/account#usage"
      title={
        usage.enforced
          ? `${label}額度已使用 ${usage.used.toLocaleString()} / ${usage.limit.toLocaleString()} tokens`
          : `${label}已使用 ${usage.used.toLocaleString()} tokens（目前僅計量，尚未限額）`
      }
      style={{
        display: "flex", alignItems: "center", gap: 8, textDecoration: "none",
        fontSize: "0.7rem", color: "var(--muted)",
        border: "1px solid var(--border)", borderRadius: 8, padding: "5px 10px",
      }}
    >
      <span>{label} {pct}%</span>
      <span aria-hidden="true" style={{ width: 44, height: 4, borderRadius: 2, background: "rgba(201,162,39,0.14)", overflow: "hidden" }}>
        <span style={{ display: "block", width: `${pct}%`, height: "100%", background: color }} />
      </span>
    </Link>
  );
}
