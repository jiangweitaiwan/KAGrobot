"use client";

import type { PlanKey } from "../../../lib/accountApi";
import { PLAN_LABEL } from "../../account/components/ui";

export const DAY_MS = 86_400_000;

export function userLabel(u: { email: string | null; display_name: string | null; uid: string; is_anonymous?: boolean }): string {
  if (u.email) return u.email;
  if (u.display_name) return u.display_name;
  return `${u.is_anonymous ? "訪客" : "未設定 Email"} ${u.uid.slice(0, 8)}…`;
}

/** "2026-12-31" → 當天本地時間 23:59:59 的 unix ms。無效輸入回 null。 */
export function endOfLocalDay(dateStr: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr);
  if (!m) return null;
  const t = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), 23, 59, 59).getTime();
  return Number.isFinite(t) ? t : null;
}

export function toDateInput(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** 剩餘整天數（向上取整）；已過期為負或 0。 */
export function daysLeft(ms: number, now = Date.now()): number {
  return Math.ceil((ms - now) / DAY_MS);
}

/** 延長 VIP：從「目前到期日」與「現在」較晚者起算。 */
export function extendedExpiry(currentExpiry: number | null, days: number, now = Date.now()): number {
  return Math.max(currentExpiry ?? 0, now) + days * DAY_MS;
}

export function describePlan(plan: string | undefined, exp: number | null | undefined): string {
  const label = PLAN_LABEL[(plan as PlanKey) ?? "normal"] ?? plan ?? "—";
  if (plan === "vip") return exp ? `${label}（至 ${new Date(exp).toLocaleDateString("zh-TW")}）` : `${label}（不過期）`;
  return label;
}

export function PlanChip({ plan, expired }: { plan: string; expired?: boolean }) {
  const cls = expired ? "adm-chip expired" : plan === "vip" ? "adm-chip vip" : plan === "guest" ? "adm-chip guest" : "adm-chip";
  return <span className={cls}>{expired ? "已到期" : PLAN_LABEL[plan as PlanKey] ?? plan}</span>;
}
